# coding:utf-8
import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel


class ConditionalLayerNorm(nn.Module):
    def __init__(self,
                 hidden_size,
                 eps=1e-12):
        super().__init__()

        self.eps = eps

        self.weight = nn.Parameter(torch.Tensor(hidden_size))
        self.bias = nn.Parameter(torch.Tensor(hidden_size))

        self.weight_dense = nn.Linear(hidden_size, hidden_size, bias=True)
        self.bias_dense = nn.Linear(hidden_size, hidden_size, bias=True)

        self.init_weight()

    def init_weight(self):
        """
        此处初始化的作用是在训练开始阶段不让 conditional layer norm 起作用
        """
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)

        nn.init.zeros_(self.weight_dense.weight)
        nn.init.zeros_(self.bias_dense.weight)

    def forward(self, inputs, cond=None):
        assert cond is not None, 'Conditional tensor need to input when use conditional layer norm'

        cond = torch.unsqueeze(cond, 2)  # (b, n, 1, h)

        weight = self.weight_dense(cond) + self.weight  # (b, 1, h)
        bias = self.bias_dense(cond) + self.bias  # (b, 1, h)

        mean = torch.mean(inputs, dim=-1, keepdim=True)  # （b, s, 1）
        outputs = inputs - mean  # (b, s, h)
        variance = torch.mean(outputs ** 2, dim=-1, keepdim=True)
        std = torch.sqrt(variance + self.eps)  # (b, s, 1)
        outputs = outputs / std  # (b, s, h)
        outputs = torch.unsqueeze(outputs, 1) # (b, 1, s, h)

        outputs = outputs * weight + bias

        return outputs


class EventExtractModel(nn.Module):

    def __init__(self,
                 bert_dir,
                 dropout_prob=0.1):
        super(EventExtractModel, self).__init__()

        assert os.path.exists(bert_dir), 'pretrained bert file does not exist'

        self.bert_module = BertModel.from_pretrained(bert_dir)
        self.hidden_size = self.bert_module.config.hidden_size
        self.num_attention_heads = self.bert_module.config.num_attention_heads

        self.triggers_classifier = nn.Sequential(
            nn.Linear(self.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(256, 3)
        )

        self.conditional_layernorm = ConditionalLayerNorm(self.hidden_size)
        self.events_tags_classifier = nn.Sequential(
            nn.Linear(self.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(256, 9)
        )

        # self.events_transform = nn.MultiheadAttention(self.hidden_size, self.num_attention_heads, batch_first=True)
        self.events_relations_classifier = nn.Sequential(
            nn.Linear(self.hidden_size*2, 256),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(256, 7)
        )

    def calc_triggers_tags(self, seq_out):

        triggers_logits = self.triggers_classifier(seq_out)

        return triggers_logits
    
    def calc_events_tags(self, seq_out, triggers_hidden):

        events_tags_hidden = self.conditional_layernorm(seq_out, triggers_hidden)
        events_tags_logits = self.events_tags_classifier(events_tags_hidden)

        return events_tags_logits
    
    def calc_events_relations(self, events_hidden):
        b, n, h = events_hidden.shape
        events_a = events_hidden.unsqueeze(2).expand(b, n, n, h)
        events_b = events_hidden.unsqueeze(1).expand(b, n, n, h)
        events_interaction = torch.cat([events_a, events_b], dim=-1)

        events_relations_logits = self.events_relations_classifier(events_interaction)
        return events_relations_logits

    def get_events_hidden(self, seq_out, triggers_hidden):
        return triggers_hidden

    def get_triggers_hidden(self, seq_out, triggers_pos, triggers_mask):

        triggers_pos = torch.reshape(triggers_pos, [triggers_pos.shape[0], -1, 1])
        triggers_pos = triggers_pos.expand(triggers_pos.size(0), triggers_pos.size(1), seq_out.size(2))
        triggers_hidden = torch.gather(seq_out, dim=1, index=triggers_pos)
        triggers_hidden = torch.reshape(triggers_hidden, list(triggers_mask.shape) + [-1])
        triggers_hidden = triggers_hidden * triggers_mask.unsqueeze(dim=3)
        triggers_hidden = torch.sum(triggers_hidden, dim=2) / (torch.sum(triggers_mask, dim=2, keepdim=True) + 1e-6)

        return triggers_hidden

    def get_seq_hidden(self, input_ids, attention_mask, token_type_ids=None):

        bert_out = self.bert_module(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        return bert_out.last_hidden_state, bert_out.pooler_output
    
    def forward(self, input_ids, attention_mask, triggers_pos, triggers_mask):
        seq_out, _ = self.get_seq_hidden(input_ids, attention_mask)
        triggers_logit = self.calc_triggers_tags(seq_out)

        triggers_hidden = self.get_triggers_hidden(seq_out, triggers_pos, triggers_mask)   
        events_tags_logit = self.calc_events_tags(seq_out, triggers_hidden)

        events_hidden = self.get_events_hidden(seq_out, triggers_hidden)
        events_relations_logit = self.calc_events_relations(events_hidden)

        return triggers_logit, events_tags_logit, events_relations_logit

    def calc_loss(self, triggers_logit, triggers_label,
                events_tags_logit, events_tags_label,
                events_relations_logit, events_relations_label,
                attention_mask, events_tags_mask, events_mask):
        
        triggers_logit = torch.reshape(triggers_logit, [-1, 3])
        triggers_label = torch.reshape(triggers_label, [-1])
        triggers_mask = torch.reshape(attention_mask, [-1])
        triggers_loss = F.cross_entropy(triggers_logit, triggers_label, reduction='none')
        triggers_loss = torch.sum(triggers_loss*triggers_mask)/(torch.sum(triggers_mask) + 1e-6)
        
        events_tags_logit = torch.reshape(events_tags_logit, [-1, 9])
        events_tags_label = torch.reshape(events_tags_label, [-1])
        events_tags_mask = torch.reshape(events_tags_mask, [-1])
        events_tags_loss = F.cross_entropy(events_tags_logit, events_tags_label, reduction='none')
        events_tags_loss = torch.sum(events_tags_loss*events_tags_mask)/(torch.sum(events_tags_mask) + 1e-6)

        events_relations_mask = torch.einsum('bn,bl->bnl', events_mask, events_mask)
        events_relations_logit = torch.reshape(events_relations_logit, [-1, 7])
        events_relations_label = torch.reshape(events_relations_label, [-1])
        events_relations_mask = torch.reshape(events_relations_mask, [-1])
        events_relations_loss = F.cross_entropy(events_relations_logit, events_relations_label, reduction='none')
        events_relations_loss = torch.sum(events_relations_loss*events_relations_mask)/(torch.sum(events_relations_mask) + 1e-6)

        return triggers_loss, events_tags_loss, events_relations_loss


        

        





    