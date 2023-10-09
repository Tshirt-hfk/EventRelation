# coding:utf-8
import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel

class BasicAttentionLayer(nn.Module):
    
    def __init__(self, embed_dim, num_heads, k_dim=None, v_dim=None, bias=True, dropout=0.0, q_transform=False):
        super(BasicAttentionLayer, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.k_dim = embed_dim // num_heads if k_dim is None else k_dim
        self.v_dim = embed_dim // num_heads if v_dim is None else v_dim
        self.bias = bias
        self.scaling = self.k_dim ** -0.5
        self.dropout = dropout

        if q_transform:
            self.q_transform = nn.Linear(embed_dim, self.num_heads * self.v_dim, bias=bias)
        else:
            self.q_transform = None
        self.q_proj = nn.Linear(embed_dim, self.num_heads * self.k_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, self.num_heads * self.k_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, self.num_heads * self.v_dim, bias=bias)

        self.init_weight()

    def init_weight(self):
        if self.q_transform is not None:
            nn.init.xavier_uniform_(self.q_transform.weight)
            nn.init.constant_(self.q_transform.bias, 0.)

        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.q_proj.weight)

        if self.bias:
            nn.init.constant_(self.q_proj.bias, 0.)
            nn.init.constant_(self.k_proj.bias, 0.)
            nn.init.constant_(self.v_proj.bias, 0.)

    def forward(self, query, key, value, mask=None):
        # bsz, len, embed_dim
        bsz = query.size(0)
        src_len = query.size(1)
        trg_len = value.size(1)

        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        q *= self.scaling
        q = q.transpose(0, 1).contiguous().view(-1, bsz * self.num_heads, self.k_dim).transpose(0, 1)
        k = k.transpose(0, 1).contiguous().view(-1, bsz * self.num_heads, self.k_dim).transpose(0, 1)
        v = v.transpose(0, 1).contiguous().view(-1, bsz * self.num_heads, self.v_dim).transpose(0, 1)

        attn_weights = torch.bmm(q, k.transpose(1, 2))
        assert list(attn_weights.size()) == [bsz * self.num_heads, src_len, trg_len]

        if mask is not None:
            attn_weights = attn_weights.view(bsz, self.num_heads, src_len, trg_len)
            mask = mask.unsqueeze(1)
            attn_weights = attn_weights - (1 - mask) * 1e6
            attn_weights = attn_weights.view(bsz * self.num_heads, src_len, trg_len)

        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

        attn = torch.bmm(attn_weights, v)

        attn = attn.view(bsz, self.num_heads, src_len, self.v_dim).transpose(1, 2).contiguous().view(bsz, src_len, self.num_heads * self.v_dim)

        if self.q_transform is not None:
            attn = attn + self.q_transform(query)

        return attn

class InteractionLayer(nn.Module):
    
    def __init__(self, embed_dim, num_heads, qk_dim=None, bias=True):
        super(InteractionLayer, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.qk_dim = embed_dim // num_heads if qk_dim is None else qk_dim
        self.bias = bias
        self.scaling = self.qk_dim ** -0.5
        self.q_proj = nn.Linear(embed_dim, self.num_heads * self.qk_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, self.num_heads * self.qk_dim, bias=bias)

        self.init_weight()

    def init_weight(self):
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.q_proj.weight)

        if self.bias:
            nn.init.constant_(self.q_proj.bias, 0.)
            nn.init.constant_(self.k_proj.bias, 0.)

    def forward(self, query, key):
        # bsz, len, embed_dim
        bsz = query.size(0)
        q_len = query.size(1)
        k_len = key.size(1)

        q = self.q_proj(query)
        k = self.k_proj(key)

        q *= self.scaling
        q = q.transpose(0, 1).contiguous().view(-1, bsz * self.num_heads, self.qk_dim).transpose(0, 1)
        k = k.transpose(0, 1).contiguous().view(-1, bsz * self.num_heads, self.qk_dim).transpose(0, 1)

        qk_weights = torch.bmm(q, k.transpose(1, 2))

        qk_weights = qk_weights.view(bsz, self.num_heads, q_len, k_len).permute(0,2,3,1)

        return qk_weights

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
            nn.GELU(),
            nn.Dropout(dropout_prob),
            nn.Linear(256, 3)
        )

        self.events_tags_classifier = InteractionLayer(self.hidden_size, 9, qk_dim=128)

        self.events_transform = BasicAttentionLayer(self.hidden_size, self.num_attention_heads, q_transform=True)
        self.events_relations_classifier = nn.Sequential(
            nn.Linear(self.hidden_size*2, 256),
            nn.GELU(),
            nn.Dropout(dropout_prob),
            nn.Linear(256, 7)
        )

    def calc_triggers_tags(self, seq_out):

        triggers_logits = self.triggers_classifier(seq_out)

        return triggers_logits
    
    def calc_events_tags(self, seq_out, triggers_hidden):

        events_tags_logits = self.events_tags_classifier(triggers_hidden, seq_out)

        return events_tags_logits
    
    def calc_events_relations(self, events_hidden):
        b, n, h = events_hidden.shape
        events_a = events_hidden.unsqueeze(2).expand(b, n, n, h)
        events_b = events_hidden.unsqueeze(1).expand(b, n, n, h)
        events_interaction = torch.cat([events_a, events_b], dim=-1)

        events_relations_logits = self.events_relations_classifier(events_interaction)
        return events_relations_logits

    def get_events_hidden(self, seq_out, triggers_hidden, attention_mask):
        triggers_hidden = triggers_hidden
        events_hidden = self.events_transform(triggers_hidden, seq_out, seq_out, attention_mask.unsqueeze(1))
        return events_hidden

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

        events_hidden = self.get_events_hidden(seq_out, triggers_hidden, attention_mask)
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


        

        





    