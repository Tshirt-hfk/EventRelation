# coding:utf-8
import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel

class SelfAttentionLayer(nn.Module):
    
    def __init__(self, embed_dim, num_heads, k_dim=None, v_dim=None, bias=True, dropout=0.0, max_positon=256):
        super(SelfAttentionLayer, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.k_dim = embed_dim // num_heads if k_dim is None else k_dim
        self.v_dim = embed_dim // num_heads if v_dim is None else v_dim
        self.bias = bias
        self.scaling = self.k_dim ** -0.5
        self.dropout = dropout

        self.q_proj = nn.Linear(embed_dim, self.num_heads * self.k_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, self.num_heads * self.k_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, self.num_heads * self.v_dim, bias=bias)

        self.rpe = nn.Parameter(torch.zeros(max_positon*2, self.num_heads, self.v_dim))

        self.q_transform = nn.Linear(embed_dim, self.embed_dim, bias=bias)
        self.out_proj = nn.Linear(self.num_heads * self.v_dim, self.embed_dim, bias=bias)
        self.layer_norm = nn.LayerNorm(self.embed_dim, eps=1e-6)

        self.init_weight()

    def init_weight(self):
        nn.init.xavier_normal_(self.q_transform.weight)
        nn.init.xavier_normal_(self.out_proj.weight)
        nn.init.xavier_normal_(self.q_proj.weight)
        nn.init.xavier_normal_(self.k_proj.weight)
        nn.init.xavier_normal_(self.v_proj.weight)

        if self.bias:
            nn.init.constant_(self.q_transform.bias, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)
            nn.init.constant_(self.q_proj.bias, 0.)
            nn.init.constant_(self.k_proj.bias, 0.)
            nn.init.constant_(self.v_proj.bias, 0.)

    def forward(self, query, key, value, pos, mask=None):
        # bsz, len, embed_dim
        bsz = query.size(0)
        src_len = query.size(1)
        trg_len = value.size(1)

        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        q *= self.scaling
        q = q.view(bsz, -1, self.num_heads, self.k_dim)
        k = k.view(bsz, -1, self.num_heads, self.k_dim)
        v = v.view(bsz, -1, self.num_heads, self.v_dim)

        attn_weights = torch.einsum("bqnd,bknd->bqkn", q, k)
        qrpe_weights = torch.einsum("bqnd,lnd->bqln", q, self.rpe)
        pos = pos.unsqueeze(3).expand(-1, -1, -1, self.num_heads)
        qrpe_weights = torch.gather(qrpe_weights, dim=2, index=pos)
        attn_weights = attn_weights + qrpe_weights

        if mask is not None:
            attn_weights = attn_weights - (1 - mask.unsqueeze(-1)) * 1e6

        attn_weights = torch.softmax(attn_weights, dim=-2)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

        attn = torch.einsum("bqkn,bknd->bqnd", attn_weights, v)
        attn = attn.contiguous().view(bsz, src_len, self.num_heads * self.v_dim)

        out = self.layer_norm(self.out_proj(attn) + self.q_transform(query))

        return out

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
        nn.init.xavier_normal_(self.k_proj.weight)
        nn.init.xavier_normal_(self.q_proj.weight)

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
        q = q.view(bsz, q_len, self.num_heads, self.qk_dim)
        k = k.view(bsz, k_len, self.num_heads, self.qk_dim)

        qk_weights = torch.einsum("bqnd,bknd->bqkn", q, k)  
        return qk_weights
    

class EventExtractModel(nn.Module):

    def __init__(self,
                 bert_dir,
                 num_trigger_tags=3,
                 num_event_tags=9,
                 num_event_relations=7,
                 max_positon=512,
                 dropout_prob=0.1):
        super(EventExtractModel, self).__init__()

        assert os.path.exists(bert_dir), 'pretrained bert file does not exist'

        self.bert_module = BertModel.from_pretrained(bert_dir)
        self.hidden_size = self.bert_module.config.hidden_size
        self.num_attention_heads = self.bert_module.config.num_attention_heads
        self.max_positon = max_positon

        self.triggers_tags_classifier = nn.Sequential(
            nn.Linear(self.hidden_size, 256),
            nn.GELU(),
            nn.Dropout(dropout_prob),
            nn.Linear(256, num_trigger_tags)
        )

        self.events_transform = SelfAttentionLayer(self.hidden_size, self.num_attention_heads, max_positon=max_positon)

        self.events_tags_classifier = InteractionLayer(self.hidden_size, num_event_tags, qk_dim=128)
        
        self.events_relations_classifier = nn.Sequential(
            nn.Linear(self.hidden_size*2, 512),
            nn.GELU(),
            nn.Dropout(dropout_prob),
            nn.Linear(512, num_event_relations)
        )

    def calc_triggers_tags(self, seq_out):

        triggers_logits = self.triggers_tags_classifier(seq_out)

        return triggers_logits
    
    def calc_events_tags(self, seq_out, events_hidden):

        events_tags_logits = self.events_tags_classifier(events_hidden, seq_out)

        return events_tags_logits
    
    def calc_events_relations(self, events_hidden):
        b, n, h = events_hidden.shape
        events_a = events_hidden.unsqueeze(2).expand(b, n, n, h)
        events_b = events_hidden.unsqueeze(1).expand(b, n, n, h)
        events_interaction = torch.cat([events_a, events_b], dim=-1)

        events_relations_logits = self.events_relations_classifier(events_interaction)
        return events_relations_logits

    def get_events_hidden(self, seq_out, triggers_hidden, event_pos, attention_mask):
        events_hidden = self.events_transform(triggers_hidden, seq_out, seq_out, event_pos, attention_mask.unsqueeze(1))
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

        seq_len = seq_out.size(1)
        triggers_pos = triggers_pos[:,:,:1]
        event_pos = torch.arange(seq_len).type_as(triggers_pos).view(1, 1, -1) + self.max_positon - triggers_pos

        events_hidden = self.get_events_hidden(seq_out, triggers_hidden, event_pos, attention_mask)

        events_tags_logit = self.calc_events_tags(seq_out, events_hidden)

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


        

        





    