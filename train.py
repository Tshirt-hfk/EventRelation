# coding=utf-8
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from transformers import AutoTokenizer
from dataset import BatchEventDataset
from model import EventExtractModel
from utils import calc_events_relations_mertic, calc_tags_metric, get_events_relations_list, get_tags_pos_list


epoch = 20
tokenizer = AutoTokenizer.from_pretrained("../pretrain/chinese-roberta-wwm-ext")
train_dataset = BatchEventDataset(file_path="./data/train.json", tokenizer=tokenizer, batch_size=32, shuffle=True, last_batch=False)
dev_dataset = BatchEventDataset(file_path="./data/dev.json", tokenizer=tokenizer, batch_size=8, shuffle=False, last_batch=True)
model = EventExtractModel("../pretrain/chinese-roberta-wwm-ext").cuda()
optimizer = optim.AdamW([
    {'params': [p for n, p in model.named_parameters() if "LayerNorm" not in n], 'weight_decay': 0.01},
    {'params': [p for n, p in model.named_parameters() if "LayerNorm" in n], 'weight_decay': 0.0}
], lr=4e-5)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9, last_epoch=-1, verbose=False)

for idx in range(epoch):
    print("epoch {} start to train ====>".format(idx))
    print("epoch: {}, lr: {}".format(idx, scheduler.get_last_lr()))
    model.train()
    for i, data in enumerate(train_dataset):
        input_ids = data['input_ids'].cuda()
        attention_mask = data['attention_mask'].cuda()
        triggers_pos = data['triggers_pos'].cuda()
        triggers_label = data['triggers_label'].cuda()
        triggers_mask = data['triggers_mask'].cuda()
        events_mask = data['events_mask'].cuda()
        events_polarity = data['events_polarity'].cuda()
        events_tags_label = data['events_tags_label'].cuda()
        events_tags_mask = data['events_tags_mask'].cuda()
        events_relations_label = data['events_relations_label'].cuda()
        
        triggers_logit, events_tags_logit, events_relations_logit = model(input_ids, attention_mask, triggers_pos, triggers_mask)

        triggers_loss, events_tags_loss, events_relations_loss = model.calc_loss(triggers_logit, triggers_label,
                        events_tags_logit, events_tags_label,
                        events_relations_logit, events_relations_label,
                        attention_mask, events_tags_mask, events_mask)
        loss = triggers_loss + events_tags_loss * min(1, idx*0.5) + events_relations_loss * min(1, idx*0.5)

        print("epoch: {}-{}, triggers_loss: {}, events_tags_loss: {}, events_relations_loss: {}".format(idx, i, triggers_loss, events_tags_loss, events_relations_loss))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    print("epoch {} start to dev ====>".format(idx))
    model.eval()
    pre_triggers_pos_list = []
    target_triggers_pos_list = []
    pred_events_tags_list = []
    target_events_tags_list = []
    pred_events_relations_list = []
    target_events_relations_list = []
    for i, data in enumerate(dev_dataset):
        input_ids = data['input_ids'].cuda()
        attention_mask = data['attention_mask'].cuda()
        triggers_pos = data['triggers_pos'].cuda()
        triggers_label = data['triggers_label'].cuda()
        triggers_mask = data['triggers_mask'].cuda()
        events_mask = data['events_mask'].cuda()
        events_polarity = data['events_polarity'].cuda()
        events_tags_label = data['events_tags_label'].cuda()
        events_tags_mask = data['events_tags_mask'].cuda()
        events_relations_label = data['events_relations_label'].cuda()

        triggers_logit, events_tags_logit, events_relations_logit = model(input_ids, attention_mask, triggers_pos, triggers_mask)

        pred_trigger_labels = torch.max(triggers_logit, dim=-1)[1]
        pre_triggers_pos_list += get_tags_pos_list(pred_trigger_labels, attention_mask)
        target_triggers_pos_list += get_tags_pos_list(triggers_label, attention_mask)

        pred_events_tag = torch.max(events_tags_logit, dim=-1)[1]
        pred_events_tag = torch.reshape(pred_events_tag, [-1, pred_events_tag.size(-1)])
        events_tags_label = torch.reshape(events_tags_label, [-1, events_tags_label.size(-1)])
        events_tags_mask = torch.reshape(events_tags_mask, [-1, events_tags_mask.size(-1)])
        events_masks = torch.reshape(triggers_mask, [-1])
        pred_events_tags_list += get_tags_pos_list(pred_events_tag, events_tags_mask, events_mask)
        target_events_tags_list += get_tags_pos_list(events_tags_label, events_tags_mask, events_mask)

        pred_events_relations = torch.max(events_relations_logit, dim=-1)[1]
        pred_events_relations_list += get_events_relations_list(pred_events_relations, events_mask)
        target_events_relations_list += get_events_relations_list(events_relations_label, events_mask)
    
    torch.save(model.state_dict(),'./output/model_{}.pt'.format(idx))
    print("trigger tags: ",calc_tags_metric(pre_triggers_pos_list, target_triggers_pos_list, frist_print=False))
    print("event tags: ", calc_tags_metric(pred_events_tags_list, target_events_tags_list, frist_print=False))
    print("event relations: ", calc_events_relations_mertic(pred_events_relations_list, target_events_relations_list, frist_print=False))
    
    scheduler.step()
        
        
        