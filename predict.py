# coding=utf-8
from dataloader import ID2TAG, LABEL2ER
from model import EventExtractModel
import torch
from transformers import AutoTokenizer
from utils import get_events_relations_list, get_tags_pos_list, get_trigger_pos_from_text


def predict(model, tokenizer, input_text):

    input_text = tokenizer.tokenize(input_text)
    tokenized_input = tokenizer(input_text, is_split_into_words=True)
    input_ids = tokenized_input.input_ids
    if len(input_ids) > 256:
        raise Exception("input text is too long!")
    attention_mask = tokenized_input.attention_mask
    input_ids = torch.LongTensor([input_ids]).cuda()
    attention_mask = torch.LongTensor([attention_mask]).cuda()
    seq_out, _ = model.get_seq_hidden(input_ids, attention_mask)
    triggers_logit = model.calc_triggers_tags(seq_out)

    triggers_tags_label = torch.max(triggers_logit, dim=-1)[1]
    triggers_pos_list_out = get_tags_pos_list(triggers_tags_label, attention_mask)[0]
    triggers_pos_list = [list(trigger_pos[1]) for trigger_pos in triggers_pos_list_out]

    max_trigger_num = len(triggers_pos_list)
    max_trigger_len = max([len(trigger_pos) for trigger_pos in triggers_pos_list] + [0])
    triggers_pos = [[trigger + [0] * (max_trigger_len - len(trigger)) for trigger in triggers_pos_list]]
    triggers_mask = [[[1] * len(trigger) + [0] * (max_trigger_len - len(trigger)) for trigger in triggers_pos_list]]
    
    triggers_pos = torch.LongTensor(triggers_pos).cuda()
    triggers_mask = torch.LongTensor(triggers_mask).cuda()
    triggers_hidden = model.get_triggers_hidden(seq_out, triggers_pos, triggers_mask)

    events_mask = torch.LongTensor([[1]*max_trigger_num]).cuda()
    events_tags_mask =  torch.einsum('bn,bl->bnl', events_mask, attention_mask)
    event_pos = torch.arange(seq_out.size(1)).type_as(triggers_pos[:,:,:1]).view(1, 1, -1) + model.max_positon - triggers_pos[:,:,:1]
    events_hidden = model.get_events_hidden(seq_out, triggers_hidden, event_pos, attention_mask)

    events_tags_logit = model.calc_events_tags(seq_out, events_hidden)
    events_tags_label = torch.max(events_tags_logit, dim=-1)[1]
    events_tags_label = torch.reshape(events_tags_label, [-1, events_tags_label.size(-1)])
    events_tags_mask = torch.reshape(events_tags_mask, [-1, events_tags_mask.size(-1)])
    events_tags_list_out = get_tags_pos_list(events_tags_label, events_tags_mask)

    events_relations_logit = model.calc_events_relations(events_hidden)
    events_relations = torch.max(events_relations_logit, dim=-1)[1]
    events_relations_list = get_events_relations_list(events_relations, events_mask)

    return input_text, triggers_pos_list_out, events_tags_list_out, events_relations_list


def predict_with_triggers(model, tokenizer, input_text, pos_list):
    
    input_text, pos_list = get_trigger_pos_from_text(tokenizer, input_text, pos_list)
    triggers_pos_list_out = [(0, (pos[0]+1, pos[1])) for pos in pos_list]
    tokenized_input = tokenizer(input_text, is_split_into_words=True)
    input_ids = tokenized_input.input_ids
    if len(input_ids) > 256:
        raise Exception("input text is too long!")
    attention_mask = tokenized_input.attention_mask
    input_ids = torch.LongTensor([input_ids]).cuda()
    attention_mask = torch.LongTensor([attention_mask]).cuda()
    seq_out, _ = model.get_seq_hidden(input_ids, attention_mask)

    triggers_pos_list = [[x+1 for x in range(pos[0], pos[1])] for pos in pos_list]
    max_trigger_num = len(triggers_pos_list)
    max_trigger_len = max([len(trigger_pos) for trigger_pos in triggers_pos_list] + [0])
    triggers_pos = [[trigger + [0] * (max_trigger_len - len(trigger)) for trigger in triggers_pos_list]]
    triggers_mask = [[[1] * len(trigger) + [0] * (max_trigger_len - len(trigger)) for trigger in triggers_pos_list]]
    
    triggers_pos = torch.LongTensor(triggers_pos).cuda()
    triggers_mask = torch.LongTensor(triggers_mask).cuda()
    triggers_hidden = model.get_triggers_hidden(seq_out, triggers_pos, triggers_mask)

    events_mask = torch.LongTensor([[1]*max_trigger_num]).cuda()
    events_tags_mask =  torch.einsum('bn,bl->bnl', events_mask, attention_mask)
    event_pos = torch.arange(seq_out.size(1)).type_as(triggers_pos[:,:,:1]).view(1, 1, -1) + model.max_positon - triggers_pos[:,:,:1]
    events_hidden = model.get_events_hidden(seq_out, triggers_hidden, event_pos, attention_mask)

    events_tags_logit = model.calc_events_tags(seq_out, events_hidden)
    events_tags_label = torch.max(events_tags_logit, dim=-1)[1]
    events_tags_label = torch.reshape(events_tags_label, [-1, events_tags_label.size(-1)])
    events_tags_mask = torch.reshape(events_tags_mask, [-1, events_tags_mask.size(-1)])
    events_tags_list_out = get_tags_pos_list(events_tags_label, events_tags_mask)

    events_relations_logit = model.calc_events_relations(events_hidden)
    events_relations = torch.max(events_relations_logit, dim=-1)[1]
    events_relations_list = get_events_relations_list(events_relations, events_mask)

    return input_text, triggers_pos_list_out, events_tags_list_out, events_relations_list

if __name__ == "__main__":

    tokenizer = AutoTokenizer.from_pretrained("../pretrain/chinese-roberta-wwm-ext")
    model = EventExtractModel("../pretrain/chinese-roberta-wwm-ext")
    model.load_state_dict(torch.load("./output/model_15.pt"))
    mdoel = model.cuda()
    model.eval()

    text = "据中国国家地震台网测定，北京时间2008年5月12日14时28分，在四川汶川县（北纬31．0度，东经103．4度）发生7．6级地震。"
    
    input_text, triggers_pos_list, events_tags_list, events_relations_list = predict(model, tokenizer, text)

    print("".join(input_text))

    print("\npredict for event analysis =====================================>")
    for i, (trigger_pos, event_tags) in enumerate(zip(triggers_pos_list, events_tags_list)):
        print("----------event {} start----------".format(i))
        _, pos = trigger_pos
        print("TRIGGER:", "".join(input_text[pos[0]-1:pos[-1]]))
        for event_tag in event_tags:
            label, pos = event_tag
            tag = ID2TAG[label]
            print(tag+":", "".join(input_text[pos[0]-1:pos[-1]]))
        print("----------event {} end----------".format(i))
    
    text = "本报讯(记者雷娜)昨天上午11时许，顺平路北窑上桥南两公里处，一辆满载煤炭的大货车为躲避前方突然并线的小轿车，失控侧翻到逆行车道。事故中司机受伤。"
    trigger_pos_list = [[2, 3], [42, 44]]
    for i, event_relations in enumerate(events_relations_list):
        for j, relation in enumerate(event_relations):
            if relation > 0:
                print("event {} - {} relation: {}".format(i, j, LABEL2ER[relation]))

    print("\nonly predict for event relation =====================================>")
    input_text, triggers_pos_list_out, events_tags_list, events_relations_list = predict_with_triggers(model, tokenizer, text, trigger_pos_list)
    for i, (trigger_pos, event_tags) in enumerate(zip(triggers_pos_list, events_tags_list)):
        print("----------event {} start----------".format(i))
        _, pos = trigger_pos
        print("TRIGGER:", "".join(input_text[pos[0]-1:pos[-1]]))
        for event_tag in event_tags:
            label, pos = event_tag
            tag = ID2TAG[label]
            print(tag+":", "".join(input_text[pos[0]-1:pos[-1]]))
        print("----------event {} end----------".format(i))
    print(events_relations_list)
    for i, event_relations in enumerate(events_relations_list):
        for j, relation in enumerate(event_relations):
            if relation > 0:
                print("event {} - {} relation: {}".format(i, j, LABEL2ER[relation]))