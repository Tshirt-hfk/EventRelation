# coding=utf-8
from dataset import ID2TAG, LABEL2ER
from model import EventExtractModel
import torch
from transformers import AutoTokenizer
from utils import get_events_relations_list, get_tags_pos_list


def predict(model, tokenizer, input_text):
    input_text = tokenizer.tokenize(input_text)
    tokenized_input = tokenizer(input_text, is_split_into_words=True)
    input_ids = tokenized_input.input_ids
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

    events_mask = [[1]*max_trigger_num]
    events_mask = torch.LongTensor(events_mask).cuda()
    events_tags_mask =  torch.einsum('bn,bl->bnl', events_mask, attention_mask)
    events_tags_logit = model.calc_events_tags(seq_out, triggers_hidden)
    events_tags_label = torch.max(events_tags_logit, dim=-1)[1]
    events_tags_label = torch.reshape(events_tags_label, [-1, events_tags_label.size(-1)])
    events_tags_mask = torch.reshape(events_tags_mask, [-1, events_tags_mask.size(-1)])
    events_tags_list_out = get_tags_pos_list(events_tags_label, events_tags_mask)

    events_hidden = model.get_events_hidden(seq_out, triggers_hidden, attention_mask)
    events_relations_logit = model.calc_events_relations(events_hidden)

    events_relations = torch.max(events_relations_logit, dim=-1)[1]
    events_relations_list = get_events_relations_list(events_relations, events_mask)

    return input_text, triggers_pos_list_out, events_tags_list_out, events_relations_list


if __name__ == "__main__":

    tokenizer = AutoTokenizer.from_pretrained("../pretrain/chinese-roberta-wwm-ext")
    model = EventExtractModel("../pretrain/chinese-roberta-wwm-ext")
    model.load_state_dict(torch.load("./output/model_9.pt"))
    mdoel = model.cuda()
    model.eval()
    text = "昨天清晨6时许，一辆乘坐12人的超载面包车行驶至京承高速进京方向时突然起火，司机和副驾驶逃生，而坐在车内的10名木工不同程度烧伤，其中一人死亡。据了解，面包车可能是自燃，司机已被警方带走调查。"
    text = "事发后，急救车赶到将两名伤者送至附近医院。据了解，面包车司机伤势无大碍，被甩出的男乘客伤势较重，正在医院治疗。目前，事故原因正在进一步调查中。"
    input_text, triggers_pos_list, events_tags_list, events_relations_list = predict(model, tokenizer, text)

    print("".join(input_text))

    for i, (trigger_pos, event_tags) in enumerate(zip(triggers_pos_list, events_tags_list)):
        print("\n----------event {} start----------".format(i))
        _, pos = trigger_pos
        print("TRIGGER:", input_text[pos[0]-1:pos[-1]])
        for event_tag in event_tags:
            label, pos = event_tag
            tag = ID2TAG[label]
            print(tag+":", input_text[pos[0]-1:pos[-1]])
        print("----------event {} end----------".format(i))
    print("\n")
    for i, event_relations in enumerate(events_relations_list):
        for j, relation in enumerate(event_relations):
            if relation > 0:
                print("event {} - {} relation: {}".format(i, j, LABEL2ER[relation]))