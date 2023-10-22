# coding=utf-8
# coding=utf-8
import json
from dataloader import ID2TAG, LABEL2ER
from model import EventExtractModel
import torch
from transformers import AutoTokenizer
from predict import predict_with_triggers
from utils import get_events_relations_list, get_tags_pos_list, get_trigger_pos_from_text


if __name__ == "__main__":

    tokenizer = AutoTokenizer.from_pretrained("../pretrain/chinese-roberta-wwm-ext")
    model = EventExtractModel("../pretrain/chinese-roberta-wwm-ext")
    model.load_state_dict(torch.load("./output/model_15.pt"))
    mdoel = model.cuda()
    model.eval()

    preds = []
    targets = []

    with open("./data/raw_dev.json", "r", encoding='utf8') as f:
        for idx, line in enumerate(f.readlines()):
            data = json.loads(line.strip())
            full_text = data['full_text']
            event_list = data['event_list']
            relation_list = data['relation_list']

            trigger_pos_list = [event['event_trigger_offset'] for event in event_list]
            trigger_id_list = [event['eid'] for event in event_list]

            input_text, triggers_pos_list_out, events_tags_list, events_relations_list = predict_with_triggers(model, tokenizer, full_text, trigger_pos_list)
            pred_relation_list = []
            for i, event_relations in enumerate(events_relations_list):
                for j, relation in enumerate(event_relations):
                    if relation > 0:
                        pred_relation_list.append({
                            "relation": LABEL2ER[relation], 
                            "head_id": trigger_id_list[i],
                            "tail_id": trigger_id_list[j]
                        })
            preds.append(pred_relation_list)
            targets.append(relation_list)
            # print(idx)
            # print(pred_relation_list)
            # print(relation_list)
            # print()
    acc_num = 0
    pred_num = 0
    label_num = 0
    for pred, target in zip(preds, targets):
        pred = {"{}-{}".format(p['head_id'], p['tail_id']): p['relation'] for p in pred}
        target = {"{}-{}".format(t['head_id'], t['tail_id']): t['relation'] for t in target}
        pred_num += len(pred)
        label_num += len(target)
        for k in pred.keys():
            if k in target.keys() and target[k]==pred[k]:
                acc_num += 1
    precision, recall = round(acc_num/(label_num+1e-12), 4), round(acc_num/(pred_num+1e-12), 4)
    f1 = round((2 * precision * recall) / (precision + recall + 1e-12), 4)
    print("f1:", f1)
    print("precision:", precision)
    print("recall:", recall)
