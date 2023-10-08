# coding=utf-8


def get_tags_pos_list(pred_labels, input_masks, events_masks=None):
    pred_labels = pred_labels.tolist()
    input_masks = input_masks.tolist()
    if events_masks is None:
        events_masks = [1] * len(input_masks)
    else:
        events_masks = events_masks.tolist()
    pred_trigger_pos_list = []
    for one_pred_labels, one_input_masks, one_mask in zip(pred_labels, input_masks, events_masks):
        if one_mask==0:
            continue
        pred_trigger_pos = []
        trigger_pos = []
        tag = -1
        for i, (label, mask) in enumerate(zip(one_pred_labels, one_input_masks)):
            if mask == 0:
                break
            if label%2 == 1:
                tag = (label-1)//2
                trigger_pos.append(i)
            elif trigger_pos and label%2==0 and (label-1)//2 == tag:
                trigger_pos.append(i)
            elif trigger_pos:
                pred_trigger_pos.append((tag, tuple(trigger_pos)))
                trigger_pos = []
                tag = -1
        if trigger_pos:
            pred_trigger_pos.append((tag, tuple(trigger_pos)))
        pred_trigger_pos_list.append(pred_trigger_pos)
    return pred_trigger_pos_list

def calc_tags_metric(pred_trigger_pos_list, target_trigger_pos_list, frist_print=False):
    pred_num = 0
    label_num = 0
    acc_num = 0
    for pred_trigger_pos, target_trigger_pos in zip(pred_trigger_pos_list, target_trigger_pos_list):
        if frist_print:
            print(pred_trigger_pos, target_trigger_pos)
            frist_print = False
        pred_trigger_pos = set(pred_trigger_pos)
        target_trigger_pos = set(target_trigger_pos)
        pred_num += len(pred_trigger_pos)
        label_num += len(target_trigger_pos)
        acc_num += len(pred_trigger_pos & target_trigger_pos)
    precision, recall = round(acc_num/(label_num+1e-12), 4), round(acc_num/(pred_num+1e-12), 4)
    f1 = (2 * precision * recall) / (precision + recall + 1e-12)
    return (f1, precision, recall), (acc_num, label_num, pred_num)

def get_events_relations_list(events_relations, events_mask):
    events_relations = events_relations.tolist()
    events_mask = events_mask.tolist()
    events_relation_list = []
    for even_relations, event_mask in zip(events_relations, events_mask):
        for even_relation, mask_1 in zip(even_relations, event_mask):
            if mask_1==0:
                continue
            er_list = []
            for er, mask_2 in zip(even_relation, event_mask):
                if mask_2==0:
                    break
                er_list.append(er)
            events_relation_list.append(er_list)
    return events_relation_list

def calc_events_relations_mertic(preds, targets, frist_print=False):
    pred_num = 0
    label_num = 0
    acc_num = 0
    total_acc_num = 0
    total_num = 0
    for pred, target in zip(preds, targets):
        if frist_print:
            print(pred, target)
            frist_print=False
        for p, t in zip(pred, target):
            if t>0 and p==t:
                acc_num+=1
            if p>0:
                pred_num+=1
            if t>0:
                label_num+=1
            if p==t:
                total_acc_num+=1
            total_num+=1
    precision, recall = round(acc_num/(label_num+1e-12), 4), round(acc_num/(pred_num+1e-12), 4)
    f1 = (2 * precision * recall) / (precision + recall + 1e-12)
    acc = round(total_acc_num/(total_num+1e-12), 4)
    return (f1, precision, recall), (acc_num, label_num, pred_num), acc

