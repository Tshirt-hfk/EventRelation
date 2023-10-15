# coding:utf-8
import math
import json
import random
import torch

TRIGGER2LABEL = {
    "O": 0,
    "B-TRIGGER": 1,
    "I-TRIGGER": 2,
}
LABEL2TRIGGER = {v:k for k,v in TRIGGER2LABEL.items()}

TAG2LABEL = {
    "O": 0,
    "B-TIME": 1,
    "I-TIME": 2,
    "B-LOC": 3,
    "I-LOC": 4,
    "B-OBJECT-S": 5,
    "I-OBJECT-S": 6,
    "B-OBJECT-O": 7,
    "I-OBJECT-O": 8,
}
LABEL2TAG = {v:k for k,v in TAG2LABEL.items()}
ID2TAG = ["TIME", "LOC", "OBJECT-S", "OBJECT-O"]

ER2LABEL = {
    'O': 0,
    'Accompany': 1, 
    'Composite': 2, 
    'Concurrency': 3, 
    'Follow': 4, 
    'Thoughtcontent': 5, 
    'Causal': 6
}
LABEL2ER = {v:k for k,v in ER2LABEL.items()}

POLARITY2LABEL = {
    "Positive": 1,
    "Negative": 0
}
LABEL2POLARITY = {v:k for k,v in POLARITY2LABEL.items()}

class EventDataset:

    def __init__(self, file_path, tokenizer):
        super().__init__()

        self.examples = []
        with open(file_path, 'r', encoding="utf-8") as f:
            for line in f.readlines():

                example = json.loads(line.strip())
                tokenized_input = tokenizer(example["sentence"], is_split_into_words=True)
                input_ids = tokenized_input.input_ids
                word_ids = tokenized_input.word_ids()
                triggers_label = [TRIGGER2LABEL['O']] * len(word_ids)
                triggers_pos = []
                events_polarity = []
                events_label = []
                for event in example['events']:
                    event_label = []
                    trigger_pos = []
                    for i, word_id in enumerate(word_ids):
                        label = TAG2LABEL['O']
                        if word_id is None:
                            event_label.append(label)
                            continue
                        if event['trigger']['offset'][0] == word_id:
                            triggers_label[i] = TRIGGER2LABEL['B-TRIGGER']
                            trigger_pos.append(i)
                        if event['trigger']['offset'][0] < word_id < event['trigger']['offset'][1]:
                            triggers_label[i] = TRIGGER2LABEL['I-TRIGGER']
                            trigger_pos.append(i)
                        if event['time'] is not None:
                            if event['time']['offset'][0] == word_id:
                                label = TAG2LABEL['B-TIME']
                            elif event['time']['offset'][0] < word_id < event['time']['offset'][1]:
                                label = TAG2LABEL['I-TIME']
                        if event['loc'] is not None:
                            if event['loc']['offset'][0] == word_id:
                                label = TAG2LABEL['B-LOC']
                            elif event['loc']['offset'][0] < word_id < event['loc']['offset'][1]:
                                label = TAG2LABEL['I-LOC']       
                        for object_s in event['object_s']:
                            if object_s['offset'][0] == word_id:
                                label = TAG2LABEL['B-OBJECT-S']
                            elif object_s['offset'][0] < word_id < object_s['offset'][1]:
                                label = TAG2LABEL['I-OBJECT-S']
                        for object_o in event['object_o']:
                            if object_o['offset'][0] == word_id:
                                label = TAG2LABEL['B-OBJECT-O']
                            elif object_o['offset'][0] < word_id < object_o['offset'][1]:
                                label = TAG2LABEL['I-OBJECT-O']
                        event_label.append(label)
                    triggers_pos.append(trigger_pos)
                    events_polarity.append(POLARITY2LABEL[event['polarity']])
                    events_label.append(event_label)    
                events_relation = [[ER2LABEL['O'] for _ in range(len(example['events']))] for _ in range(len(example['events']))]
                for relation in example['relations']:
                    events_relation[relation['eid_s']][relation['eid_o']] = ER2LABEL[relation['type']]
                self.examples.append({
                    "input_ids": input_ids,
                    "triggers_label": triggers_label,
                    "triggers_pos": triggers_pos,
                    "events_polarity": events_polarity,
                    "events_label": events_label,
                    "events_relation": events_relation
                })

    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]
    

class BatchEventDataset(EventDataset):

    def __init__(self, file_path, tokenizer, batch_size=32, shuffle=True, drop_last=False):
        super().__init__(file_path, tokenizer)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.batch_examples = []

    def __len__(self):
        return len(self.batch_examples)
    
    def __getitem__(self, idx):
        return self.batch_examples[idx]
    
    def batch_padding(self):
        self.batch_examples.clear()
        if self.shuffle:
            random.shuffle(self.examples)
        if not self.drop_last:
            num = math.ceil(len(self.examples)/self.batch_size)
        else:
            num = len(self.examples)//self.batch_size
        for i in range(num):
            input_ids_list = [example['input_ids'] for example in self.examples[i*self.batch_size:(i+1)*self.batch_size]]
            triggers_label_list = [example['triggers_label'] for example in self.examples[i*self.batch_size:(i+1)*self.batch_size]]
            triggers_pos_list = [example['triggers_pos'] for example in self.examples[i*self.batch_size:(i+1)*self.batch_size]]
            events_polarity_list = [example['events_polarity'] for example in self.examples[i*self.batch_size:(i+1)*self.batch_size]]
            events_label_list = [example['events_label'] for example in self.examples[i*self.batch_size:(i+1)*self.batch_size]]
            events_relation_list = [example['events_relation'] for example in self.examples[i*self.batch_size:(i+1)*self.batch_size]]
            max_len = max([len(ids) for ids in input_ids_list])
            batch_input_ids = [ids + [0] * (max_len-len(ids)) for ids in input_ids_list]
            batch_attention_mask = [[1] * len(ids)  + [0] * (max_len-len(ids)) for ids in input_ids_list]
            batch_triggers_label = [triggers_label + [0] * (max_len-len(triggers_label)) 
                                        for triggers_label in triggers_label_list]
            
            max_trigger_num = max([len(pos) for pos in triggers_pos_list])
            max_trigger_len = max([max([len(trigger) for trigger in trigger_pos] + [0]) for trigger_pos in triggers_pos_list])
            batch_triggers_pos = [[trigger + [0] * (max_trigger_len - len(trigger)) for trigger in trigger_pos] + 
                                    [[0]*max_trigger_len] * (max_trigger_num-len(trigger_pos)) 
                                    for trigger_pos in triggers_pos_list]
            batch_triggers_mask = [[[1] * len(trigger) + [0] * (max_trigger_len - len(trigger)) for trigger in trigger_pos] + 
                                        [[0]*max_trigger_len] * (max_trigger_num-len(trigger_pos)) 
                                        for trigger_pos in triggers_pos_list]
            
            max_event_num = max([len(events_label) for events_label in events_label_list])        
            max_event_tag_len = max([max([len(labels) for labels in events_label] + [0]) for events_label in events_label_list])

            batch_events_mask = [[1]*len(events_relation) + [0]*(max_event_num-len(events_relation)) 
                                    for events_relation in events_relation_list]
            batch_events_polarity = [events_polarity + [0]*(max_event_num-len(events_polarity)) 
                                        for events_polarity in events_polarity_list]
            batch_events_tags_label = [[event_label + [0] * (max_event_tag_len - len(event_label)) for event_label in events_label] + 
                                        [[0] * max_event_tag_len] * (max_event_num - len(events_label))
                                        for events_label in events_label_list]
            batch_events_tags_mask = [[[1]*len(event_label) + [0]*(max_event_tag_len - len(event_label)) for event_label in events_label] + 
                                        [[0] * max_event_tag_len] * (max_event_num - len(events_label))
                                        for events_label in events_label_list]
            batch_events_relations_label = [[event_relation + [0]*(max_event_num - len(event_relation)) for event_relation in events_relation] + 
                                                [[0] * max_event_num] * (max_event_num - len(events_relation))
                                                for events_relation in events_relation_list]
            

            batch_input_ids = torch.LongTensor(batch_input_ids)
            self.batch_examples.append({
                "input_ids": torch.LongTensor(batch_input_ids),
                "attention_mask": torch.LongTensor(batch_attention_mask),
                "triggers_label": torch.LongTensor(batch_triggers_label),
                "triggers_pos": torch.LongTensor(batch_triggers_pos),
                "triggers_mask": torch.LongTensor(batch_triggers_mask),
                "events_mask": torch.LongTensor(batch_events_mask),
                "events_polarity": torch.LongTensor(batch_events_polarity),
                "events_tags_label": torch.LongTensor(batch_events_tags_label),
                "events_tags_mask": torch.LongTensor(batch_events_tags_mask),
                "events_relations_label": torch.LongTensor(batch_events_relations_label)
            })
        
    def __iter__(self):
        
        self.batch_padding()
        self.idx = 0
        return self
    
    def __next__(self):
        if self.idx >= len(self.batch_examples):
            raise StopIteration
        data = self.batch_examples[self.idx]
        self.idx += 1
        return data

