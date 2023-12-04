# coding:utf-8
from collections import defaultdict
import os, json
import xml.etree.ElementTree as ET
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("../pretrain/chinese-roberta-wwm-ext")


def parseID(id_strs):
    ids = []
    id_strs = id_strs.replace("，", ",")
    for id_str in id_strs.split(","):
        if "-" in id_str:
            s_id, e_id = id_str.split('-')
            for id in range(int(s_id[1:]), int(e_id[1:])+1):
                ids.append(id)
        else:
            ids.append(int(id_str[1:]))
    # print(id_strs, ids)
    return [id-1 for id in ids]


def find_sublist(cutted_text, content_cutted_text, start=0):
    for i in range(start, len(content_cutted_text)-len(cutted_text)+1):
        j = 0
        while j < len(cutted_text):
            if cutted_text[j] != content_cutted_text[i+j]:
                break
            j += 1
        if j == len(cutted_text):
            return [i, i+j]
    # print(content_cutted_text[start:])
    # print(cutted_text)
    # print("-----------------------")
    return [-1, -1]

event_relation_type = set()


def parse_xml_content(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    content = root.find('Content')
    eRelation = root.findall('eRelation')
    content_text = ''.join([text.strip() for text in content.itertext()])
    content_cutted_text = tokenizer.tokenize(content_text)
    events = content.findall('.//Event')

    # (eid, polarity, trigger, time, loc, object_s, object_o)
    event_list = []
    for event in events:
        polarity = "Positive"
        if event.get('polarity') == "negative":
            polarity = "Negative"
        event_list.append({
            'eid': int(event.get('eid')[1:])-1,
            'polarity': polarity,
            'trigger': None,
            'time': None,
            'loc': None,
            'object_s': [],
            "object_o": []
        })
    event_list.sort(key=lambda x:x['eid'])
    start = 0
    for event in events:
        eid = int(event.get('eid')[1:])-1
        event_text = ''.join([text.strip() for text in event.itertext()])
        event_cutted_text = tokenizer.tokenize(event_text)
        start, _ = find_sublist(event_cutted_text, content_cutted_text, start)
        for child in event:
            if child.tag=="Denoter":
                id = eid
                text = child.text.strip()
                cutted_text = tokenizer.tokenize(text)
                offset = find_sublist(cutted_text, content_cutted_text, start)
                event_list[id]['trigger'] = {
                    "text": cutted_text,
                    "type": child.get("type"),
                    "offset": offset
                }
            elif child.tag=="Time":
                id = eid
                text = child.text.strip()
                cutted_text = tokenizer.tokenize(text)
                offset = find_sublist(cutted_text, content_cutted_text, start)
                event_list[id]['time'] = {
                    "text": cutted_text,
                    "type": child.get("type"),
                    "offset": offset
                }
            elif child.tag=="Location":
                id = eid
                text = child.text.strip()
                cutted_text = tokenizer.tokenize(text)
                offset = find_sublist(cutted_text, content_cutted_text, start)
                event_list[id]['loc'] = {
                    "text": cutted_text,
                    "offset": offset
                }
            elif child.tag=="Object" or child.tag=="Participant":
                sid = child.get('sid')
                if sid:
                    for id in parseID(sid):
                        text = child.text.strip()
                        cutted_text = tokenizer.tokenize(text)
                        offset = find_sublist(cutted_text, content_cutted_text, start)
                        event_list[id]['object_s'].append({
                            "text": cutted_text,
                            "offset": offset
                        })
                oid = child.get('oid')
                if oid:
                    for id in parseID(oid):
                        text = child.text.strip()
                        cutted_text = tokenizer.tokenize(text)
                        offset = find_sublist(cutted_text, content_cutted_text, start)
                        event_list[id]['object_o'].append({
                            "text": cutted_text,
                            "offset": offset
                        })
        
    event_relation_list = []
    for e_relation in eRelation:
        eid_s = e_relation.get('cause_eid') or e_relation.get('thoughtevent_eid') or e_relation.get('bevent_eid') or e_relation.get('accompanya_eid') or e_relation.get('fevent_eid') or e_relation.get('concurrencya_eid')
        eid_s = int(eid_s[1:])-1
        for eid_o in parseID(e_relation.get('effect_eid') or e_relation.get('thoughtcontent_eids') or e_relation.get('aevent_eid') or e_relation.get('accompanyb_eid') or e_relation.get('sevent_eids') or e_relation.get('concurrencyb_eid')):
            event_relation_list.append({
                "type": e_relation.get('relType'),
                "eid_s": eid_s,
                "eid_o": eid_o,
            })
    
    data = {
        "sentence": content_cutted_text,
        "events": event_list,
        "relations": event_relation_list
    }
    return [data]


def parse_xml_paragraph(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    content = root.find('Content')
    eRelation = root.findall('eRelation')
    event_relation_list = []
    for e_relation in eRelation:
        eid_s = e_relation.get('cause_eid') or e_relation.get('thoughtevent_eid') or e_relation.get('bevent_eid') or e_relation.get('accompanya_eid') or e_relation.get('fevent_eid') or e_relation.get('concurrencya_eid')
        eid_s = int(eid_s[1:])-1
        for eid_o in parseID(e_relation.get('effect_eid') or e_relation.get('thoughtcontent_eids') or e_relation.get('aevent_eid') or e_relation.get('accompanyb_eid') or e_relation.get('sevent_eids') or e_relation.get('concurrencyb_eid')):
            event_relation_list.append({
                "type": e_relation.get('relType'),
                "eid_s": eid_s,
                "eid_o": eid_o
            })
            if eid_s == eid_o:
                print(e_relation.get('relType'))
            # if "Composite" == e_relation.get('relType'):
            #     print(e_relation.get('relType'))
            event_relation_type.add(e_relation.get('relType'))
    paragraphs = content.find('.//Paragraph')
    
    data = []
    for paragraph in paragraphs:
        paragraph_text = ''.join([text.strip() for text in paragraph.itertext()])
        paragraph_cutted_text = tokenizer.tokenize(paragraph_text)
        events = paragraph.findall('.//Event')

        # (eid, polarity, trigger, time, loc, object_s, object_o)
        event_dict = {}
        for event in events:
            polarity = "Positive"
            if event.get('polarity') == "negative":
                polarity = "Negative"
            elif event.get('polarity') is not None:
                print(event.get('polarity'))
            event_dict[int(event.get('eid')[1:])-1] = {
                        'eid': int(event.get('eid')[1:])-1,
                        'polarity': polarity,
                        'trigger': None,
                        'time': None,
                        'loc': None,
                        'object_s': [],
                        "object_o": []
                    }
        start = 0
        for event in events:
            eid = int(event.get('eid')[1:])-1
            event_text = ''.join([text.strip() for text in event.itertext()])
            event_cutted_text = tokenizer.tokenize(event_text)
            start, _ = find_sublist(event_cutted_text, paragraph_cutted_text, start)
            for child in event:
                if child.tag=="Denoter":
                    id = eid
                    text = child.text.strip()
                    cutted_text = tokenizer.tokenize(text)
                    offset = find_sublist(cutted_text, paragraph_cutted_text, start)
                    event_dict[id]['trigger'] = {
                        "text": cutted_text,
                        "type": child.get("type"),
                        "offset": offset
                    }
                elif child.tag=="Time":
                    id = eid
                    text = child.text.strip()
                    cutted_text = tokenizer.tokenize(text)
                    offset = find_sublist(cutted_text, paragraph_cutted_text, start)
                    event_dict[id]['time'] = {
                        "text": cutted_text,
                        "type": child.get("type"),
                        "offset": offset
                    }
                elif child.tag=="Location":
                    ids = parseID(child.get('lid'))
                    for id in ids:
                        if id not in event_dict.keys():
                            continue                 
                        text = child.text.strip()
                        cutted_text = tokenizer.tokenize(text)
                        offset = find_sublist(cutted_text, paragraph_cutted_text, start)
                        event_dict[id]['loc'] = {
                            "text": cutted_text,
                            "offset": offset
                        }
                elif child.tag=="Object" or child.tag=="Participant":
                    sid = child.get('sid')
                    if sid:
                        for id in parseID(sid):
                            if id not in event_dict.keys():
                                continue
                            text = child.text.strip()
                            cutted_text = tokenizer.tokenize(text)
                            offset = find_sublist(cutted_text, paragraph_cutted_text, start)
                            event_dict[id]['object_s'].append({
                                "text": cutted_text,
                                "offset": offset
                            })
                    oid = child.get('oid')
                    if oid:
                        for id in parseID(oid):
                            if id not in event_dict.keys():
                                continue
                            text = child.text.strip()
                            cutted_text = tokenizer.tokenize(text)
                            offset = find_sublist(cutted_text, paragraph_cutted_text, start)
                            event_dict[id]['object_o'].append({
                                "text": cutted_text,
                                "offset": offset
                            })

        event_list = sorted(list(event_dict.values()), key=lambda x:x['eid'])
        event_ids = {}
        for i, x in enumerate(event_list):
            event_ids[x['eid']] = i
            x.pop('eid')
        p_event_relation_list = []
        for er in event_relation_list:
            eid_s = er['eid_s']
            eid_o = er['eid_o']
            if eid_s in event_ids.keys() and eid_o in event_ids.keys():
                p_event_relation_list.append({
                    "type": er['type'],
                    "eid_s": event_ids[eid_s],
                    "eid_o": event_ids[eid_o]
                })
        data.append({
            "sentence": paragraph_cutted_text,
            "events": event_list,
            "relations": p_event_relation_list
        })
    return data


paragraph_data = []
content_data = []
for filepath, dirnames, filenames in os.walk('./CEC'):
    for filename in filenames:
        print(os.path.join(filepath, filename))
        paragraph_data += parse_xml_paragraph(os.path.join(filepath, filename))
        content_data += parse_xml_content(os.path.join(filepath, filename))     
print("event relation type:", event_relation_type)
print("-------------------------------------------------------------------------")
data_len = defaultdict(int)
for line in paragraph_data:
    sentence = line['sentence']
    data_len[len(sentence)] += 1
data_len = sorted([(l, n) for l,n in data_len.items()], key=lambda x:x[0])
print("paragraph data length:", data_len)
print("-------------------------------------------------------------------------")
data_len = defaultdict(int)
for line in content_data:
    sentence = line['sentence']
    data_len[len(sentence)] += 1
data_len = sorted([(l, n) for l,n in data_len.items()], key=lambda x:x[0])
print("content data length:", data_len)

with open("./data/cec.json", "w", encoding='utf8') as f:
    for line in paragraph_data+content_data:
        if len(line['sentence']) > 510: continue
        f.write(json.dumps(line, ensure_ascii=False)+"\n")