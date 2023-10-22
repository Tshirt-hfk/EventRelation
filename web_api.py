# coding=utf-8
import torch
from flask import Flask, jsonify, abort, request
from dataloader import ID2TAG, LABEL2ER
from model import EventExtractModel
from predict import predict, predict_with_triggers
from transformers import AutoTokenizer
app = Flask(__name__)

tokenizer = AutoTokenizer.from_pretrained("../pretrain/chinese-roberta-wwm-ext")
model = EventExtractModel("../pretrain/chinese-roberta-wwm-ext")
model.load_state_dict(torch.load("./output/model_15.pt"))
mdoel = model.cuda()
model.eval()

@app.route('/')
def hello_world():
    return 'Hello World!'

@app.route("/event_relation_analysis", methods=["GET", "POST"])
def process():
    if request.method == "GET":
        text = request.args.get("text")
    if request.method == "POST":
        if request.content_type.startswith('application/json'):
            text = request.json.get('text')
        elif request.content_type.startswith('multipart/form-data'):
            text = request.form.get('text')
        else:
            text = request.values.get("text")
    input_text, triggers_pos_list, events_tags_list, events_relations_list = predict(model, tokenizer, text)
    return jsonify({
        "input_text": input_text,
        "triggers_pos_list": triggers_pos_list,
        "events_tags_list": events_tags_list,
        "events_relations_list": events_relations_list
    })


@app.route("/event_relation", methods=["POST"])
def process_v2():
    if request.content_type.startswith('application/json'):
        text = request.json.get('full_text')
        event_list = request.json.get('event_list')
    elif request.content_type.startswith('multipart/form-data'):
        text = request.form.get('full_text')
        event_list = request.form.get('event_list')
    else:
        text = request.values.get("full_text")
        event_list = request.values.get('event_list')
    event_num = len(event_list)
    eid_list = [event["eid"] for event in event_list]
    pos_list = [event["event_trigger_offset"] for event in event_list]

    try:
        input_text, triggers_pos_list, events_tags_list, events_relations_list = predict_with_triggers(model, tokenizer, text, pos_list)
        rsp_json = {
            "code": 200,
            "message": "success",
            "result": [
            ]
        }
        for i in range(event_num):
            for j in range(event_num):
                if events_relations_list[i][j] > 0:
                    rsp_json["result"].append({
                        "head_id": eid_list[i],
                        "tail_id": eid_list[j],
                        "relation": LABEL2ER[events_relations_list[i][j]]
                    })

    except Exception as e:
        rps_json = {
                "code": 500,
                "message": "%s" % e
            }
    
    return jsonify(rsp_json)

if __name__ == '__main__':
    app.run(debug=True)

# 访问链接 http://127.0.0.1:5000/event_relation_analysis?text=%E6%98%A8%E5%A4%A9%E6%B8%85%E6%99%A86%E6%97%B6%E8%AE%B8%EF%BC%8C%E4%B8%80%E8%BE%86%E4%B9%98%E5%9D%9012%E4%BA%BA%E7%9A%84%E8%B6%85%E8%BD%BD%E9%9D%A2%E5%8C%85%E8%BD%A6%E8%A1%8C%E9%A9%B6%E8%87%B3%E4%BA%AC%E6%89%BF%E9%AB%98%E9%80%9F%E8%BF%9B%E4%BA%AC%E6%96%B9%E5%90%91%E6%97%B6%E7%AA%81%E7%84%B6%E8%B5%B7%E7%81%AB%EF%BC%8C%E5%8F%B8%E6%9C%BA%E5%92%8C%E5%89%AF%E9%A9%BE%E9%A9%B6%E9%80%83%E7%94%9F%EF%BC%8C%E8%80%8C%E5%9D%90%E5%9C%A8%E8%BD%A6%E5%86%85%E7%9A%8410%E5%90%8D%E6%9C%A8%E5%B7%A5%E4%B8%8D%E5%90%8C%E7%A8%8B%E5%BA%A6%E7%83%A7%E4%BC%A4%EF%BC%8C%E5%85%B6%E4%B8%AD%E4%B8%80%E4%BA%BA%E6%AD%BB%E4%BA%A1%E3%80%82%E6%8D%AE%E4%BA%86%E8%A7%A3%EF%BC%8C%E9%9D%A2%E5%8C%85%E8%BD%A6%E5%8F%AF%E8%83%BD%E6%98%AF%E8%87%AA%E7%87%83%EF%BC%8C%E5%8F%B8%E6%9C%BA%E5%B7%B2%E8%A2%AB%E8%AD%A6%E6%96%B9%E5%B8%A6%E8%B5%B0%E8%B0%83%E6%9F%A5%E3%80%82