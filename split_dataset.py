# coding=utf-8
import json
import random

random.seed(555)

input_files = ["./data/cec.json"]
train_file = "./data/train.json"
dev_file = "./data/dev.json"

dev_data_num = 50

total_data = []
train_data = []
dev_data = []

for input_file in input_files:
    with open(input_file, "r", encoding="utf-8") as f:
        total_data += f.readlines()

random.shuffle(total_data)
print("total data: ", len(total_data))
train_data, dev_data = total_data[:-50], total_data[-50:]
print("train data: ", len(train_data))
print("dev data: ", len(dev_data))

with open(train_file, "w", encoding="utf-8") as f:
    for line in train_data:
        f.write(line)

with open(dev_file, "w", encoding="utf-8") as f:
    for line in dev_data:
        f.write(line)

