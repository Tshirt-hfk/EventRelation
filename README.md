# 事件抽取&事件关系预测
首先预测Denoter(Trigger)，再根据预测的Denoter(Trigger)的pos取embeding预测事件数据结构（标记）和事件之间关系
# 数据集
CEC 采用了 XML 语言作为标注格式，其中包含了六个最重要的数据结构（标记）：Event、Denoter、Time、Location、Participant 和 Object。Event用于描述事件；Denoter(Trigger)、Time、Location、Participant 和 Object用于描述事件的指示词和要素。此外，我们还为每一个标记定义了与之相关的属性。与ACE和TimeBank语料库相比，CEC语料库的规模虽然偏小，但是对事件和事件要素的标注却最为全面。
# 依赖
pip install -r requirements.txt
# 模型训练和使用
1. 处理CEC数据 python prepare_cec.py
2. 划分CEC数据 python split_dataset.py
3. 进行模型训练 python train.py
4. 模型离线推理 python predict.py
5. 提供web服务 python web_api.py
# 效果
#### 指标：              (f1, precision, recall)
#### trigger tags:       (0.8214, 0.8142, 0.8288)
#### event tags:         (0.7728, 0.7556, 0.7907)
#### event relations:    (0.5494, 0.5208, 0.5814)