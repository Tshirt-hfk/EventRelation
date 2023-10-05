# 事件抽取&事件关系预测
首先预测Denoter(Trigger)，再根据预测的Denoter(Trigger)的pos取embeding抽取事件数据结构（标记）和预测事件之间关系
# 数据集
CEC 采用了 XML 语言作为标注格式，其中包含了六个最重要的数据结构（标记）：Event、Denoter、Time、Location、Participant 和 Object。Event用于描述事件；Denoter(Trigger)、Time、Location、Participant 和 Object用于描述事件的指示词和要素。此外，我们还为每一个标记定义了与之相关的属性。与ACE和TimeBank语料库相比，CEC语料库的规模虽然偏小，但是对事件和事件要素的标注却最为全面。
