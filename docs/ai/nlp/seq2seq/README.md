---
sidebar: auto
collapsable: true
---
# seq2seq
## seq2seq介绍


[seq2seq论文官方链接(点击查看)](https://arxiv.org/pdf/1409.3215)
### 论文简述

seq2seq的出现是为了解决深度神经网络不能用于解决序列映射序列任务的问题，团队使用多层LSTM将输入序列映射到一个固定维度的向量，然后使用另一个深度lstm从向量种解码目标序列，以达到序列映射到序列的目的。斌给钱在WMT14数据集中的英法互译任务中取得了不错的成绩。同时团队还得出了源句电刀规律：颠倒所有源句（而非目标句）中的词序能显著提高 LSTM 的性能，因为这样做在源句和目标句之间引入了许多短期依赖关系，从而使优化问题变得更容易。

### pytorch接口介绍


### 接口的简单调用


### 手动实现正向传播

## 总结

在本次实验中，我手动实现了rnn的正向传播，理解了普通rnn的推理过程，同时学习了lstm和gru的原理和代码实现，为后续深入学习rnn打下了基础。