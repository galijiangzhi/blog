---
sidebar: auto
collapsable: true
---
# SSD
## SSD简介
SSD:直接回归目标类别和位置，在不同尺度的特征图上进行预测，端到端的训练，不挑分辨率

[overfeat论文官方链接(点击查看)](https://arxiv.org/abs/1312.6229)

[overfeat论文pdf链接(点击查看)](http://www.apache2.sanyueyu.top/blog/ai/object_detection/overfeat/overfeat.pdf)

[overfeat中文论文pdf链接(点击查看)（用ai&有道词典翻译的，质量一般）](http://www.apache2.sanyueyu.top/blog/ai/object_detection/overfeat/overfeatcn.pdf)

### 论文简述
传统的物体检测依赖于手工设计特征提取算法，泛用性很差。OverFeat算法整合了分类、定位和检测三个任务，通过使用滑动窗口方式在不同尺度上应用卷积神经网络。该框架使用单个网络在图像分类、定位和检测这三个方向上同时进行训练。利用卷积神经网络在不同任务之间的特征共享特性，提高每个任务的效果。


## 手动实现算法（准备阶段）

