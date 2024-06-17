---
sidebar: auto
collapsable: true
---
# GoogLeNet
## GoogLeNet介绍

[GoogLeNet_v1论文官方链接(点击查看)](https://arxiv.org/abs/1409.4842)

[GoogLeNet_v1论文备用链接(点击查看)](http://www.apache2.sanyueyu.top/blog/ai/image_classification/googlenet/GoogLeNet_v1.pdf)

下面中文论文中有些图片分辨率太低了，图片方面可以参考上面备用链接里的图片，而且GoogLeNet的论文写的巨抽象，大概是因为我见识浅薄，读起来很吃力

[GoogLeNet_v1论文中文pdf链接(点击查看)（本人翻译能力和手段有限，可以看看别人写的）](http://www.apache2.sanyueyu.top/blog/ai/image_classification/googlenet/GoogLeNet_v1cn.pdf)

首先，在引言部分，GoogLeNet团队提出，该算法设计的原则不是一味追求准确率，在大部分实验中模型设计都是为了在推理的时候保持150亿次算数运算，这样可以让模型可以实际投入到边缘计算项目中，而不是只能出现在实验室中

其次，GoogLeNet团队根据"Network in network"这篇论文中提到的技术，在模型中使用1\*1的卷积对模型参数进行简化