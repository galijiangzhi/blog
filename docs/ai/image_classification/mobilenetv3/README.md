---
sidebar: auto
collapsable: true
---
# MobileNet_V2
## MobileNet_V2论文

[MobileNet_V2论文官方链接(点击查看)](http://xxx.itp.ac.cn/pdf/1905.02244)

[MobileNet_V2论文备用链接(点击查看)](http://www.apache2.sanyueyu.top/blog/ai/image_classification/MobileNet_V2/MobileNet_V2.pdf)

下面中文论文中有些图片分辨率太低了，图片方面可以参考上面备用链接里的图片

[MobileNet_V2论文中文pdf链接(点击查看)（本人翻译能力和手段有限，可以看看别人写的）](http://www.apache2.sanyueyu.top/blog/ai/image_classification/MobileNet_V2/MobileNet_V2cn.pdf)



## MobileNet_V2介绍

### inverted升维的数学原理

## 手动实现算法（准备阶段）
准备阶段我们要做一些准备工作同时处理一下数据集，这里我选择使用MNIST数据集

工作化境：
>CPU: I3 10105F （x86_64）<br>
>GPU: ASUS 3060 12G<br>
>RAM: 威刚 DDR4 40G 2666<br>
>主板：MSI B560M-A<br>
>硬盘：WDC SN550 1T<br>

>OS: UBUNTU22.04<br>
>python版本：3.11.7<br>，
>torch版本：2.2.1<br>
>jupyter notebook<br> 

**注意事项：本实验尽量在有gpu的平台进行，使用个人电脑的cpu也可以将模型优化到不错的状态**



### 检查算力平台情况

    import torch

    # 检查是否有可用的GPU
    if torch.cuda.is_available():
        # 获取GPU设备数量
        device_count = torch.cuda.device_count()
        print(f"发现 {device_count} 个可用的GPU 设备")
        # 获取每个GPU的名称
        for i in range(device_count):
            print(f"GPU 设备 {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("没有发现可用的GPU")

输出结果：

![](./2.png)


### 数据预处理

    from torchvision import transforms
    from torchvision import datasets
    from torch.utils.data import DataLoader
    import torch.nn.functional as F
    import torch.optim as optim
    import torch.nn as nn

    batch_size = 32
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,),(0.3081)),
        transforms.Resize((224, 224))
    ])
    train_dataset = datasets.MNIST(root='./dataset/mnist/',train=True,download=True,transform = transform)
    test_dataset = datasets.MNIST(root='./dataset/mnist/',train=False,download=True,transform = transform)
    train_loader = DataLoader(train_dataset,shuffle=True,batch_size=batch_size)
    test_loader = DataLoader(test_dataset,shuffle=True,batch_size=batch_size)


## 手动实现算法（动手阶段）
### 模型实现--构建模型
模型的内部结构参考下面这张官方给出的结构图：前面基本都是一样的，后面稍微做了修改

![](./3.png)

    class ConvBNReLU(nn.Module):
        def __init__(self, in_channels, out_channels, stride):
            super(ConvBNReLU, self).__init__()
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
            self.bn = nn.BatchNorm2d(out_channels)

        def forward(self, x):
            return F.relu6(self.bn(self.conv(x)))

    class InvertedResidual(nn.Module):
        def __init__(self, in_channels, out_channels, stride, expand_ratio):
            super(InvertedResidual, self).__init__()
            self.stride = stride
            self.use_res_connect = self.stride == 1 and in_channels == out_channels

            hidden_dim = int(in_channels * expand_ratio)
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )

        def forward(self, x):
            if self.use_res_connect:
                return x + self.conv(x)
            else:
                return self.conv(x)

    class MobileNetV2(nn.Module):
        def __init__(self, num_classes=1000):
            super(MobileNetV2, self).__init__()
            self.model = nn.Sequential(
                ConvBNReLU(1, 32, stride=2),
                InvertedResidual(32, 16, stride=1, expand_ratio=1),
                InvertedResidual(16, 16, stride=2, expand_ratio=6),
                InvertedResidual(16, 24, stride=2, expand_ratio=6),
                InvertedResidual(24, 24, stride=1, expand_ratio=6),
                InvertedResidual(24, 24, stride=1, expand_ratio=6),
                InvertedResidual(24, 32, stride=2, expand_ratio=6),
                InvertedResidual(32, 32, stride=1, expand_ratio=6),
                InvertedResidual(32, 32, stride=1, expand_ratio=6),
                InvertedResidual(32, 32, stride=1, expand_ratio=6),
                InvertedResidual(32, 64, stride=2, expand_ratio=6),
                InvertedResidual(64, 64, stride=1, expand_ratio=6),
                InvertedResidual(64, 64, stride=1, expand_ratio=6),
                InvertedResidual(64, 96, stride=1, expand_ratio=6),
                InvertedResidual(96, 96, stride=1, expand_ratio=6),
                InvertedResidual(96, 96, stride=1, expand_ratio=6),
                InvertedResidual(96, 160, stride=2, expand_ratio=6),
                InvertedResidual(160, 160, stride=1, expand_ratio=6),
                nn.Conv2d(160, 320, kernel_size=1, bias=False),
                nn.BatchNorm2d(320),
                nn.ReLU6(inplace=True),
            )

            self.avgpool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(320, num_classes)

        def forward(self, x):
            x = self.model(x)
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x
### 模型实现--构建训练和测试函数

首先构建损失函数和优化器

    import torch.optim as optim
    criterion = torch.nn.CrossEntropyLoss()#交叉熵损失
    optimizer = optim.SGD(model.parameters(),lr=0.05,momentum=0.5)

构建训练函数和测试函数

    def train(epoch):
        model.train()
        running_loss =0.0
        for batch_idx,data in enumerate(train_loader,0):
            inputs,labels = data
            # print(labels)
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            # print(outputs)
            loss = criterion(outputs,labels)
            loss.backward()
            optimizer.step()
    
            running_loss += loss.item()
            if batch_idx % 50 == 49:
                print(f'{epoch+1,batch_idx+1} loss :{running_loss/batch_idx}')

    def test():
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data in test_loader:
                images,labels = data
                images,labels = images.to(device),labels.to(device)
                outputs=model(images)
                _,predicted = torch.max(outputs.data,dim=1)#再1维度（横维度）查找最大的值，返回最大值,下标
                total += labels.size(0)
                correct +=(predicted == labels).sum().item()
                # for i in range(len(predicted)):
                    # print(f"预测值: {predicted[i]}, 真实值: {labels[i]}")
        print(f'{100*correct/total}')

### 模型实现--小批量随机梯度下降
运行训练函数（每训练一轮之后，保存一下模型，防止突然断电，血的教训）

    for i in range(5):
        train(i)
        torch.save(model,'MobileNet_V2_mnist.pth')

经过五轮训练，损失函数最终下降到0.016左右，而上次我们使用MobileNet_V1做实验的时候，损失为0.0175左右：

![](./4.png)


接下来观察一下测试情况

![](./5.png)

上次我们做实验的结果是MobileNet_V1的准确率为98.99，而这次使用V2的准确率为99.19，可以看到提升还是很大的。

至此实验完成。

## 总结

在这一次实验中我对残差神经网络有了一点新的认识，比如残差网络中维度匹配不是必要的，可以根据自己对精度-速度的取舍来判断要不要进行维度匹配。