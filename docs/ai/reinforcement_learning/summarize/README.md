---
sidebar: auto
collapsable: true
---
# reinforcement_learning

## 强化学习概述

强化学习，是在与环境的互动当中，为了达成一个目标而进行的学习过程。其主要的要素包括：agent，environment，goal。

>> environment 指的是环境，可以理解为一个游戏
>
>> goal 指的是目标
>
>> agent 指的是与环境互动的主体，可以理解为玩游戏的玩家

当有了玩家和环境之后，我们就要设计这个游戏的玩法，玩法的主要要素包括：state，action，reward。

>> state指的是状态，可以理解为玩家和环境的状态，这个状态很广泛，举例说明的话，状态可以包括比如自身和对手的位置，等级等一系列当前确定的信息。
>
>> action指的是行动，指的是agent通过当前的state做出来的行动决策。
>
>> reward指的是奖励，指的是，当agent做出action之后得到的即时反馈，在强化学习中reward通常是一个实数，并且可能是0，比如在五子棋比赛中，只有当agent赢得比赛，我们才会给他一个大于0的reward。reward的设置是很重要的，比如在一场篮球赛中，我们的reward如果设置成进球可以获得奖励，那我们的模型就会更倾向于进攻，但篮球的规则是最终得分多的队伍才是赢家，所以一个合理的奖励应该是在己方进球的同时防止对方进球。

接下来是强化学习的核心结构：policy，value。

>> policy指的是策略，策略是指，在某个状态下应该做出什么操作，策略本身是一个函数，当我们把state输入进去，策略函数会告诉我们需要采取什么样的action。
>
>> value指的是价值，价值本身也是一个函数，当我们把状态输入进去，价值函数会对玩家将来能获得的奖励进行一个评估。

强化学习的核心就是怎么通过当前的状态来进行下一步行动，以及怎么让玩家在最后得到更高的价值。

强化学习的特点：<br>
trial and error （试错） 强化学习是一种试错学习，也就是在不断地尝试中学习。<br>
delayed reward （延迟奖励） 行动没有对应即时的奖励，但是他一定有价值，比如下棋，每一步棋都没有实质的奖励，直到最后胜利，才会有奖励。<br>

强化学习的核心问题：exploration vs exploitation(探索与开发) 在强化学习中，不应一味使用当前的最优方案，这样会使我们的模型陷入一个局部最优解，应该一边完善当前的解一边探索新的解
## 强化学习例子 K-armed Bandit（多臂老虎机）

### enviroment

现在我们有两台老虎机，分别叫“左老虎机”和“右老虎机”，在后台规则中“左老虎机”每次启动都会获得一个奖励，奖励的值是服从均值500，标准差为50的正态分布，而启动“右老虎机”得到的奖励值服从均值550，标准差为100的正态分布。作为玩家的agent并不知道这两台老虎机的奖励的分布规律。

### goal

我们希望在进行启动老虎机之后能得到更多的奖励值，这也是我们这次实验的最终目的，

### value

我们使用平均奖励作为奖励的评估，这一行动价值的估计方法称为sample-average（样本平均），是最简单的一种方法。

### policy

策略函数的选择有很多种，最直接的方法就是选择启动当前时刻价值最大的老虎机，这一策略被称为greedy(贪婪策略)，利用我们已经学习到的价值函数选择价值最大的老虎机，但是这个策略并不是很好的策略，假设我们先启动了“左老虎机”得到了510的价值，此时“右老虎机”的价值还是0，如果使用贪婪策略，则模型会一直选择“左老虎机”。此时我们可以设计一个补救策略，就是在开始的时候强行让所有老虎机都启动一次，再做贪婪选择。但是，因为随机性的存在，只启动一次显然是不够的。为此我们设计了一个新的策略，让两台老虎机的初始值很大，比如1000，随着启动次数的增加，最后两个老虎机的期望值会逐渐逼近他们的后台配置值。同时为了加大随机性以防止模型陷入死角，我们提出了一种新的策略：ε-greedy,其含义是，在大部分情况下使用贪婪策略，以一定概率ε执行随机策略，在不同的问题中ε的最优值也是不同的。并且我们可以随着时间变化修改ε的值。

### 代码构建

首先：创建老虎机类，定义启动函数，计算奖励的函数

    import numpy as np
    import random

    class laohuji:#创建老虎机类
        def __init__(self,mean,std_dev):
            self.q = 1000
            self.n = 1
            self.v = 1000
            self.mean = mean
            self.std_dev = std_dev
            self.v_list = []
        def start(self):
            q = int(np.random.normal(self.mean, self.std_dev))
            self.q += q
            self.n +=1
            return q
        def value(self):
            self.v = self.q/self.n
            self.v_list.append(int(self.v))
            return self.v

然后创建active函数：

    def action():#模型决定的操作
        print(f'左老虎机价值：{zuolaohuji.value()}')
        print(f'右老虎机价值：{youlaohuji.value()}')
        if zuolaohuji.value() > youlaohuji.value():
            q = zuolaohuji.start()
            print(f'启动左老虎机，奖励为{q}')
        else:
            q = youlaohuji.start()
            print(f'启动右老虎机，奖励为{q}')

初始化“左老虎机”和“右老虎机”：

    zuolaohuji = laohuji(500,50)
    youlaohuji = laohuji(550,100)


使用ε-greedy方法，这里我设置了十分之一的概率随机：

    for i in range (1000):
        if random.random() < 0.1:
            if random.random() < 0.5:
                zuolaohuji.start()
            else:
                youlaohuji.start()
        else:
            action()

观察两台老虎机的value：

    import matplotlib.pyplot as plt

    plt.plot(zuolaohuji.v_list, label='zuolaohuji')
    plt.plot(youlaohuji.v_list, label='youlaohuji')

    plt.legend()

    plt.grid(True)

    plt.xlabel('n')
    plt.ylabel('v')

    plt.show()

结果如下图：

![](./1.png)

稍后我们可以启动一下active函数，看看模型会做出什么决策：

    action()
    
![](./2.png)

可以观察到，模型的决策还是没有什么问题的。

## 强化学习误差

首先我们回忆以下使用sample average样本平均来估计价值的方法，假设Qn+1是采取n次行动后对行动价值的估计值，Ri是第i次操作产生的实际价值，根据sample average我们可以得到下面这个公式，我们对他进行推导如图所示：
![](./3.png)
我们对最后得到的推到公式进行解释：Qn是在第n次行动之前的价值估计，Qn+1是第n次行动之后的价值估计，(Rn-Qn)则是误差，也就是之前行动中预测值与实际值之间的偏差，1/n则是学习率，也就是根据误差对新预测结果的调整幅度，当我们把学习率设置为1/n的时候，越往后这个学习率会越小，也就是模型对后面的行动的价值重视程度越来越低，如果我们不使用1/n，转而使用一个常数a，则刚好相反，模型会更看重最近得到的奖励的情况。这种方法也被称为weighted average(加权平均)。这里需要注意的是，sample average中初始值对最后的价值估计影响很小，几乎可以忽略不计，但是weighted average中初始值对最后的价值估计影响可能比较大。<br>



## 华为人工智能发展战略

## 人工智能的正义和未来
