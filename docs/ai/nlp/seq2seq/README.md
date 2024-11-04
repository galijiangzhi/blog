---
sidebar: auto
collapsable: true
---
# RNN
## RNN介绍
### 基本介绍
RNN神经网络现在一般称为循环神经网络，但是从模型的运算过程来讲应该翻译成递归神经网络，这是一种用于处理序列数据的神经网络架构。与传统的前馈神经网络不同，RNN具有循环连接的特性，使其能够在时间维度上保留和利用之前的信息。这使得RNN特别适合处理时间序列数据、自然语言处理、语音识别等任务。

RNN的基本单元可以看作是一个包含隐藏状态的神经元，每个时间步的输入不仅以来于当前的输入，还依赖于前一个时间步的隐藏状态，这样说可能还是有点抽象，这样我们用一个实际的例子来解释，假设我们输入了一段文本：“张三在电子产品市场买了一个苹果14”,在一个理想化的没有关联性的模型（非RNN模型）中对文本进行分析：他可能是这样的：<br>
![](./1.png)<br>
我们可以看到，模型可以识别出这些词的意思，但是因为没有联系上下文所以识别出来的这个意思也不是很准确，比如这里的苹果经过模型就会识别成蔷薇科-苹果属-家苹果种的那个苹果。而循环神经网络可以将上一次的输出传递给下一次模型的计算，让模型有了联系上下文的能力，如下图所示<br>
![](./2.png)<br>
此时，在上图中的输出h(5)中：苹果就不太可能被识别成蔷薇科-苹果属-家苹果种的那个苹果，因为在上文中提到了电子市场，所以这个苹果更有可能被识别成苹果公司推出的电子产品iphone，h(6)中的14更有可能被识别成iphone的型号，而不是数量。

至此我们可以看出，在序列数据中，上下文的联系对于我们提取序列信息具有重要作用。

### 结构介绍

下面我们对RNN的结构进行一个深入的认识：

**隐藏层**：RNN的隐藏层也可以称为记忆单元，该结构在获取过去信息的同时，接受当前的数据，记忆单元有一些改进版本，比如长短期记忆网络(LSTM)和门控循环单元(GRU)，这些改进的版本包含了更复杂的记忆单元设计，以更好地处理长序列数据。<br>
**模型结构**：RNN的主要结构包括：单向循环，双向循环，多层的单向或双向叠加。

>>单向循环神经网络：单向循环神经网络是指信息在网络中只能按照一个方向传播，通常是从输入序列的第一个时间步到最后一个时间步。这种结构适用于顺序敏感的任务，如语言建模、情感分析等。
>
>>双向循环神经网络：双向循环神经网络是指在网络中同时使用两个单向RNN，一个按照正向顺序传播信息，另一个按照逆向顺序传播信息。这样可以更好地捕捉序列数据中的双向依赖关系，提高模型的性能。
>
>>多层的单向或双向循环神经网络：多层的单向或双向循环神经网络是指在网络中堆叠多个RNN层，可以是单向RNN或双向RNN。通过堆叠多个循环层，可以增加网络的表示能力，使得网络可以学习更复杂的模式和关系。

**RNN的优点**：可以处理变长序列，模型大小于序列长度无关，计算量与训练长度呈线性增长，考虑历史信息，便于流式输出。<br>
**RNN的缺点**：串行计算比较慢，无法获取太长的历史信息。

### pytorch接口介绍

pytorch的torch库种提供了torch.nn.RNN()接口，方便我们构建RNN模型。<br>
torch.nn.RNN()继承自torch.nn.Module,其主要参数包括如下：<br>

>>input_size 输入特征的维度，即每次输入的特征数
>
>>hidden_size 隐藏状态的维度，即每个隐藏层的输出特征的维度
>
>>num_layers ：rnn的层数，默认为1，可以通过堆叠多个rnn来构建更急深的网络
>
>>nonlinearity :默认非线性激活函数，一般用tanh或者relu，默认值为tanh
>
>>bias：是否使用偏执，默认为true，如果设置成false，则不使用偏置。
>
>>batch_first：设置输入输出格式，如果设置为true，则输入和输出的形状为（batch,seq,feature），否则为(seq,batch,feature)。默认为false。<br>
    batch：批次大小，表示一次训练或推理中输入的样本数量。批次大小通常是一个超参数，可以根据可用的内存和计算资源进行调整。使用批次处理可以提高训练效率并加速收敛。<br>
    seq，序列长度，表示每个输入序列的时间步数或长度。在处理时间序列数据（如文本、语音或其他序列数据）时，序列长度是指序列中包含的时间步的数量。例如，在自然语言处理中，一个句子的单词数量可以被视为序列长度。<br>
    feature：特征维度，表示每个时间步的特征数量。特征维度是指在每个时间步中输入的特征的数量。例如，在处理图像序列时，特征维度可能是图像的通道数（如 RGB 图像的特征维度为 3）；在处理文本时，特征维度可能是词嵌入的大小。
>
>>dropout：设置丢弃率
>
>>bidirectional：如果设置为true，则为双向rnn，默认为false
>
>>proj_size:尽在使用投影rnn时有效，指定投影的输出大小。默认为0，表示没有投影

### 接口的简单调用

这里我们可以调用一些简单的接口来做一下小实验，代码如下：

    import torch
    import torch.nn as nn

    model = nn.RNN(4,3,1,batch_first=True)
    input = torch.randn(1,2,4)
    output,h_n=model(input)
    print(f"单向rnn\toutput:\n\t{output}")
    print(f"单向rnn\th_n:\n\t{h_n}")
    model = nn.RNN(4,3,1,batch_first=True,bidirectional=True)
    output,h_n=model(input)
    print(f"双向rnn\toutput:\n\t{output}")
    print(f"双向rnn\th_n:\n\t{h_n}")

输出结果如图：<br>
![](./3.png)<br>

这里我们构建了一个单向rnn和一个双向rnn，观察结果可以发现，双向rnn在输出结果的时候，输出的每个结果序列都比单向rnn要长一倍，在输出最后一次计算的结果序列时，双向rnn比单向rnn多了一个维度，这是因为双向rnn在计算的时候不仅需要计算正向结果，还需要计算反向结果。

### 手动实现正向传播

首先我们看一下pytorch提供的正向传播公式，可以直接看下图，或者[点击此处](https://pytorch.org/docs/stable/generated/torch.nn.RNN.html)进入pytorch官网查看文档：<br>
![](./4.png)

通过这个公式我们知道，想要正向推理RNN的结果，需要以下组成部分：
>>Xt：输入序列种第t个时间步的输入数据
>
>>Wih：输入到隐藏层的权重矩阵
>
>>bih：输入到隐藏层的偏执向量
>
>>ht-1:上一个时间步的隐藏状态
>
>>whh：隐藏层到隐藏层的权重矩阵
>
>>bhh:隐藏层到隐藏层的偏置向量

这些部分的工作状态如下图所示，入口为Xt（在图中发光的块），图中的箭头，红色为张量乘操作，紫色为张量加操作，黑色矩形框框起来的就是隐藏层所做的操作:
![](./5.png)

接下来我们使用代码手动复刻以下这个结构：

    def rnn_forward(input,weight_ih,weight_hh,bias_ih,bias_hh,h_prev):
        bs,t,input_size = input.shape#获取批次大小，序列长度和输入维度
        h_dim = weight_ih.shape[0]#获取输入->隐藏层的权重矩阵的第一个维度，用来构建输出矩阵
        # print(weight_ih)
        h_out = torch.zeros(bs,t,h_dim)#构建输出矩阵
        # print(h_out)
        for i in range(t):#循环遍历每一个时间步
            # print(input[:,i,:])
            x = input[:,i,:].unsqueeze(2)#这一步让原本的行数据变成列数据，好用来做向量乘法
            w_ih_batch = weight_ih.unsqueeze(0).tile(bs,1,1)
            w_hh_batch = weight_hh.unsqueeze(0).tile(bs,1,1)
            w_times_x = torch.bmm(w_ih_batch,x).squeeze(-1)#讲数据和权重进行并行乘法运算，然后删除最后一个维度
            w_times_h = torch.bmm(w_hh_batch,h_prev.unsqueeze(2)).squeeze(-1)
            #下面的代码将前一时刻的数据处理结果和当前输入的数据处理结果相加，再加上偏置
            h_prev = torch.tanh(w_times_x+bias_ih+w_times_h+bias_hh)
            h_out[:,i,:] = h_prev #记录结果
        return h_out,h_prev.unsqueeze(0)

随后我们验证一下这个结构是否正确，我们将rnn_forward的结果（下图中绿色框）和pytorch api的结果（下图中红色框）进行比较：

    bs,t = 2,3 #批次大小，输入序列长度
    input_size,hidden_size = 2,3 #输入特征维度，隐含层特征维度
    input = torch.randn(bs,t,input_size)
    print(f'输入特征张量为:{input}')

    h_prev = torch.zeros(bs,hidden_size)
    rnn = nn.RNN(input_size,hidden_size,batch_first=True)

    rnn_out,rnn_n = rnn(input,h_prev.unsqueeze(0))
    print(f"\n\npytorch\toutput:\n\t{rnn_out}")
    print(f"pytorch\th_n:\n\t{rnn_n}")

    f_out,f_n=rnn_forward(input,rnn.weight_ih_l0,rnn.weight_hh_l0,rnn.bias_ih_l0,rnn.bias_hh_l0,h_prev)
    print(f"\n\nmy_function\toutput:\n\t{rnn_out}")
    print(f"my_function\th_n:\n\t{rnn_n}")

结果如下，我们可以看到，我们写的函数和pytorch提供的api计算的结果相同：
![](./6.png)

我们再试一下双向rnn，双向rnn的参数量为单向的两倍，内部操作是：先将序列正向算一遍，然后再反向算一遍，然后把结果拼接起来，这里注意反向拼接的时候需要对结果进行翻转，我们直接看代码：

    def bidirectional_rnn_forward(input,weight_ih,weight_hh,bias_ih,bias_hh,h_prev,\
                                weight_ih_reverse,weight_hh_reverse,bias_ih_reverse,bias_hh_reverse,h_prev_reverse):
        bs,t,input_size = input.shape
        h_dim = weight_ih.shape[0]
        h_out = torch.zeros(bs,t,h_dim*2)
        forward_output = rnn_forward(input,weight_ih,weight_hh,bias_ih,bias_hh,h_prev)[0]#正向推理结果
        backward_output = rnn_forward(torch.flip(input,[1]),weight_ih_reverse,weight_hh_reverse,bias_ih_reverse,bias_hh_reverse,h_prev_reverse)[0]#反向推理结果
        backward_output = torch.flip(backward_output,[1])#反向推理结果翻转
        h_out[:,:,:h_dim] = forward_output
        h_out[:,:,h_dim:] = backward_output
        print(h_out)

    bs,t = 2,3 #批次大小，输入序列长度
    input_size,hidden_size = 2,3 #输入特征维度，隐含层特征维度
    model = nn.RNN(input_size,hidden_size,batch_first=True,bidirectional=True)
    print("pytorch_api\toutput:\n")
    print(model(input)[0])
    h_prev=torch.zeros(2,bs,hidden_size)
    print("my_function\toutput:\n")
    bidirectional_rnn_forward(input,model.weight_ih_l0,model.weight_hh_l0,model.bias_ih_l0,model.bias_hh_l0,h_prev[0],\
                            model.weight_ih_l0_reverse,model.weight_hh_l0_reverse,model.bias_ih_l0_reverse,model.bias_hh_l0_reverse,h_prev[1])
                        
输出结果如下，pytorch_api计算的结果（红色框框选）和我们的双向rnn函数（绿色框框选）计算的结果一致。
    
![](./7.png)

## LSTM（长短时记忆网络）

长短时记忆网络（Long Short-Term Memory，LSTM）是一种特殊的循环神经网络（RNN）架构，专门设计用于解决传统RNN中存在的长期依赖问题。LSTM通过引入门控机制来控制信息的流动，从而更好地捕捉和记忆长序列中的重要信息。

RNN是想把所有的信息都记住，不管是有用的信息还是没用的信息，而LSTM设计了一个记忆细胞，具备选择性记忆的功能，可以选择重要信息记忆，过滤掉噪声信息，减轻记忆负担。

### 原理

![](./8.png)

LSTM的正向传播过程如上图所示，我们再对比以下rn的正向传播，这里我给出rnn的正向传播过程，如下图所示：

![](./9.png)

我们可以看出，rnn的内部比lstm简单很多，同时rnn的输入只有一个，而lstm的输入有两个。

这里我们将lstm的模块单独拿出来讲解，如下图所示：

![](./10.png)

这里Ct-1指的是上一时刻的记忆细胞，ht-1指的是上一时刻的状态，σ指的是门单元，输出值是0~1之间的一个值，ft是遗忘门，it是输入门，ot是输出门，每个模块对应的公式如下：

![](./11.png)

接下来从公式的角度来解释一下lstm单元：

Xt代表当前时间步的输入，ht-1表示上一个时间步的输出，这两个张量通过各自的W矩阵计算之后相加，作为三个σ的输入，ft遗忘门和ot输出门是相加矩阵直接进入σ，it更新们则是将进行σ和tanh操作之后相乘，最后这三个门会用于对Ct-1的加工，分别负责遗忘信息，输入信息和输出信息。

![](./12.png)

记忆细胞：记忆细胞的结构如上图所示，在lstm的每个时间步里，都有一个记忆细胞，这个东西给予了lstm选择记忆功能，使得lstm有能力自由选择每个时间步里记忆的内容。

这里我们模拟一个期末考试的场景对记忆细胞进行深入理解，假设Xt输入是线性代数，我们希望学习线性代数来应对考试，首先ft门的任务就是，遗忘掉记忆细胞中和线性代数无关的东西，
实现方法就是对应元素相乘，σ的输出都是0~1的数，和线性代数相关性大的部分就接近1，关系不大的就接近0，这样相乘之后越接近1的内容保留的就越好。

it代表输入门，由两个分支组成，分别是σ和tanh，tanh就是传统rnn神经网络中生成信息的方法，σit是输入们，他也会生成一个张量，里面越接近线性代数的东西对应的值越大，对于没有用的知识保留的程度则很小，比如说在学习线性代数的时候，他会尽可能保留数学知识，但是对于数学历史之类考试不考的东西保留的程度很小。

经过ft和it的更新之后，我们会生成新的记忆，也就是Ct，Ct就是之前的记忆经过“遗忘不重要的东西”，“记住新的重要的东西”之后得到的新的记忆。

输出门ot则是使用新的记忆进行tanh，进行门操作，保留重要的部分，同时遗忘不重要的东西，作为本次的输出。

至此，lstm的原理我们就理解完成了。 

在rnn中，由于记忆单元和门控制机制的存在，可以缓解rnn梯度消失的问题，记忆单元有点像cnn中的残差，可以让关键信息一直传递下去。

### 代码实现

首先导包，确认gpu状态

    import torch
    import torch.nn as nn
    if torch.cuda.is_available():
        # 获取GPU设备数量
        device_count = torch.cuda.device_count()
        print(f"发现 {device_count} 个可用的GPU 设备")
        # 获取每个GPU的名称
        for i in range(device_count):
            print(f"GPU 设备 {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("没有发现可用的GPU")

然后创建lstm基本单元，这里要根据下面的公式创建：

![](./13.png)

    class lstm_cell(nn.Module):
        def __init__(self,in_dim,hidden_dim):
            super(lstm_cell,self).__init__()
            self.ix_linear = nn.Linear(in_dim,hidden_dim)
            self.ih_linear = nn.Linear(hidden_dim,hidden_dim)
            self.fx_linear = nn.Linear(in_dim,hidden_dim)
            self.fh_linear = nn.Linear(hidden_dim,hidden_dim)
            self.ox_linear = nn.Linear(in_dim,hidden_dim)
            self.oh_linear = nn.Linear(hidden_dim,hidden_dim)
            self.cx_linear = nn.Linear(in_dim,hidden_dim)
            self.ch_linear = nn.Linear(hidden_dim,hidden_dim)

        def forward(self,x,h_1,c_1):
            i = torch.sigmoid(self.ix_linear(x)+self.ih_linear(h_1))
            f = torch.sigmoid(self.fx_linear(x)+self.fh_linear(h_1))
            o = torch.sigmoid(self.ox_linear(x)+self.oh_linear(h_1))
            c_h = torch.tanh(self.cx_linear(x)+self.ch_linear(h_1))
            c = f*c_1+i*c_h
            h = o*torch.tanh(c)
            return h,c

然后构造lstm网络：

    class lstm(nn.Module):
        def __init__(self,in_dim,hidden_dim):
            super(lstm,self).__init__()
            self.hidden_dim = hidden_dim
            self.lstm_cell = lstm_cell(in_dim,hidden_dim)
        def forward(self,x):
            outs=[]
            h,c=None,None
            for seq_x in x:
                if h is None:h=torch.randn(x.shape[1],self.hidden_dim)
                if c is None:c=torch.randn(x.shape[1],self.hidden_dim)
                h,c = self.lstm_cell(seq_x,h,c)
                outs.append(torch.unsqueeze(h,0))
            outs = torch.cat(outs)
            return outs,h

构建训练数据，查看正向推理将结果：

    batch_size = 24
    seq_lens = 7
    in_dim = 12
    out_dim = 6
    rnn = lstm(in_dim,out_dim)
    x = torch.randn(seq_lens,batch_size,in_dim)
    rnn(x)

输出的结果如下：

![](./14.png)

如此，实验结束

## GRU(门控迭代网络)


门控循环单元（Gated Recurrent Unit，GRU）是一种常用的循环神经网络（RNN）变体，用于处理序列数据和时间序列数据。GRU通过引入门控机制来解决传统RNN中的梯度消失和梯度爆炸等问题，从而更好地捕捉长期依赖关系。

门控制主要包括更新门和重置门，如下图所示，通过计算两个可以训练的参数来控制模型的记忆和遗忘，当我们掌握了lstm之后，gru就很好理解。

![](./15.png)

候选隐状态：候选隐状态是一个用于更新当前时间步的隐状态的临时状态，计算规则如下：

![](./16.png)

隐状态：隐状态是指在模型中不直接观测到的状态，通常用于表示模型内部的信息和学习到的特征，其计算过程如下所示：

![](./17.png)

总体来说，GRU就是上个时间步的信息\*一个权重+当前时间步*(1-权重)，这可以看成是lstm的一个简化版本。

### 代码实现

想对GRU进行代码实现，首先要看GRU的结构和公式，如下图所示：

![](./18.png)

首先导包，确认gpu情况

    import torch
    import torch.nn as nn
    if torch.cuda.is_available():
        # 获取GPU设备数量
        device_count = torch.cuda.device_count()
        print(f"发现 {device_count} 个可用的GPU 设备")
        # 获取每个GPU的名称
        for i in range(device_count):
            print(f"GPU 设备 {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("没有发现可用的GPU")

定义gru模块

    class gru_cell(nn.Module):
        def __init__(self,in_dim,hidden_dim):
            super(gru_cell,self).__init__()
            self.rx_linear = nn.Linear(in_dim,hidden_dim)
            self.rh_linear = nn.Linear(hidden_dim,hidden_dim)
            self.zx_linear = nn.Linear(in_dim,hidden_dim)
            self.zh_linear = nn.Linear(hidden_dim,hidden_dim)
            self.hx_linear = nn.Linear(in_dim,hidden_dim)
            self.hh_linear = nn.Linear(hidden_dim,hidden_dim)
        def forward(self,x,h_1):
            z = torch.sigmoid(self.zx_linear(x)+self.zh_linear(h_1))
            r = torch.sigmoid(self.rx_linear(x)+self.rh_linear(h_1))
            h_h = torch.tanh(self.hx_linear(x)+self.hh_linear(r*h_1))
            h = z*h_1 + (1-z)*h_h
            return h

定义gru网络

    class gru(nn.Module):
        def __init__(self,in_dim,hidden_dim):
            super(gru,self).__init__()
            self.hidden_dim = hidden_dim
            self.gru_cell = gru_cell(in_dim,hidden_dim)
        def forward(self,x):
            outs=[]
            h = None
            for seq_x in x:
                if h is None:h=torch.randn(x.shape[1],self.hidden_dim)
                h = self.gru_cell(seq_x,h)
                outs.append(torch.unsqueeze(h,0))
            outs = torch.cat(outs)
            return outs,h

测试网络

    batch_size = 24
    seq_lens = 7
    in_dim = 12
    out_dim = 6
    rnn = gru(in_dim,out_dim)
    x = torch.randn(seq_lens,batch_size,in_dim)
    rnn(x)

![](./19.png)

至此，rnn的三种基本网络的原理和实验我就做完了。

## 总结

在本次实验中，我手动实现了rnn的正向传播，理解了普通rnn的推理过程，同时学习了lstm和gru的原理和代码实现，为后续深入学习rnn打下了基础。