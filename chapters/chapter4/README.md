# PyTorch 基础知识

> 导读: 什么是PyTorch？我们如何用PyTorch快速实现一个完整的神经网络训练流程？首先阅读链接的官方 Pytorch 教程。然后，将完成有关张量 (Tensor)、Autograd、神经网络和分类器训练/评估的练习。有些问题会要求您实现几行代码，而其他问题会要求您猜测操作的输出是什么，或者识别代码的问题。强烈建议您在参考解决方案中查找问题之前先亲自尝试这些问题。
## 本教程目标：
1. 在 PyTorch 中执行张量运算。
2. 了解 Autograd 背景下神经网络的后向和前向传递。
3. 检测 PyTorch 训练代码中的常见问题。
## 本教程内容：
### 0. 快速开始

方法一：colab已经自动帮我们安装好了torch库，我们可以直接使用。建议可以直接先使用在线的编译器，先快速理解知识点。

![](img/0-1.png)

方法二：[在vscode/pycharm上通过pip install 来安装torch,torchvision](https://www.bilibili.com/video/BV1hE411t7RN/?t=734&vd_source=6d9a3bf0aa736e90be2bf85ca031f921)

### 1. Tensor

我们将从最基本的张量开始。首先，浏览官方张量教程[这里](https://pytorch.org/tutorials/beginner/blitz/tensor_tutorial.html)。

> 张量是一种特殊的数据结构，与数组和矩阵非常相似。在PyTorch中，我们使用张量对模型的输入和输出以及模型的参数进行编码。张量与NumPy的 ndarray 类似，不同之处在于张量可以在GPU或其他专用硬件上运行以加速计算。如果您熟悉 ndarrays，那么您就会熟悉Tensor API。如果没有，请按照此下面的问题进行操作。最好可以不看答案操作一遍，先思考一下，再去搜索一下，最后比对一下正确的操作。这样子效果是最好的。

1. 将二维列表 `[[5,3], [0,9]]` 转换为一个张量
2. 使用区间 `[0, 1)` 上均匀分布的随机数创建形状 `(5, 4)` 的张量`t`
3. 找出张量`t`所在的设备及其数据类型。
4. 创建形状 `(4,4)` 和 `(4,4)` 的两个随机张量，分别称为`u`和`v`。将它们连接起来形成形状为 `(8, 4)` 的张量。
5. 连接 `u` 和 `v` 以创建形状 `(2, 4, 4)` 的张量。
6. 连接 `u` 和 `v` 形成一个张量，称为形状 `(4, 4, 2)` 的 `w`。
7. 索引 `w` 位于 `3, 3, 0`。将该元素称为`e`。
8. 会在 `u` 或 `v` 的哪一个中找到 `w`？并核实。
9. 创建一个形状为 `(4, 3)` 的全为 `1` 的张量 `a`。对`a`进行元素级别的自乘操作。
10. 向`a`添加一个额外的维度（新的第 `0` 维度）。
11. 执行 `a` 与转置矩阵的乘法。
12. `a.mul(a)` 会产生什么结果？

### 2. Autograd 和神经网络

接下来，我们学习[自动梯度](https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html)教程和[神经网络](https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html)教程。

神经网络(NN) 是对某些输入数据执行的嵌套函数的集合。这些函数由参数（由权重和偏差组成）定义，这些参数在PyTorch中存储在张量中。可以使用 torch.nn 包构建神经网络。

训练神经网络分两步进行：

- 前向传播：在前向传播中，神经网络对正确的输出做出最佳猜测。它通过每个函数运行输入数据来进行猜测。
- 反向传播：在反向传播中，神经网络根据其猜测的误差按比例调整其参数。它通过从输出向后遍历、收集误差相对于函数参数（梯度）的导数并使用梯度下降来优化参数来实现这一点。

更一般地，神经网络的典型训练过程如下：

- 定义具有一些可学习参数（或权重）的神经网络
- 迭代输入数据集
- 通过网络处理输入
- 计算损失（输出距离正确还有多远）
- 将梯度传播回网络参数
- 更新网络的权重，通常使用简单的更新规则：权重=权重-学习率梯度

有了这些教程，我们就可以尝试以下练习了！假设我们有以下起始代码，将下面这段代码复制到你的编辑器中：

```python
import torch
from torchvision.models import resnet18, ResNet18_Weights
model = resnet18(weights=ResNet18_Weights.DEFAULT)
data = torch.rand(1, 3, 64, 64)
labels = torch.rand(1, 1000)
```

13. 使用数据对模型进行前向传递并将其保存为 `preds`。
14. `preds` 的形状应该是什么？验证你的猜测。
15. 将 `resnet18` 的 `conv1` 属性的权重参数保存为 `w`。打印 `w` 因为我们稍后需要它（请注意，我的 `w` 不会与你的相同）。
16. `w` 的 `grad` 属性应该是什么？请验证。
17. 创建一个[交叉熵](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)损失对象，并用它来使用 `labels` 和 `preds` 计算损失，保存为 `loss`。打印 `loss`，因为我们稍后需要它。
18. 打印最后一次产生 `loss` 损失的数学运算。
19. 执行反向传播。
20. `w` 应该改变吗？检查 #3 的输出。
21. `w` 的 `grad` 属性会与 #4 不同吗？并验证。
22. `loss` 的 `grad` 属性应该返回什么？验证一下。
23. `loss` 的 `requires_grad` 属性应该是什么？验证一下。
24. `labels` 的 `requires_grad` 属性应该是什么？验证一下。
25. 如果你再次执行反向传播会发生什么？
26. 创建一个学习率 (`lr=1e-2`) 和动量 (`momentum=0.9`) 的 [SGD](https://pytorch.org/docs/stable/generated/torch.optim.SGD.html#torch.optim.SGD) 优化器对象，并执行一步。
27. `w` 是否应该改变？检查第3题的输出。
28. `loss` 是否应该改变？检查第5题的输出。
29. 将所有可训练参数的梯度清零。
30. `w` 的 `grad` 属性应该是什么？验证一下。
31. 在不运行的情况下，判断以下代码是否会成功执行。
32. 判断以下代码是否会成功执行。
33. 判断以下代码是否会成功执行。
34. 对于不能执行的代码，你如何修改其中一个 `.backward` 行使其工作？
35. 以下代码的输出是什么？
36. 以下代码的输出是什么？
37. 以下代码有什么问题？
38. 按正确的顺序排列训练循环的以下步骤（有多种正确答案，但你在教程中会看到一种典型的设置）：以下代码的输出是什么？
39. 我们将实现一个有一个隐藏层的神经网络。这个网络将接受一个32x32的灰度图像输入，展开它，通过一个有100个输出特征的仿射变换，应用 `ReLU` 非线性，然后映射到目标类别（10）。实现初始化和前向传递，完成以下代码。使用 `nn.Linear`, `F.relu`, `torch.flatten`
40. 用两行代码验证你能通过上述网络进行前向传递。
41. 在不运行代码的情况下，猜测以下语句的结果是什么？
42. 获取网络参数的名称
43. 以下语句指的是哪个网络层？它将评估什么？
44. 以下示意图包含了实现一个神经网络所需的所有信息。实现初始化和前向传递，完成以下代码。使用 `nn.Conv2d`, `nn.Linear`, `F.max_pool2d`, `F.relu`, `torch.flatten`。提示：`ReLU` 在子采样操作后和前两个全连接层之后应用。
45. 修改上述代码，使用 `nn.MaxPool2d` 代替 `F.max_pool2d`
46. 尝试通过将第一个卷积层的输出通道数从6增加到12来增加网络的宽度。你还需要改变什么？

### **3. 训练分类器**

接下来，我们进入教程的最后一部分：[Cifar10教程](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)。这个教程通过以下步骤来训练一个图像分类器：

- 使用torchvision加载和归一化 (normalize) CIFAR10训练和测试数据集 
- 定义一个卷积神经网络 
- 定义一个损失函数
- 在训练数据上训练网络
- 在测试数据上测试网络

完成上述教程后，回答以下问题：

48. 以下数据集加载代码可以运行，但代码中是否有错误？这些错误的影响是什么？如何修复这些错误？
49. 编写两行代码从数据加载器中获取随机的训练图像（假设上面的错误已经修复）。
50. 以下训练代码可以运行，但代码中是否有错误（包括计算效率低下）？这些错误的影响是什么？如何修复这些错误？

### 4. 总结

切记以上代码可能刚开始看会有些困难，但这很正常，一定要克制住自己的好奇心直接看答案，哪怕实在想不出来，也应该先把自己的思考和尝试写下来。然后再去看答案，比对自己的输出和答案有什么区别(敏感的人可能发现了，这和前向传播和计算损失很像)，确实是这样，就是需要不断训练自己的大脑。

熟悉PyTorch可能需要一些时间，这很正常！PyTorch是深度学习开发的强大工具。完成上面的练习后，可以在[这里](https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html)查看快速入门教程，该教程将涵盖更多方面，包括保存和加载模型以及数据集和数据加载器。在学习API的过程中，记住关键的使用模式可能会很有用；我喜欢的一个PyTorch备忘录可以在[这里](https://github.com/pytorch/tutorials/blob/master/beginner_source/PyTorch Cheat.md)找到。

这就是我们关于PyTorch基础知识的全部内容！恭喜你 - 现在你已经具备了开始解决利用PyTorch的更复杂深度学习代码的能力。