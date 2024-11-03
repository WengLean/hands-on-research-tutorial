# 6. 如何开展和记录实验

> 导读: 当我们开始训练多个具有不同超参数的模型，我们就需要对实验开始进行管理。***\*我们将其分为三个部分：实验追踪、和配置设置\****。我们将使用SwanLab来演示实验记录和追踪；然后，学习如何配置我们深度学习应用的参数。
>
> 本次课程目的在于能够让你了解并实践如何将实验管理工具整合到你的模型训练工作流程中。本节还是在上一个图像分类任务代码的基础上继续进行改进。
## 本教程目标
1. 通过SwanLab管理实验记录
2. 了解参数配置
## 本教程内容
### 0. 训练流程

这是第2节课的代码，如果不熟悉，再回去看视频讲解，多看几遍

```python
import torch
import torchvision
import torchvision.transforms as transforms

# 1.构建数据集
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 4

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# 2.定义神经网络
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()

# 3.定义 Loss 函数和优化器
import torch.optim as optim

criterion = nn.CrossEntropyLoss() # risk loss
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# 4.训练网络
for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        if (i > 5):
          break # 为了增加训练速度，正常需要训练所有数据
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

print('Finished Training')
```

### 1. 实验记录

大家是不是都曾遇到过这样的情况：如果没有良好的实验记录工具，我们最终也许会得到一个性能非常好的模型，但我们不记得其超参数选择，或者启动 100 个实验却无法轻松跟踪哪个模型表现最好，而实验跟踪工具能帮助我们解决这些问题。

**Logging**

通常来说我们在训练的过程中，通常会打印我们正在使用的超参数，以及模型训练时的损失+准确性。就比如上面打印的结果一般。我们能看到每一个epoch的损失是多少。下面展示我们如何用SwanLab管理实验记录

> SwanLab是一款开源、轻量级的AI实验跟踪工具，提供了一个跟踪、比较、和协作实验的平台，旨在加速AI研发团队100倍的研发效率。其提供了友好的API和漂亮的界面，结合了超参数跟踪、指标记录、在线协作、实验链接分享、实时消息通知等功能，让您可以快速跟踪ML实验、可视化过程、分享给同伴。借助SwanLab，科研人员可以沉淀自己的每一次训练经验，与合作者无缝地交流和协作，机器学习工程师可以更快地开发可用于生产的模型。

```python
import swanlab
swanlab.login()

def train(epochs, learning_rate):
  print(f"Training for {epochs} epochs with learning rate {learning_rate}")

  swanlab.init(
        # Set the project where this run will be logged
        project="example", 
        # Track hyperparameters and run metadata
        config={
        "learning_rate": learning_rate,
        "epochs": epochs,
        })
  
  
  # 构造数据集
  transform = transforms.Compose(
      [transforms.ToTensor(),
      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

  batch_size = 4

  trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                          download=True, transform=transform)
  trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
  
                                            shuffle=True, num_workers=2)
  # 定义网络
  net = Net()

  # 定义损失和优化器
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9) # 学习率作为一个可以调整的参数


  # 训练网络
  for epoch in range(epochs):  # epochs作为参数传入

      running_loss = 0.0
      for i, data in enumerate(trainloader, 0):
          # get the inputs; data is a list of [inputs, labels]
          if (i > 5):
            break
          inputs, labels = data

          # zero the parameter gradients
          optimizer.zero_grad()

          # forward + backward + optimize
          outputs = net(inputs)
          loss = criterion(outputs, labels)
          loss.backward()
          optimizer.step()
          running_loss += loss.item()
      print(f"epoch={epoch}, loss={running_loss}")
      swanlab.log({"loss": running_loss})
      running_loss = 0.0
  swanlab.finish()
train(epochs=10, learning_rate=0.01)
```

我们在这里使用 3 个函数：swanlab.init、swanlab.log 和 swanlab.finish——它们各自的作用是什么？

\- 我们在脚本开头调用一次 swanlab.init() 来初始化新项目。这会创建新的运行并启动后台进程来同步数据。

\- 我们调用 swanlab.log(dict) 将指标、媒体或自定义对象的字典记录到步骤中。我们可以看到我们的模型和数据如何随着时间的推移而演变。

\- 我们调用swanlab.finish来使运行完成，并完成所有数据的上传。

让我们看看在 swanlab 网站上看到了什么，应该看到我们的准确性和损失曲线。

<img src="img/1-1.png" style="zoom:50%;" />

在我们的信息选项卡中，我们还应该能够看到配置和摘要，告诉我们 loss 的最终值。

我们已经获得了两个不错的功能：

1. 能够看到循环每一步的损失如何变化。
2. 能够看到与运行相关的配置（超参数）。
3. 能够看到我们的运行最终获得的loss损失。

#### 参数进行配置

我们不希望用硬编码的路径名、模型名和超参数来训练深度学习模型。我们希望能够使用一个配置文件，根据使用的数据集、模型或配置进行修改。硬编码是什么？是指在编写程序时，直接将具体的值（如字符串、数字、路径等）写入源代码中，而不是通过变量、配置文件、数据库查询或其他动态方法来获取这些值。(这其实不是一个好习惯，但是经常有人这样做)

首先，让我们从一些错误的配置深度学习运行的方法开始。假设我们想从命令行控制数据集的 batch_size。可能在某台机器上工作时，你可以使用较大的 batch_size，而在另一台机器上则不行。最基本的做法是记住更改硬编码的 batch size。

```
batch_size = 128
# batch_size = 4
```

像上面那种方法并不是一个好的选择，因为每次都要去更改源码。

第二种解决方案是在运行脚本时将`batch_size`的值作为参数传递进去。这样我们就可以根据所用的显卡来改变它。我们可以通过`sys.argv`使用命令行参数来实现这一点。

使用 swanlab.config 保存你的训练配置，例如：

超参数

输入设置，例如数据集名称或模型类型

实验的任何其他变量

swanlab.config 使你可以轻松分析你的实验并在将来复现你的工作。你还可以在SwanLab应用中比较不同实验的配置，并查看不同的训练配置如何影响模型输出。

### 2.设置实验配置

config 通常在训练脚本的开头定义。当然，不同的人工智能工作流可能会有所不同，因此 config 也支持在脚本的不同位置定义，以满足灵活的需求。

以下部分概述了定义实验配置的不同场景。

#### 2.1SwanLab中设置

下面的代码片段演示了如何使用Python字典定义 config，以及如何在初始化SwanLab实验时将该字典作为参数传递：

```
import swanlab
swanlab.login()
# 定义一个config字典
config = {
  "hidden_layer_sizes": [64, 128],
  "activation": "ELU",
  "dropout": 0.5,
  "num_classes": 10,
  "optimizer": "Adam",
  "batch_normalization": True,
  "seq_length": 100,
}

# 在你初始化SwanLab时传递config字典
run = swanlab.init(project="config_example", config=config)
```

访问 config 中的值与在Python中访问其他字典的方式类似：

1. 用键名作为索引访问值

```
hidden_layer_sizes = swanlab.config["hidden_layer_sizes"]
hidden_layer_sizes
```

​	2.用 get() 方法访问值

```
activation = swanlab.config.get("activation")
activation
```

3. 用点号访问值

```
dropout = swanlab.config.dropout
dropout
```

#### 2.4 使用Hydra进行配置

我们不希望用硬编码的路径名、模型名和超参数来训练深度学习模型。我们希望能够使用一个配置文件，根据使用的数据集、模型或配置进行修改。硬编码是什么？是指在编写程序时，直接将具体的值（如字符串、数字、路径等）写入源代码中，而不是通过变量、配置文件、数据库查询或其他动态方法来获取这些值。(这其实不是一个好习惯，但是经常有人这样做)

错误的方法

首先，让我们从一些错误的配置深度学习运行的方法开始。假设我们想从命令行控制数据集的 batch_size。可能在某台机器上工作时，你可以使用较大的 batch_size，而在另一台机器上则不行。最基本的做法是记住更改硬编码的 batch size。

```python
batch_size = 128
# batch_size = 4
```

像上面那种方法并不是一个好的选择，因为每次都要去更改源码。

第二种解决方案是在运行脚本时将`batch_size`的值作为参数传递进去。这样我们就可以根据所用的显卡来改变它。我们可以通过`sys.argv`使用命令行参数来实现这一点。

**main.py**

```python
import sys
batch_size = sys.argv[1]
```

如果我们希望batch_size 设置成16，我们可以这样调用：`python main.py 16`。如果我们需要配置多个设置，直接使用`sys.argv`工作可能就不那么用户友好，这时我们可能希望使用一个解析器。其中最流行的是`argparse`模块：

```python
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('batch_size', metavar='B', type=int,
                    help='batch_size for the model')

args = parser.parse_args()
print(args.batch_size)
```

**练习**

> 让脚本接收批处理大小（batch_size）、学习率（learning_rate）和丢弃率（dropout）作为参数，并为每个参数使用合适的类型。如果未提供这些参数，除了学习率（learning_rate）外，其他都应使用默认值，学习率是必须提供的。

这样操作在当前情境下或许可行，但是一旦我们有上百个参数时，显式地为每个希望不同于默认值的参数指定值就会变得非常困难！要是能有一种方式将这些配置存储在一个文件中就好了。

#### Hydra

[Hydra](https://hydra.cc/docs/intro/)是一个开源的Python框架，它简化了研究和其他复杂应用程序的开发。Hydra这个名字来源于其能够运行多个类似任务的能力——就像一个多头的九头蛇一样。

我们将遵循Hydra的[教程](https://hydra.cc/docs/tutorials/basic/your_first_app/simple_cli/)，但会加入一些我自己的理解和调整。

```python
from omegaconf import DictConfig, OmegaConf
import hydra

@hydra.main(version_base=None)
def run(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg)) # {}

if __name__ == "__main__":
    run()
```

在这个示例中，Hydra创建了一个空的配置（cfg）对象，并将其传递给hydra.main装饰器。

> 提示： “OmegaConf是一个基于YAML的分层配置系统，支持从多种来源（文件、CLI参数、环境变量）合并配置，无论配置是如何创建的，都能提供一致的API。” “装饰器是Python中一个重要的部分。简单来说：它们是修改其他函数功能的函数。它们有助于使我们的代码更简洁、更符合Python风格。大多数初学者不知道在哪里使用它们，所以我将分享一些装饰器可以使你的代码更精简的场景。”

我们可以通过命令行使用“+”来添加新的配置值。

```python
# should return “batch_size: 16”
python run.py +batch_size=16
```

这里就是Hydra开始起作用的地方。由于在命令行中输入参数十分繁琐，我们可以开始使用配置文件。Hydra的配置文件是YAML文件，并且应该具有.yaml的文件扩展名。

我们在与run.py相同的目录下创建一个config.yaml文件，并用我们的配置信息填充它。

**config.yaml**

```yaml
batch_size: 16
```

现在，我们需要告诉Hydra在哪里找到这个配置文件。请注意，config_name应当与我们的文件名匹配，并且config_path是相对于应用程序的相对路径。

```python
@hydra.main(version_base=None, config_path=".", config_name="config")
```

我们现在可以使用命令 `python run.py` 来运行 `run.py`，并且应该能看到打印出的 batch_size。这里的一个很酷的功能是，我们可以通过命令行来覆盖配置文件中的值（这次，我们可以省略“+”，因为配置值并不是新的：

```python
python run.py batch_size=32 # should print 32
```

让我们开始让我们的配置变得更加有用：

```yaml
loss: cross_entropy
batch_size: 64
num_workers: 4
name: ??? # Missing value, must be populated prior to access

optim: # Config is hierarchical
  name: adam
  lr: 0.0001
  weight_decay: ${optim.lr} # Value interpolation
  momentum: 0.9
```

这里有一些新内容： 我们正在使用层次结构（例如 `cfg.optim.name`） 我们正在进行值的插值（例如 `cfg.optim.weight_decay`） 我们指定了一个必须填充的缺失值

让我们看看它是如何工作的：

```python
from omegaconf import DictConfig, OmegaConf
import hydra

@hydra.main(version_base=None, config_path=".", config_name="config")
def run(cfg: DictConfig):
    assert cfg.optim.name == 'adam'          # attribute style access
    assert cfg["optim"]["lr"] == 0.0001      # dictionary style access
    assert cfg.optim.weight_decay == 0.0001  # Value interpolation
    assert isinstance(cfg.optim.weight_decay, float) # Value interpolation type

    print(cfg.name)                       # raises an exception

if __name__ == "__main__":
    run()

```

我们应该会遇到 "omegaconf.errors.MissingMandatoryValue: Missing mandatory value: name" 这个错误。我们可以通过在运行程序时指定一个名称来解决这个问题。

```python
python run.py name=exp1 # Should print ‘exp1’
```

现在我们来增加一点复杂性。假设我们想要创建一个优化器类。

```python
class Optimizer:
    """Optimizer class."""
    algo: str
    lr: float

    def __init__(self, algo: str, lr: float) -> None:
        self.algo = algo
        self.lr = lr
    
    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)
```

现在我们可以使用当前的配置实例化这个优化器类了。

```python
@hydra.main(version_base=None, config_path=".", config_name="config")
def run(cfg: DictConfig):
    opt = Optimizer(cfg.optim.name, cfg.optim.lr)
    print(str(opt))
```

我们应当看到 `<class '__main__.Optimizer'>: {'algo': 'adam', 'lr': 0.0001}`。

我们能否直接通过Hydra实例化优化器呢？Hydra提供了`hydra.utils.instantiate()`（及其别名`hydra.utils.call()`）用于实例化对象和调用函数。建议在创建对象时使用`instantiate`，在调用函数时使用`call`。

我们可以使用一个简单的配置：

**config2.yml**

```yaml
optimizer:
 _target_: run.Optimizer
 algo: SGD
 lr: 0.01
```

我们可以这样进行实例化：

```python
from hydra.utils import instantiate
@hydra.main(version_base=None, config_path=".", config_name="config2")
def run(cfg: DictConfig):
    opt = instantiate(cfg.optimizer)
    print(opt)
```

来自[官方教程](https://hydra.cc/docs/advanced/instantiate_objects/overview/)的专业提示： `call/instantiate`支持以下功能： 命名参数：配置字段（除了像`_target_`这样的保留字段）作为命名参数传递给目标。配置中的命名参数可以通过在`instantiate()`调用站点传递同名的命名参数来覆盖。 位置参数：配置中可以包含一个`_args_`字段，表示要传递给目标的位置参数。位置参数可以通过在`instantiate()`调用时传递位置参数一起来覆盖。

我们甚至可以进行递归实例化。

**config3.yaml**

```yaml
trainer:
 _target_: run.Trainer
 optimizer:
   _target_: run.Optimizer
   algo: SGD
   lr: 0.01
 dataset:
   _target_: run.Dataset
   name: Imagenet
   path: /datasets/imagenet
```

以下代码可以在实例化我们的`Trainer`的同时，也实例化我们的`Dataset`和`Optimizer`。

```python
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.utils import instantiate

class Dataset:
    name: str
    path: str

    def __init__(self, name: str, path: str) -> None:
        self.name = name
        self.path = path

class Optimizer:
    """Optimizer class."""
    algo: str
    lr: float

    def __init__(self, algo: str, lr: float) -> None:
        self.algo = algo
        self.lr = lr

    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)

class Trainer:
    def __init__(self, optimizer: Optimizer, dataset: Dataset) -> None:
        self.optimizer = optimizer
        self.dataset = dataset

@hydra.main(version_base=None, config_path=".", config_name="config3")
def run(cfg: DictConfig):
    opt = instantiate(cfg.trainer)
    print(opt)

```

**练习**

> 展示你将用来实例化一个包含两个线性层的`torch.nn.Sequential`对象的`config.yaml`文件和`train.py`文件。

针对yaml文件的解决[方案](https://www.sscardapane.it/tutorials/hydra-tutorial/#variable-interpolation),也可以直接看下面：

```yaml
_target_: torch.nn.Sequential
_args_:
  - _target_: torch.nn.Linear
    in_features: 9216
    out_features: 100

  - _target_: torch.nn.Linear
    in_features: ${..[0].out_features}
    out_features: 10
```

相较于单一的配置文件，我们常常需要多个配置文件。在机器学习中，这些文件用于指定不同的数据集、模型或日志行为等我们可能想要使用的设置。因此，我们通常会使用“配置组”（Config Group），它为每种数据集、模型配置选项或日志行为等持有一个文件。

一个机器学习应用的配置组可能看起来像[这样](https://github.com/ashleve/lightning-hydra-template)：

```yaml
configs/
├── dataset
│   ├── cifar10.yaml
│   └── mnist.yaml
├── defaults.yaml
├── hydra
│   ├── defaults.yaml
│   └── with_ray.yaml
├── model
│   ├── small.yaml
│   └── large.yaml
├── normalization
│   ├── batch.yaml
│   ├── default.yaml
│   ├── group.yaml
│   ├── instance.yaml
│   └── nonorm.yaml
├── train
│   └── defaults.yaml
└── wandb
    └── defaults.yaml
```

请通读配置组文档和默认值文档，以便理解配置组和默认值的概念。

总体而言，我们会执行以下操作：

1. 创建一个目录，有时称为`confs/`或`configs/`，用于存放所有配置文件。
2. 我们可以指定要使用的配置文件。例如，如果我们想在数据集中使用`cifar10.yaml`，我们将使用命令`python run.py dataset=cifar10`。

这意味着通过命令行参数，我们可以灵活地选择不同的配置文件来适应不同的需求，比如切换数据集、模型或调整日志行为等，而无需直接修改主代码文件。配置组允许我们组织和管理这些配置文件，使其更加有序和易于维护。

**cifar10.yaml**

```yaml
---
name: cifar10
dir: cifar10/
train_batch: 32
test_batch: 10
image_dim:
    - 32
    - 32
    - 3
num_classes: 10
```

3. `defaults.yaml`文件用于指定默认使用的数据集或模型。

**defaults.yaml**

```yaml
---
defaults:
    - dataset: mnist
    - model: ${dataset}
    - train: defaults
    - wandb: defaults
    - hydra: defaults
    - normalization: default
model:
    num_groups: -1
```

**练习**

> 配置一个小型模型和一个大型模型。大型模型实例化一个torch.nn.Sequential对象，包含三个线性层；小型模型则包含两个线性层,将小型模型设为默认模型。

小贴士：Hydra与W&B的集成：今天我们已经了解了两个工具，W&B和Hydra。如何让这两个工具协同工作呢？这里有一些使用模式需要了解。

请参考这个[教程](https://wandb.ai/adrishd/hydra-example/reports/Configuring-W-B-Projects-with-Hydra--VmlldzoxNTA2MzQw)，查看如何将两者结合使用的示例代码。

## 参考资料

[Bohrium (dp.tech)](https://bohrium.dp.tech/home)

