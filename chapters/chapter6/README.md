# 6. 如何开展和记录实验

> 导读: 当我们开始训练多个具有不同超参数的模型，我们就需要对实验开始进行管理。**我们将其分为三个部分：实验追踪、超参数搜索和配置设置**。我们将使用 Weights & Biases 来演示实验记录和追踪；然后，我们将利用 Weights & Biases Sweeps 对训练超参数进行超参数搜索；最后，我们将使用 Hydra 来优雅地配置我们日益复杂的深度学习应用。
>
> 本次课程目的在于能够让你了解并实践如何将实验管理工具整合到你的模型训练工作流程中。
## 本教程目标
1. 通过Weights & Biases管理实验记录
2. 使用 Sweeps 执行超参数搜索。
3. 使用 Hydra 管理复杂的配置。
## 本教程内容
### 0. 安装

```python
conda create --name l8 python=3.9
conda install -n l8 ipykernel --update-deps --force-reinstall
conda install -n l8 pytorch torchvision torchaudio -c pytorch-nightly
conda install -n l8 -c conda-forge wandb
conda install -c conda-forge hydra-core
```

或者用`pip install` 库

### 1. 实验记录

大家是不是都曾遇到过这样的情况：如果没有良好的实验记录工具，我们最终也许会得到一个性能非常好的模型，但我们不记得其超参数选择，或者启动 100 个实验却无法轻松跟踪哪个模型表现最好，而实验跟踪工具能帮助我们解决这些问题。

**Logging**

通常来说我们在训练的过程中，通常会打印我们正在使用的超参数，以及模型训练时的损失+准确性。

```python
import random

def run_training_run_txt_log(epochs, lr):
    print(f"Training for {epochs} epochs with learning rate {lr}")
    offset = random.random() / 5
   
    for epoch in range(2, epochs):
        # 模拟训练过程
        acc = 1 - 2 ** -epoch - random.random() / epoch - offset
        loss = 2 ** -epoch + random.random() / epoch + offset
        print(f"epoch={epoch}, acc={acc}, loss={loss}")

# 进行一次学习率为0.1的训练运行
run_training_run_txt_log(epochs=10, lr=0.01)
```

下面展示我们如何用Weights & Biases管理实验记录

#### Weights and Biases

Weights & Biases 是：

> “开发者构建更好模型、更快开发的机器学习平台。使用 W&B 的轻量级、可互操作的工具，可以快速追踪实验、版本化和迭代数据集、评估模型性能、复现模型、可视化结果并发现回归问题，并与同事分享发现。
>
> 你使用哪种实验追踪工具不是一个标准答案：有人喜欢 Weights and Biases，简称 wandb：你可以按其最初的意图读作 w-and-b，或者读作 wan-db（因为它像数据库一样保存东西）。替代选择包括 Tensorboard、Neptune 和 Tensorboard。”

让我们开始使用wandb吧！

系统可能会提示您创建账户，然后添加您的token。

```python
# Log in to your W&B account
import wandb
wandb.login()
```

我们现在将在上面提供的函数进行修改，展示如何使用wandb。

```python
import random

def run_training_run(epochs, lr):
      print(f"Training for {epochs} epochs with learning rate {lr}")

      wandb.init(
            # Set the project where this run will be logged
            project="example", 
            # Track hyperparameters and run metadata
            config={
            "learning_rate": lr,
            "epochs": epochs,
            })
      
      offset = random.random() / 5
      print(f"lr: {lr}")
      for epoch in range(2, epochs):
            # simulating a training run
            acc = 1 - 2 ** -epoch - random.random() / epoch - offset
            loss = 2 ** -epoch + random.random() / epoch + offset
            print(f"epoch={epoch}, acc={acc}, loss={loss}")
            wandb.log({"acc": acc, "loss": loss})

      wandb.finish()

run_training_run(epochs=10, lr=0.01)
```

我们在这里使用 3 个函数：wandb.init、wandb.log 和 wandb.finish——它们各自的作用是什么？

- 我们在脚本开头调用一次 wandb.init() 来初始化新项目。这会在 W&B 中创建新的运行并启动后台进程来同步数据。
- 我们调用 wandb.log(dict) 将指标、媒体或自定义对象的字典记录到步骤中。我们可以看到我们的模型和数据如何随着时间的推移而演变。
- 我们调用wandb.finish来使运行完成，并完成所有数据的上传。

让我们看看在 wandb 网站上看到了什么，应该看到我们的准确性和损失曲线。

![](img/1-1.png)

在我们的信息选项卡中，我们还应该能够看到配置和摘要，告诉我们 acc 和 loss 的最终值。

![](img/1-2.png)

我们已经获得了两个不错的功能：

1. 能够看到循环每一步的准确性和损失如何变化。
2. 能够看到与运行相关的配置（超参数）。
3. 能够看到我们的运行最终获得的准确率acc和loss损失。

#### 多次实验

我们现在要增加一些复杂性。当我们通常训练模型时，我们会尝试不同的超参数。我们将调整的最重要的超参数之一是学习率，另一个可能是训练的轮数（epochs）。那么我们如何记录多个运行呢？

```python
def run_multiple_training_runs(epochs, lrs):
    for epoch in epochs:
        for lr in lrs:
            run_training_run(epoch, lr)

# Try different values for the learning rate
epochs = [100, 120, 140]
lrs = [0.1, 0.01, 0.001, 0.0001]
run_multiple_training_runs(epochs, lrs)
```

正如你所看到的，这使用了我们上面已经写好的函数，以不同的学习率和epoch多次调用它。让我们看看我们得到了什么。

我们可以访问 wandb 的网站并进入表格选项卡。

![](img/1-3.png)

在左边我们可以看到每一次的实验，点击进去，可以看到每一组的实验参数配置，比如我们选择第一组，happy-water-13，可以看到下面这个界面：

![](img/1-4.png)

#### 模型保存和加载

我们经过不懈的努力终于获得了我们期望的结果！现在我们决定要使用其中一个训练好的模型。我们可以在 W&B 上查找该运行的配置，然后重新训练模型并保存它！但是，如果我们在运行时就保存了与其关联的模型，那该多好，这样我们就可以直接加载它，对吗？

那么，我们应该如何实现这一点呢？我们可以使用 Weights & Biases 的 Artifacts 功能来跟踪数据集、模型、依赖项和结果，贯穿于整个机器学习流程的每一步。Artifacts 可以轻松获得文件更改的完整且可审核的历史记录。根据[文档](https://docs.wandb.ai/guides/artifacts)：

Artifacts 可以被视为一个版本化的目录。Artifacts 要么是运行的输入，要么是运行的输出。常见的 artifacts 包括整个训练集和模型。可以将数据集直接存储到 artifacts 中，或者使用 artifact 引用指向其他系统中的数据，如 Amazon、或你自己的系统。

使用 4 行简单的代码来记录 wandb Artifacts非常容易：

```python
wandb.init()
artifact = wandb.Artifact(<enter_filename>, type='model')
artifact.add_file(<file_path>)
wandb.run.log_artifact(artifact)
```

如果我们有一行用于保存 PyTorch 模型的代码：

```python
model_path = f"model_{epoch}.pt"
torch.save(model.state_dict(), model_path)
```

我们可以修改它以将artifacts上传到 wandb 上。

```python
# Log the model to wandb
model_path = f"model_{epoch}.pt"
torch.save(model.state_dict(), model_path)
artifact = wandb.Artifact(model_path, type='model')
artifact.add_file(model_path)
wandb.run.log_artifact(artifact)
```

现在我们可以看到我们的模型checkpoint保存在 W&B 中：

![](img/1-5.png)

我们还可以看到相关的元数据：

![](img/1-6.png)

**练习**

> 撰写代码，以便您可以在训练时保存前 3 个最佳模型。提示：参见[这里](How to save all your trained model weights locally after every epoch.ipynb)。

如果我们有保存的模型，我们现在可以加载该模型。假设我们原来的加载过程是从本地保存的checkpoint加载：

```python
model.load_state_dict(torch.load("model_9.pt"))
```

我们现在可以使用

```python
run = wandb.init()
artifact = run.use_artifact('YOUR_PATH/model_9.pt:v1', type='model')
artifact_dir = artifact.download()
model.load_state_dict(torch.load(artifact_dir + "/model_9.pt"))
```

### 2. 超参数搜索

当我们有多种超参数选择时，我们想要都尝试一下，这意味着使用不同的超参数值运行模型。

#### 搜索选项

我们可以决定如何采样超参数的值，包括贝叶斯优化、网格搜索和随机搜索。 在网格搜索中，我们为每个超参数定义一组可能的值，然后搜索会为每个可能的超参数值组合训练一个模型。 例如：使用 epochs = [100, 120, 140] 和 lrs = [0.1, 0.01, 0.001, 0.0001]，我们的网格将是 list(itertools.product(epochs, lrs))，即 [(100, 0.1), (100, 0.01), (100, 0.001), (100, 0.0001), (120, 0.1), (120, 0.01), (120, 0.001), (120, 0.0001), (140, 0.1), (140, 0.01), (140, 0.001), (140, 0.0001)]。 在随机搜索中，我们为每个超参数提供一个统计分布，从中采样值。在这里，我们通常会控制或限制使用的超参数组合数量。 在贝叶斯优化中，使用先前迭代的结果来决定下一组超参数值，采用一种序列模型优化（SMBO）算法。

这种方法在参数数量增加时不太可扩展。

#### Weights & Biases 超参数优化 (Sweeps)

正如文档所述： “Weights & Biases 超参数优化有两个组件：控制器和一个或多个代理。控制器选择新的超参数组合。通常，控制器由 Weights & Biases 服务器管理。代理向 Weights & Biases 服务器查询超参数，并使用这些超参数进行模型训练。然后将训练结果报告给控制器。代理可以在一台或多台机器上运行一个或多个进程。”

一旦我们有了 Weights & Biases 的训练代码，添加超参数优化只需三步：

1. 定义超参数优化配置
2. 初始化超参数优化（控制器）
3. 启动超参数优化代理

让我们看看它的实际操作。假设我们有以下代码：

```python
import wandb
def my_train_func():
    # read the current value of parameter "a" from wandb.config
    wandb.init()
    a = wandb.config.a

    wandb.log({"a": a, "accuracy": a + 1})
```

```python
sweep_configuration = {
    "name": "my-awesome-sweep",
    "metric": {"name": "accuracy", "goal": "maximize"},
    "method": "grid",
    "parameters": {
        "a": {
            "values": [1, 2, 3, 4]
        }
    }
}
```

注意事项：

- 使用的是网格搜索
- 我们指定了要优化的指标——这仅被某些搜索策略和停止标准使用。请注意，我们必须在 Python 脚本中将变量 accuracy（在此示例中）记录到 W&B 中，这一点我们已经完成。
- 我们为 “a” 指定了值。

步骤 2：初始化超参数优化

在这一步中，我们启动上述的超参数优化控制器：

```python
sweep_id = wandb.sweep(sweep=sweep_configuration, project='my-first-sweep')
```

步骤 3：启动超参数优化代理

最后，我们启动代理，提供超参数优化 ID、要调用的函数以及（可选）要运行的次数（count）。

```python
wandb.agent(sweep_id, function=my_train_func, count=4)
```

把[以上步骤](https://docs.wandb.ai/ref/python/agent)放到一起

```python
import wandb
sweep_configuration = {
    "name": "my-awesome-sweep",
    "metric": {"name": "accuracy", "goal": "maximize"},
    "method": "grid",
    "parameters": {
        "a": {
            "values": [1, 2, 3, 4]
        }
    }
}

def my_train_func():
    # read the current value of parameter "a" from wandb.config
    wandb.init()
    a = wandb.config.a

    wandb.log({"a": a, "accuracy": a + 1})

sweep_id = wandb.sweep(sweep_configuration)

# run the sweep
wandb.agent(sweep_id, function=my_train_func)
```

#### 课堂练习 

> 修改以下代码，以便你可以在其上运行超参数优化（sweep）。选择 val_loss 作为你要优化的指标。为 batch_size、epochs 和 learning rate 选择合理的选项。 现在，对于 learning rate，使用一个分布，该分布在 exp(min) 和 exp(max) 之间进行采样，使得自然对数在 min 和 max 之间均匀分布。

将你的解决方案与[此处](https://docs.wandb.ai/guides/sweeps/add-w-and-b-to-your-code)的解决方案进行比较。

```python
import numpy as np 
import random

def train_one_epoch(epoch, lr, bs): 
  acc = 0.25 + ((epoch/30) +  (random.random()/10))
  loss = 0.2 + (1 - ((epoch-1)/10 +  random.random()/5))
  return acc, loss

def evaluate_one_epoch(epoch): 
  acc = 0.1 + ((epoch/20) +  (random.random()/10))
  loss = 0.25 + (1 - ((epoch-1)/10 +  random.random()/6))
  return acc, loss

def main():
    run = wandb.init(project='my-first-sweep')

    # this is key: we define values from `wandb.config` instead of 
    # defining hard values
    lr  =  wandb.config.lr
    bs = wandb.config.batch_size
    epochs = wandb.config.epochs

    for epoch in np.arange(1, epochs):
      train_acc, train_loss = train_one_epoch(epoch, lr, bs)
      val_acc, val_loss = evaluate_one_epoch(epoch)

      wandb.log({
        'epoch': epoch, 
        'train_acc': train_acc,
        'train_loss': train_loss, 
        'val_acc': val_acc, 
        'val_loss': val_loss
      })
```

### 3. 使用Hydra进行配置

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

