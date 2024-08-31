# Change Detection Laboratory

使用 [PyTorch](https://pytorch.org/) 开发的基于深度学习遥感影像变化检测项目，可作为算法开发、训练框架，也可作为基线测试平台。

*CDLab也拥有 [PaddlePaddle 版本](https://github.com/Bobholamovic/CDLab-PP)。*

## 依赖库

> opencv-python==4.1.1  
  pytorch==1.6.0  
  torchvision==0.7.0  
  pyyaml==5.1.2  
  scikit-image==0.15.0  
  scikit-learn==0.21.3  
  scipy==1.3.1  
  tqdm==4.35.0

在 Python 3.7.4，Ubuntu 16.04 环境下测试通过。

## 快速上手

在 `src/constants.py` 文件中将相应常量修改为数据集存放的位置。

### 数据预处理

针对数据集的预处理脚本存放在 `scripts/` 目录下。

### 模型训练

运行如下指令以从头训练一个模型：

```bash
python train.py train --exp_config PATH_TO_CONFIG_FILE
```

在 `configs/` 目录中已经包含了一部分现成的配置文件，可供直接使用。*注意，由于时间和精力的限制，配置文件中提供的超参数可能并没有经过充分调优，您可以通过修改超参数进一步提升模型的效果。*

训练脚本开始执行后，首先会输出配置信息到屏幕，然后会有出现一个提示符，指示您输入一些笔记。这些笔记将被记录到日志文件中。如果在一段时间后，您忘记了本次实验的具体内容，这些笔记可能有助于您回想起来。当然，您也可以选择按下回车键直接跳过。

如果需要从一个检查点（checkpoint）开始继续训练，运行如下指令：

```bash
python train.py train --exp_config PATH_TO_CONFIG_FILE --resume PATH_TO_CHECKPOINT
```

以下是对其它一些常用命令行选项的介绍：

- `anew`: 如果您希望指定的检查点只是用于初始化模型参数，指定此选项。请注意，从一个不兼容的模型中获取部分层的参数对待训练的模型进行初始化也是允许的。
- `save_on`: 如果需要在进行模型评估的同时储存模型的输出结果，指定此选项。项目默认采用基于 epoch 的训练器。在每个 epoch 末尾，训练器也将在验证集上评估模型的性能。
- `log_off`: 指定此选项以禁用日志文件。
- `tb_on`: 指定此选项以启用 TensorBoard 日志。
- `debug_on`: 指定此选项以在程序崩溃处自动设置断点，便于进行事后调试。

在训练过程中或训练完成后，您可以在 `exp/DATASET_NAME/weights/` 目录下查看模型权重文件，在 `exp/DATASET_NAME/logs/` 目录下查看日志文件，在 `exp/DATASET_NAME/out/` 目录下查看输出的变化图。

### 模型评估

使用如下指令在测试集上评估已训练好的模型：

```bash
python train.py eval --exp_config PATH_TO_CONFIG_FILE --resume PATH_TO_CHECKPOINT --save_on --subset test
```

本项目也提供在大幅栅格影像上进行滑窗测试的功能。使用如下指令：

```bash
python sw_test.py --exp_config PATH_TO_CONFIG_FILE \
  --resume PATH_TO_CHECKPOINT --ckp_path PATH_TO_CHECKPOINT \
  --t1_dir PATH_TO_T1_DIR --t2_dir PATH_TO_T2_DIR --gt_dir PATH_TO_GT_DIR
```

对于 `src/sw_test.py` 文件，其它一些常用的可选命令行参数包括：
- `--window_size`: 设置滑窗大小。
- `--stride`: 设置滑窗滑动步长。
- `--glob`: 指定在 `t1_dir`、`t2_dir`、和 `gt_dir` 中匹配文件名的 pattern （通配符）。
- `--threshold`: 指定将模型输出的变化概率二值化时使用的阈值。

不过，请注意当前 `src/sw_test.py` 功能有限，并不支持一些较为复杂的自定义预处理和后处理模块。

## 使用第三方库中的模型

目前本项目支持对 [change_detection.pytorch](https://github.com/likyoo/change_detection.pytorch) 库中模型的训练和评估。您只需通过修改配置文件即可使用 cdp 库中的模型。请参考位于 `configs/svcd/config_svcd_cdp_unet.yaml` 的示例配置文件。

当前支持的 cdp 版本号为 0.1.0。

## 预置模型列表

| 模型名称 | 对应名称 | 链接
|:-:|:-:|:-:|
| CDNet | `CDNet` | [paper](https://doi.org/10.1007/s10514-018-9734-5) |
| FC-EF | `UNet` | [paper](https://ieeexplore.ieee.org/abstract/document/8451652) |
| FC-Siam-conc | `SiamUNet-conc` | [paper](https://ieeexplore.ieee.org/abstract/document/8451652) |
| FC-Siam-diff | `SiamUNet-diff` | [paper](https://ieeexplore.ieee.org/abstract/document/8451652) |
| STANet | `STANet` | [paper](https://www.mdpi.com/2072-4292/12/10/1662) |
| DSIFN | `IFN` | [paper](https://www.sciencedirect.com/science/article/pii/S0924271620301532) |
| SNUNet | `SNUNet` | [paper](https://ieeexplore.ieee.org/document/9355573) |
| BIT | `BIT` | [paper](https://ieeexplore.ieee.org/document/9491802) |
| L-UNet | `LUNet` | [paper](https://ieeexplore.ieee.org/document/9352207) |
| DSAMNet | `DSAMNet` | [paper](https://ieeexplore.ieee.org/document/9467555) |
| P2V-CD | `P2V` | [paper](https://ieeexplore.ieee.org/document/9975266) |

## 预置数据集列表

| 数据集名称 | 对应名称 | 链接 |
|:-:|:-:|:-:|
| Synthetic images and real season-varying remote sensing images | `SVCD` | [google drive](https://drive.google.com/file/d/1GX656JqqOyBi_Ef0w65kDGVto-nHrNs9) |
| WHU building change detection dataset | `WHU` | [website](http://study.rsgis.whu.edu.cn/pages/download/building_dataset.html) |
| LEVIR building change detection dataset | `LEVIRCD` | [website](https://justchenhao.github.io/LEVIR/) |

## SVCD 数据集上测试结果

| 模型 | Precision | Recall | F1 | OA |
|:-:|:-:|:-:|:-:|:-:|
| CDNet | 92.99 | 87.08 | 89.94 | 97.59 |
| FC-EF | 94.28 | 83.80 | 88.73 | 97.37 |
| FC-Siam-conc | 94.57 | 91.34 | 92.93 | 98.28 |
| FC-Siam-diff | 95.87 | 90.60 | 93.16 | 98.36 |
| STANet | 89.22 | 98.25 | 93.52 | 98.32 |
| DSIFN | 97.64 | 96.35 | 96.99 | 99.26 |
| SNUNet | 97.89 | 97.25 | 97.57 | 99.40 |
| BIT | 97.20 | 96.38 | 96.79 | 99.21 |
| L-UNet | 96.48 | 94.79 | 95.63 | 98.93 |
| DSAMNet | 92.78 | 98.06 | 95.35 | 98.82 |
| P2V-CD | 98.57 | 98.26 | 98.42 | |
