# 基于专利的多标签分类



## 背景

最近在做专利的多标签分类，一份专利可能被分类到多个标签上。

我准备了数据集保存data文件夹下，其中只有一部分的数据。

## Quick Start

### 模型训练

在`train.py`中运行下述代码：

```python
patent_trainer = PatentTrainer()
patent_trainer.train()
```

linux 用户可以很方便地使用命令行传递参数。为了使用linux和window都使用同样的参数可以运行代码，我们通过json文件传递模型运行的参数。

模型训练的参数在 `train_params.json`文件中。

通过下述命令就可以启动模型训练：

```shell
python train.py --config train_params.json
```

在output文件夹中，best_model是训练好的最佳模型。

### 模型预测

```python
patent_trainer = PatentTrainer()
patent_trainer.predict()
```

模型预测的参数在`predict_params.json`文件中。

```shell
python train.py --config predict_params.json
```

模型预测完成的结果会保存在output数据集下。

