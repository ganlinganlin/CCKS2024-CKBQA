
# 这里导入了一些用于处理日志、数学计算、文件操作、图形绘制等的模块。具体的导入内容如下：
# - `os`: 提供了与操作系统交互的功能，用于文件路径等操作。
# - `math`: 提供了一些数学函数的功能。
# - `json`: 用于处理 JSON 格式的数据。
# - `matplotlib.pyplot`: 用于绘制图表。
# - `List` 和 `Optional`: 用于声明变量的类型注解。
# - `transformers.trainer.TRAINER_STATE_NAME`: 从 transformers 库的 `trainer` 模块导入了 `TRAINER_STATE_NAME` 常量。
# 此外，还导入了一个自定义模块 `llmtuner.extras.logging` 中的 `get_logger` 函数。
import os
import math
import json
import matplotlib.pyplot as plt
from typing import List, Optional
from transformers.trainer import TRAINER_STATE_NAME

from llmtuner.extras.logging import get_logger


logger = get_logger(__name__)

# 这是一个用于对一组浮点数进行指数移动平均 (Exponential Moving Average, EMA) 的函数。具体来说，它实现了类似 TensorBoard 中的指数移动平均。
# 函数的输入是一个包含浮点数的列表 `scalars`，输出是对这组浮点数进行指数移动平均后的新列表 `smoothed`。算法使用了一个类似 sigmoid 函数的权重计算。
#     对于输入列表中的每个值，都会计算一个平滑后的值，并将其添加到输出列表中。
def smooth(scalars: List[float]) -> List[float]:
    r"""
    EMA implementation according to TensorBoard.
    """
    last = scalars[0]
    smoothed = list()
    weight = 1.8 * (1 / (1 + math.exp(-0.05 * len(scalars))) - 0.5) # a sigmoid function
    for next_val in scalars:
        smoothed_val = last * weight + (1 - weight) * next_val
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed

# 这是一个名为 `plot_loss` 的函数，用于从训练过程的日志中绘制损失曲线。以下是函数的关键点解释：
# 1. **参数:**
#    - `save_dictionary`: 表示保存图形的目录路径。
#    - `keys`: 可选参数，表示要绘制的损失指标的列表，默认为 ["loss"]。
# 2. **读取训练状态数据:**
#    - 打开保存在 `save_dictionary` 中的训练状态文件（可能是 `TRAINER_STATE_NAME`）。
#    - 使用 `json.load` 读取训练状态数据。
# 3. **遍历指标:**
#    - 对于指定的每个损失指标 `key`，提取相应的步数 (`steps`) 和指标值 (`metrics`)。
# 4. **绘制图形:**
#    - 如果找不到任何指标值，记录一个警告并跳过。
#    - 使用 `plt.plot` 绘制原始指标曲线和平滑后的曲线。
#    - 设置图形标题、轴标签、图例等。
#    - 将图形保存为 PNG 格式。
# 这个函数的主要目的是从训练状态文件中提取损失指标的历史记录，并绘制损失曲线。
def plot_loss(save_dictionary: os.PathLike, keys: Optional[List[str]] = ["loss"]) -> None:

    with open(os.path.join(save_dictionary, TRAINER_STATE_NAME), "r", encoding="utf-8") as f:
        data = json.load(f)

    for key in keys:
        steps, metrics = [], []
        for i in range(len(data["log_history"])):
            if key in data["log_history"][i]:
                steps.append(data["log_history"][i]["step"])
                metrics.append(data["log_history"][i][key])

        if len(metrics) == 0:
            logger.warning(f"No metric {key} to plot.")
            continue

        plt.figure()
        plt.plot(steps, metrics, alpha=0.4, label="original")
        plt.plot(steps, smooth(metrics), label="smoothed")
        plt.title("training {} of {}".format(key, save_dictionary))
        plt.xlabel("step")
        plt.ylabel(key)
        plt.legend()
        plt.savefig(os.path.join(save_dictionary, "training_{}.png".format(key)), format="png", dpi=100)
        print("Figure saved:", os.path.join(save_dictionary, "training_{}.png".format(key)))
