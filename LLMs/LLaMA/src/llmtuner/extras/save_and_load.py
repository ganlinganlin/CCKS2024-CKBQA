
# 这段代码导入了一些与文件系统、PyTorch和Hugging Face Transformers库相关的模块和函数。以下是每个导入语句的解释：
# 1. **`import os`：**
#    - 导入 Python 内置的 `os` 模块，用于与操作系统进行交互，例如文件路径的操作等。
# 2. **`import torch`：**
#    - 导入 PyTorch 库，用于深度学习任务。
# 3. **`from transformers.trainer import WEIGHTS_NAME`：**
#    - 从 Hugging Face Transformers 库的 `trainer` 模块中导入 `WEIGHTS_NAME` 变量。这是一个常量，通常用于指代模型权重文件的名称。
# 4. **`from llmtuner.extras.logging import get_logger`：**
#    - 从自定义的 `llmtuner.extras.logging` 模块中导入 `get_logger` 函数。这可能是一个用于获取日志记录器的函数。
# 这些导入语句显示了一些用于文件系统操作、深度学习任务和日志记录的模块和函数。
import os
import torch
from transformers.trainer import WEIGHTS_NAME

from llmtuner.extras.logging import get_logger


logger = get_logger(__name__)


# 这是一个名为 `load_valuehead_params` 的函数，用于从预训练模型的检查点目录中加载值头（value head）的参数。以下是函数的关键点解释：
# 1. **参数:**
#    - `model`: 要加载参数的 PyTorch 模型。
#    - `checkpoint_dir`: 包含检查点文件的目录路径。
# 2. **值头文件路径:**
#    - 构建值头文件路径，使用 `os.path.join` 将目录路径和权重文件名 (`WEIGHTS_NAME`) 连接起来。
# 3. **检查文件存在性:**
#    - 如果值头文件不存在，记录一个警告并返回 `False`。
# 4. **加载值头参数:**
#    - 使用 `torch.load` 从值头文件中加载参数。
#    - 使用 `model.register_buffer` 注册模型的缓冲区，分别对应值头权重和偏置项。
#    - `persistent=False` 表示这些缓冲区不应该被保存到检查点中。
# 5. **返回结果:**
#    - 返回 `True` 表示成功加载值头参数。
# 这个函数的主要目的是从检查点目录中加载值头的权重和偏置项，并将它们注册为模型的缓冲区。
def load_valuehead_params(model: torch.nn.Module, checkpoint_dir: os.PathLike) -> bool:
    vhead_file = os.path.join(checkpoint_dir, WEIGHTS_NAME)
    if not os.path.exists(vhead_file):
        logger.warning("Provided path ({}) does not contain valuehead weights.".format(checkpoint_dir))
        return False
    vhead_params = torch.load(vhead_file, map_location="cpu")
    model.register_buffer("reward_head_weight", vhead_params["v_head.summary.weight"], persistent=False)
    model.register_buffer("reward_head_bias", vhead_params["v_head.summary.bias"], persistent=False)
    model.register_buffer("default_head_weight", torch.zeros_like(vhead_params["v_head.summary.weight"]), persistent=False)
    model.register_buffer("default_head_bias", torch.zeros_like(vhead_params["v_head.summary.bias"]), persistent=False)
    return True
