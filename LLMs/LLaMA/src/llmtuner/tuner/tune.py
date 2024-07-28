
# 代码导入了各种模块和函数，这些模块和函数是进行transformers库超参数调优所必需的。
# 使用TYPE_CHECKING条件导入了transformers模块中的TrainerCallback。
# 使用get_logger函数初始化了一个记录器（logger）。

from typing import TYPE_CHECKING, Any, Dict, List, Optional

from llmtuner.extras.callbacks import LogCallback
from llmtuner.extras.logging import get_logger
from llmtuner.tuner.core import get_train_args, load_model_and_tokenizer
from llmtuner.tuner.pt import run_pt
from llmtuner.tuner.sft import run_sft
from llmtuner.tuner.rm import run_rm
from llmtuner.tuner.ppo import run_ppo
from llmtuner.tuner.dpo import run_dpo

if TYPE_CHECKING:
    from transformers import TrainerCallback


logger = get_logger(__name__)


# 这段代码定义了一个名为 `run_exp` 的函数，用于运行一个实验。让我们逐行解释这段代码：
# 1. **函数定义：**
#    - 这是一个函数定义，名为 `run_exp`。
#    - 函数接受两个可选参数：`args` 和 `callbacks`。`args` 是一个字典，用于传递实验的参数；`callbacks` 是一个回调函数列表，用于在训练过程中执行某些操作。
# 2. **获取训练参数：**
#    - 调用 `get_train_args` 函数，该函数返回一个包含训练所需参数的元组，然后通过多个变量赋值语句将其拆分为各个参数。
# 3. **设置回调函数：**
#    - 如果没有提供回调函数列表 (`callbacks`)，则创建一个包含默认的 `LogCallback` 的列表。
# 4. **根据任务阶段运行相应的超参数调优函数：**
#    - 根据 `general_args.stage` 的值，选择性地调用不同的超参数调优函数。这里包括了 "pt"、"sft"、"rm"、"ppo" 和 "dpo" 这几个任务阶段。
#    - 如果阶段未知，则抛出 `ValueError` 异常，提示 "Unknown task"。
# 总体而言，这个函数看起来是一个用于运行基于 Hugging Face Transformers 库的超参数调优实验的主要控制流程。
def run_exp(args: Optional[Dict[str, Any]] = None, callbacks: Optional[List["TrainerCallback"]] = None):
    model_args, data_args, training_args, finetuning_args, generating_args, general_args = get_train_args(args)
    callbacks = [LogCallback()] if callbacks is None else callbacks

    if general_args.stage == "pt":
        run_pt(model_args, data_args, training_args, finetuning_args, callbacks)
    elif general_args.stage == "sft":
        run_sft(model_args, data_args, training_args, finetuning_args, generating_args, callbacks)
    elif general_args.stage == "rm":
        run_rm(model_args, data_args, training_args, finetuning_args, callbacks)
    elif general_args.stage == "ppo":
        run_ppo(model_args, data_args, training_args, finetuning_args, generating_args, callbacks)
    elif general_args.stage == "dpo":
        run_dpo(model_args, data_args, training_args, finetuning_args, callbacks)
    else:
        raise ValueError("Unknown task.")


# 这段代码定义了一个名为 `export_model` 的函数，用于导出模型。让我们逐行解释这段代码：
# 1. **函数定义：**
#    - 这是一个函数定义，名为 `export_model`。
#    - 函数接受两个可选参数：`args` 是一个字典，用于传递导出模型的参数；`max_shard_size` 是一个可选的字符串参数，用于指定最大碎片大小，默认为 "10GB"。
# 2. **获取训练参数和模型、分词器：**
#    - 调用 `get_train_args` 函数，获取训练所需的参数。
#    - 调用 `load_model_and_tokenizer` 函数，加载模型和分词器。
# 3. **恢复填充方向并保存模型和分词器：**
#    - 将分词器的填充方向设置为 "left"。
#    - 使用 `model.save_pretrained` 方法保存模型到指定的输出目录，并可选地指定 `max_shard_size`。
#    - 尝试使用 `tokenizer.save_pretrained` 方法保存分词器到输出目录，如果失败则发出警告。
# 总体而言，这个函数看起来是一个用于导出经过微调的模型和分词器的实用程序函数，同时进行了一些配置的处理，以确保导出的模型和分词器能够正确地使用。
def export_model(args: Optional[Dict[str, Any]] = None, max_shard_size: Optional[str] = "10GB"):
    model_args, _, training_args, finetuning_args, _, _ = get_train_args(args)
    model, tokenizer = load_model_and_tokenizer(model_args, finetuning_args)
    tokenizer.padding_side = "left"  # restore padding side
    tokenizer.init_kwargs["padding_side"] = "left"
    model.save_pretrained(training_args.output_dir, max_shard_size=max_shard_size)
    try:
        tokenizer.save_pretrained(training_args.output_dir)
    except:
        logger.warning("Cannot save tokenizer, please copy the files manually.")


if __name__ == "__main__":
    run_exp()
