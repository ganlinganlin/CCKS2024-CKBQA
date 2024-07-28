
# 这段代码涉及到哈希计算、类型提示和日志记录。以下是每个导入语句和相关变量的解释：
# 1. **`import hashlib`：**
#    - 导入 Python 内置的 `hashlib` 模块，用于进行哈希计算。
# 2. **`from typing import TYPE_CHECKING, Dict, List, Optional, Union`：**
#    - 导入一些与类型提示相关的类和模块，包括 `TYPE_CHECKING`、`Dict`、`List`、`Optional` 和 `Union`。
# 3. **`from llmtuner.extras.logging import get_logger`：**
#    - 导入 `get_logger` 函数，可能用于获取一个用于日志记录的 logger。
# 4. **`if TYPE_CHECKING:`：**
#    - 检查是否是类型检查阶段，如果是，执行以下的导入。
# 5. **`from datasets import Dataset, IterableDataset`：**
#    - 导入 `Dataset` 和 `IterableDataset` 类，这可能与数据集的操作有关。
# 6. **`from transformers import TrainingArguments`：**
#    - 导入 `TrainingArguments` 类，可能用于存储训练模型时的参数配置。
# 7. **`from llmtuner.hparams import DataArguments`：**
#    - 导入 `DataArguments` 类，这可能是与数据相关的一些参数配置。

import hashlib
from typing import TYPE_CHECKING, Dict, List, Optional, Union

from llmtuner.extras.logging import get_logger

if TYPE_CHECKING:
    from datasets import Dataset, IterableDataset
    from transformers import TrainingArguments
    from llmtuner.hparams import DataArguments

# 8. **`logger = get_logger(__name__)`：**
#    - 使用 `get_logger` 函数创建一个 logger 对象，用于在代码中进行日志记录。
# 9. **`EXT2TYPE` 字典:**
#    - 定义了一个名为 `EXT2TYPE` 的字典，将文件扩展名映射到文件类型。
#    - 例如，"csv" 映射到 "csv"，"json" 和 "jsonl" 都映射到 "json"，"txt" 映射到 "text"。
# 这些导入语句和变量定义显示了代码可能用于处理数据集、哈希计算和日志记录的一些功能。
logger = get_logger(__name__)

EXT2TYPE = {
    "csv": "csv",
    "json": "json",
    "jsonl": "json",
    "txt": "text"
}




# 这是一个名为 `checksum` 的函数，用于进行数据文件的 SHA-1 校验。以下是函数的关键点解释：
# 1. **参数:**
#    - `data_files`: 一个包含文件路径的列表，表示要进行校验的数据文件列表。
#    - `file_sha1`: 可选参数，表示期望的 SHA-1 哈希值。
# 2. **校验 SHA-1:**
#    - 如果没有提供 `file_sha1`（期望的哈希值），则记录一个警告，表示在 `dataset_info.json` 中缺少 SHA-1 哈希值。
#    - 如果数据文件列表不恰好包含一个文件，记录一个警告，表示文件数量不符合预期。
#    - 打开数据文件，读取其内容，并计算其 SHA-1 哈希值。
#    - 将计算得到的 SHA-1 哈希值与期望的哈希值进行比较。
#    - 如果两者不匹配，记录一个警告，表示 SHA-1 校验失败。
# 这个函数的主要目的是检查数据文件的 SHA-1 哈希值是否符合预期。如果没有提供期望的哈希值，或者文件数量不符合预期，或者哈希值不匹配，都会记录相应的警告信息。这是一种用于确保数据文件完整性的简单校验方法。如果你对其中的某一部分有特定的问题，或者需要更多的解释，请随时告诉我！
def checksum(data_files: List[str], file_sha1: Optional[str] = None) -> None:
    if file_sha1 is None:
        logger.warning("Checksum failed: missing SHA-1 hash value in dataset_info.json.")
        return

    if len(data_files) != 1:
        logger.warning("Checksum failed: too many files.")
        return

    with open(data_files[0], "rb") as f:
        sha1 = hashlib.sha1(f.read()).hexdigest()
        if sha1 != file_sha1:
            logger.warning("Checksum failed: mismatched SHA-1 hash value at {}.".format(data_files[0]))

# 这是一个名为 `split_dataset` 的函数，用于将数据集划分为训练集和验证集。以下是函数的关键点解释：
# 1. **参数:**
#    - `dataset`: 数据集对象，可以是常规的 `Dataset` 或可迭代的 `IterableDataset`。
#    - `data_args`: 数据相关的参数配置。
#    - `training_args`: 训练相关的参数配置。
# 2. **训练模式判定:**
#    - 检查 `training_args.do_train`，如果为 `True`，表示在训练模式下。
# 3. **验证集大小判定:**
#    - 如果验证集大小 (`data_args.val_size`) 大于 `1e-6`（一个非常小的正数），则进行数据集的划分。
#    - 如果 `data_args.streaming` 为 `True`，表示使用流式数据集，进行相应的划分和处理。
# 4. **训练模式下的数据集划分:**
#    - 如果使用流式数据集 (`data_args.streaming`)，则按照指定的验证集大小进行数据集的拆分。
#    - 如果不使用流式数据集，使用 `train_test_split` 方法划分训练集和验证集。
# 5. **返回值:**
#    - 返回一个字典，包含划分后的数据集，键为 "train_dataset" 和 "eval_dataset"。
# 这个函数的主要目的是根据训练模式和验证集大小，将数据集进行划分，以便在训练过程中使用。如果你对其中的某一部分有特定的问题，或者需要更多的解释，请随时告诉我！
def split_dataset(
    dataset: Union["Dataset", "IterableDataset"],
    data_args: "DataArguments",
    training_args: "TrainingArguments"
) -> Dict[str, "Dataset"]:
    if training_args.do_train:
        if data_args.val_size > 1e-6: # Split the dataset
            if data_args.streaming:
                val_set = dataset.take(int(data_args.val_size))
                train_set = dataset.skip(int(data_args.val_size))
                dataset = dataset.shuffle(buffer_size=data_args.buffer_size, seed=training_args.seed)
                return {"train_dataset": train_set, "eval_dataset": val_set}
            else:
                val_size = int(data_args.val_size) if data_args.val_size > 1 else data_args.val_size
                dataset = dataset.train_test_split(test_size=val_size, seed=training_args.seed)
                return {"train_dataset": dataset["train"], "eval_dataset": dataset["test"]}
        else:
            if data_args.streaming:
                dataset = dataset.shuffle(buffer_size=data_args.buffer_size, seed=training_args.seed)
            return {"train_dataset": dataset}
    else: # do_eval or do_predict
        return {"eval_dataset": dataset}
