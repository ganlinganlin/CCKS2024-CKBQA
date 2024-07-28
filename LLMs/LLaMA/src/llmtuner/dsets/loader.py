
# 1. `os`: 提供了与操作系统交互的功能，例如路径操作等。
# 2. `from typing import TYPE_CHECKING, List, Union`: 从 `typing` 模块导入了一些类型提示相关的类和工具。`TYPE_CHECKING` 通常用于在类型检查时执行一些特定的代码。
# 3. `from datasets import concatenate_datasets, interleave_datasets, load_dataset`: 导入了一些与数据集处理相关的函数。这可能与 Hugging Face 的 datasets 库有关。
# 4. `from llmtuner.dsets.utils import checksum, EXT2TYPE`: 导入了来自 `llmtuner.dsets.utils` 模块的 `checksum` 和 `EXT2TYPE`。
# 5. `from llmtuner.extras.logging import get_logger`: 导入了来自 `llmtuner.extras.logging` 模块的 `get_logger` 函数。
# 6. `if TYPE_CHECKING: ...`: 这段代码在检查是否处于类型检查模式。在类型检查模式时，一些特定的代码块可能会被执行。
# 7. `from datasets import Dataset, IterableDataset` 和 `from llmtuner.hparams import ModelArguments, DataArguments`: 进一步导入了一些数据集和超参数相关的类。

import os
from typing import TYPE_CHECKING, List, Union
from datasets import concatenate_datasets, interleave_datasets, load_dataset

from llmtuner.dsets.utils import checksum, EXT2TYPE
from llmtuner.extras.logging import get_logger

if TYPE_CHECKING:
    from datasets import Dataset, IterableDataset
    from llmtuner.hparams import ModelArguments, DataArguments


# 这段代码看起来是一个函数 `get_dataset`，该函数用于加载和处理数据集。以下是代码的主要逻辑：
# 1. `logger = get_logger(__name__)`: 创建一个日志记录器，该记录器使用了名为 `__name__` 的模块名。
logger = get_logger(__name__)


# 2. `get_dataset` 函数:
#    - 接受两个参数 `model_args` 和 `data_args`，它们可能是与模型和数据相关的参数。
#    - 返回一个 `Dataset` 或 `IterableDataset` 对象，具体返回类型取决于数据集是可迭代的还是不可迭代的。
def get_dataset(
    model_args: "ModelArguments",
    data_args: "DataArguments"
) -> Union["Dataset", "IterableDataset"]:
    max_samples = data_args.max_samples
    all_datasets: List[Union["Dataset", "IterableDataset"]] = [] # support multiple datasets

    # 3. 函数内部逻辑：
    #    - 遍历数据集列表 `data_args.dataset_list`，该列表包含要加载的不同数据集的属性。
    #    - 根据每个数据集的属性，确定数据加载的来源（来自 Hugging Face Hub，本地脚本或本地文件）。

    for dataset_attr in data_args.dataset_list:
        logger.info("Loading dataset {}...".format(dataset_attr))

        if dataset_attr.load_from == "hf_hub":
            data_path = dataset_attr.dataset_name
            data_files = None
        elif dataset_attr.load_from == "script":
            data_path = os.path.join(data_args.dataset_dir, dataset_attr.dataset_name)
            data_files = None
        elif dataset_attr.load_from == "file":
            data_path = None
            data_files: List[str] = []

            if os.path.isdir(os.path.join(data_args.dataset_dir, dataset_attr.dataset_name)): # directory
                for file_name in os.listdir(os.path.join(data_args.dataset_dir, dataset_attr.dataset_name)):
                    data_files.append(os.path.join(data_args.dataset_dir, dataset_attr.dataset_name, file_name))
                    if data_path is None:
                        data_path = EXT2TYPE.get(file_name.split(".")[-1], None)
                    else:
                        assert data_path == EXT2TYPE.get(file_name.split(".")[-1], None), "file type does not match."
            elif os.path.isfile(os.path.join(data_args.dataset_dir, dataset_attr.dataset_name)): # single file
                data_files.append(os.path.join(data_args.dataset_dir, dataset_attr.dataset_name))
                data_path = EXT2TYPE.get(dataset_attr.dataset_name.split(".")[-1], None)
            else:
                raise ValueError("File not found.")

            assert data_path, "File extension must be txt, csv, json or jsonl."
            checksum(data_files, dataset_attr.dataset_sha1)
        else:
            raise NotImplementedError

        #    - 使用 `load_dataset` 函数加载数据集，可以从文件、目录或 Hugging Face Hub 加载。
        dataset = load_dataset(
            data_path,
            data_files=data_files,
            split=data_args.split,
            cache_dir=model_args.cache_dir,
            streaming=data_args.streaming,
            use_auth_token=True if model_args.use_auth_token else None
        )

        #    - 如果设置了 `max_samples`，截取数据集的前 `max_samples` 个样本。
        #    - 对数据集进行列重命名，以确保数据集的列名与期望的一致。
        #    - 如果设置了 `system_prompt`，则将系统提示添加到数据集中。
        #    - 将处理后的数据集添加到 `all_datasets` 列表中。
        if max_samples is not None:
            max_samples_temp = min(len(dataset), max_samples)
            dataset = dataset.select(range(max_samples_temp))

        for column_name in ["prompt", "query", "response", "history"]: # align datasets
            if getattr(dataset_attr, column_name) and getattr(dataset_attr, column_name) != column_name:
                dataset = dataset.rename_column(getattr(dataset_attr, column_name), column_name)

        if dataset_attr.system_prompt: # add system prompt
            if data_args.streaming:
                dataset = dataset.map(lambda _: {"system": dataset_attr.system_prompt})
            else:
                dataset = dataset.add_column("system", [dataset_attr.system_prompt] * len(dataset))

        all_datasets.append(dataset)


    # 4. 最后，根据不同的数据集混合策略，返回一个合并或混合后的数据集。
    # 这段代码主要用于处理和加载多个数据集，并根据指定的参数进行相应的处理。
    if len(data_args.dataset_list) == 1:
        return all_datasets[0]
    elif data_args.mix_strategy == "concat":
        if data_args.streaming:
            logger.warning("The samples between different datasets will not be mixed in streaming mode.")
        return concatenate_datasets(all_datasets)
    elif data_args.mix_strategy.startswith("interleave"):
        if not data_args.streaming:
            logger.warning("We recommend using `mix_strategy=concat` in non-streaming mode.")
        stopping_strategy = "first_exhausted" if data_args.mix_strategy.endswith("under") else "all_exhausted"
        return interleave_datasets(all_datasets, data_args.interleave_probs, stopping_strategy=stopping_strategy)
    else:
        raise ValueError("Unknown mixing strategy.")
