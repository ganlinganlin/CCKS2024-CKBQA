
# 这段代码导入了一些 Python 内置的模块和一些额外的库，包括：
# - `os`: 提供与操作系统相关的功能，如文件和目录操作。
# - `json`: 用于处理 JSON 格式的数据。
# - `List`: 用于表示列表类型的类型提示。
# - `Literal`: 用于表示包含有限个固定值的类型。
# - `Optional`: 用于表示可选的类型。
# - `dataclass`: 用于创建数据类，通过装饰器方式简化类的定义。
# - `field`: 用于定义数据类中的字段，提供对字段的额外配置。
# 这些库和模块通常在 Python 中用于文件操作、数据处理和类型提示等任务。
import os
import json
from typing import List, Literal, Optional
from dataclasses import dataclass, field

# 这是一个使用 `dataclass` 装饰器定义的 Python 数据类 (`DatasetAttr`)。它包含以下字段：
# - `load_from`: 一个字符串，表示数据集加载的来源。
# - `dataset_name`: 一个可选的字符串，表示数据集的名称。
# - `dataset_sha1`: 一个可选的字符串，表示数据集的 SHA-1 哈希值。
# - `system_prompt`: 一个可选的字符串，表示系统提示。
# - `ranking`: 一个可选的布尔值，默认为 `False`，表示是否进行排名。
# - `prompt`: 一个可选的字符串，表示提示。
# - `query`: 一个可选的字符串，表示查询。
# - `response`: 一个可选的字符串，表示响应。
# - `history`: 一个可选的字符串，表示历史。
# `__repr__` 方法被重写，用于返回数据集名称的字符串表示形式。
# 这个数据类的目的可能是用于配置数据集的相关属性。
@dataclass
class DatasetAttr:

    load_from: str
    dataset_name: Optional[str] = None
    dataset_sha1: Optional[str] = None
    system_prompt: Optional[str] = None
    ranking: Optional[bool] = False
    prompt: Optional[str] = "instruction"
    query: Optional[str] = "input"
    response: Optional[str] = "output"
    history: Optional[str] = None

    def __repr__(self) -> str:
        return self.dataset_name


# 这是一个使用 `dataclass` 装饰器定义的 Python 数据类 (`DataArguments`)。它包含了一系列与数据集和训练参数相关的字段：
# - `template`: 一个可选的字符串，表示在训练和推断中用于构建提示的模板。
# - `dataset`: 一个可选的字符串，表示要使用的提供的数据集的名称。使用逗号分隔多个数据集。
# - `dataset_dir`: 一个可选的字符串，表示包含数据集的文件夹的名称。
# - `split`: 一个可选的字符串，表示用于训练和评估的数据集拆分。
# - `cutoff_len`: 一个可选的整数，表示在标记化之后模型输入的最大长度。
# - `streaming`: 一个可选的布尔值，表示是否启用流模式。
# - `buffer_size`: 一个可选的整数，表示在流模式下从中随机采样示例的缓冲区的大小。
# - `mix_strategy`: 一个可选的字符串，表示数据集混合中使用的策略。
# - `interleave_probs`: 一个可选的字符串，表示从数据集中采样数据的概率。使用逗号分隔多个数据集。
# - `overwrite_cache`: 一个可选的布尔值，表示是否覆盖缓存的训练和评估集。
# - `preprocessing_num_workers`: 一个可选的整数，表示用于预处理的进程数。
# - `max_samples`: 一个可选的整数，用于调试目的，截断每个数据集的示例数。
# - `eval_num_beams`: 一个可选的整数，表示用于评估的 beam 数。该参数将传递给 `model.generate`。
# - `ignore_pad_token_for_loss`: 一个可选的布尔值，表示在损失计算中是否忽略与填充标签对应的标记。
# - `system_prompt`: 一个可选的字符串，表示在用户查询之前添加的系统提示。在训练中使用 `|` 分隔多个提示。
# - `val_size`: 一个可选的浮点数，表示开发集的大小，应为整数或范围为 `[0,1)` 的浮点数。
# - `sft_packing`: 一个可选的布尔值，表示是否在监督微调阶段对问题和答案进行打包。
# 这个数据类的目的是存储用于训练和评估的数据集相关的参数。
@dataclass
class DataArguments:
    r"""
    Arguments pertaining to what data we are going to input our model for training and evaluation.
    """
    template: Optional[str] = field(
        default=None,
        metadata={"help": "Which template to use for constructing prompts in training and inference."}
    )
    dataset: Optional[str] = field(
        default="alpaca_en",
        metadata={"help": "The name of provided dataset(s) to use. Use commas to separate multiple datasets."}
    )
    dataset_dir: Optional[str] = field(
        default="data",
        metadata={"help": "The name of the folder containing datasets."}
    )
    split: Optional[str] = field(
        default="train",
        metadata={"help": "Which dataset split to use for training and evaluation."}
    )
    cutoff_len: Optional[int] = field(
        default=2048,
        metadata={"help": "The maximum length of the model inputs after tokenization."}
    )
    streaming: Optional[bool] = field(
        default=False,
        metadata={"help": "Enable streaming mode."}
    )
    buffer_size: Optional[int] = field(
        default=16384,
        metadata={"help": "Size of the buffer to randomly sample examples from in streaming mode."}
    )
    mix_strategy: Optional[Literal["concat", "interleave_under", "interleave_over"]] = field(
        default="concat",
        metadata={"help": "Strategy to use in dataset mixing."}
    )
    interleave_probs: Optional[str] = field(
        default=None,
        metadata={"help": "Probabilities to sample data from datasets. Use commas to separate multiple datasets."}
    )
    overwrite_cache: Optional[bool] = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets."}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."}
    )
    max_samples: Optional[int] = field(
        default=None,
        metadata={"help": "For debugging purposes, truncate the number of examples for each dataset."}
    )
    eval_num_beams: Optional[int] = field(
        default=None,
        metadata={"help": "Number of beams to use for evaluation. This argument will be passed to `model.generate`"}
    )
    ignore_pad_token_for_loss: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."}
    )
    system_prompt: Optional[str] = field(
        default=None,
        metadata={"help": "System prompt to add before the user query. Use `|` to separate multiple prompts in training."}
    )
    val_size: Optional[float] = field(
        default=0,
        metadata={"help": "Size of the development set, should be an integer or a float in range `[0,1)`."}
    )
    sft_packing: Optional[bool] = field(
        default=False,
        metadata={"help": "Packing the questions and answers in the supervised fine-tuning stage."}
    )

    # 这个 `init_for_training` 方法的目的是为训练阶段初始化数据集。它支持混合多个数据集。具体步骤包括：
    # 1. 将 `dataset` 字段中的数据集名称拆分为一个列表，并读取 `dataset_info.json` 文件。
    # 2. 根据数据集信息，初始化 `DatasetAttr` 对象的列表 `dataset_list`。
    # 3. 如果提供了 `system_prompt`，则将其拆分为列表，确保其数量与数据集数量一致。
    # 4. 如果提供了 `interleave_probs`，则将其转换为浮点数列表。
    # 5. 针对每个数据集，根据 `dataset_info` 中的信息创建一个 `DatasetAttr` 对象，并将其添加到 `dataset_list` 列表中。
    # 这个方法的最终结果是一个 `dataset_list`，其中包含了每个数据集的相关属性。
    def init_for_training(self): # support mixing multiple datasets
        dataset_names = [ds.strip() for ds in self.dataset.split(",")]
        with open(os.path.join(self.dataset_dir, "dataset_info.json"), "r") as f:
            dataset_info = json.load(f)

        prompt_list = self.system_prompt.split("|") if self.system_prompt else [None]
        prompt_list = prompt_list * (len(dataset_names) // len(prompt_list))
        assert len(prompt_list) == len(dataset_names), "Number of system prompts should be equal to datasets or 1."

        if self.interleave_probs is not None:
            self.interleave_probs = [float(prob.strip()) for prob in self.interleave_probs.split(",")]

        self.dataset_list: List[DatasetAttr] = []
        for i, name in enumerate(dataset_names):
            if name not in dataset_info:
                raise ValueError("Undefined dataset {} in dataset_info.json.".format(name))

            if "hf_hub_url" in dataset_info[name]:
                dataset_attr = DatasetAttr("hf_hub", dataset_name=dataset_info[name]["hf_hub_url"])
            elif "script_url" in dataset_info[name]:
                dataset_attr = DatasetAttr("script", dataset_name=dataset_info[name]["script_url"])
            else:
                dataset_attr = DatasetAttr(
                    "file",
                    dataset_name=dataset_info[name]["file_name"],
                    dataset_sha1=dataset_info[name].get("file_sha1", None)
                )

            if "columns" in dataset_info[name]:
                dataset_attr.prompt = dataset_info[name]["columns"].get("prompt", None)
                dataset_attr.query = dataset_info[name]["columns"].get("query", None)
                dataset_attr.response = dataset_info[name]["columns"].get("response", None)
                dataset_attr.history = dataset_info[name]["columns"].get("history", None)

            dataset_attr.ranking = dataset_info[name].get("ranking", False)
            dataset_attr.system_prompt = prompt_list[i]
            self.dataset_list.append(dataset_attr)
