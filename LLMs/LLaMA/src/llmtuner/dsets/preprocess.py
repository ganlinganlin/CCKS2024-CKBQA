
# 这段代码涉及到一些模块和类型的导入，以下是对代码的简要解释：
# 1. `import tiktoken`: 导入了 `tiktoken` 模块，这可能是用于计算token数量的工具。
# 2. `from typing import TYPE_CHECKING, Any, Dict, Generator, List, Literal, Union`: 导入了一系列类型提示相关的类和工具，
#     包括了 `TYPE_CHECKING`，`Any`，`Dict`，`Generator`，`List`，`Literal`，和 `Union`。
# 3. `from llmtuner.extras.constants import IGNORE_INDEX`: 从 `llmtuner.extras.constants` 模块导入了名为 `IGNORE_INDEX` 的常量。
# 4. `from llmtuner.extras.template import get_template_and_fix_tokenizer`: 从 `llmtuner.extras.template` 模块导入了 `get_template_and_fix_tokenizer` 函数。
# 5. `if TYPE_CHECKING: ...`: 这段代码在检查是否处于类型检查模式。在类型检查模式时，一些特定的代码块可能会被执行。
# 6. `from datasets import Dataset, IterableDataset`: 导入了 `Dataset` 和 `IterableDataset` 类，这些可能与 Hugging Face 的 datasets 库有关。
# 7. `from transformers import Seq2SeqTrainingArguments`: 导入了 `Seq2SeqTrainingArguments` 类，这可能与 Hugging Face 的 transformers 库中的模型训练参数有关。
# 8. `from transformers.tokenization_utils import PreTrainedTokenizer`: 导入了 `PreTrainedTokenizer` 类，这可能与 Hugging Face 的 transformers 库中的分词器有关。
# 9. `from llmtuner.hparams import DataArguments`: 从 `llmtuner.hparams` 模块导入了 `DataArguments` 类，这可能是一些超参数相关的定义。
# 这些导入语句表明你的代码中涉及到了处理 NLP 模型训练数据、模型训练参数以及一些超参数的相关逻辑。
import tiktoken
from typing import TYPE_CHECKING, Any, Dict, Generator, List, Literal, Union
from itertools import chain

from llmtuner.extras.constants import IGNORE_INDEX
from llmtuner.extras.template import get_template_and_fix_tokenizer

if TYPE_CHECKING:
    from datasets import Dataset, IterableDataset
    from transformers import Seq2SeqTrainingArguments
    from transformers.tokenization_utils import PreTrainedTokenizer
    from llmtuner.hparams import DataArguments


# 这段代码定义了一个名为 `preprocess_dataset` 的函数，用于预处理数据集。以下是代码的主要逻辑：
# 1. 函数签名：
#    - 接受参数：
#       - `dataset`: 要预处理的数据集，可以是 `Dataset` 或 `IterableDataset` 类型。
#       - `tokenizer`: 预训练分词器，类型为 `PreTrainedTokenizer`。
#       - `data_args`: 数据集相关的超参数，类型为 `DataArguments`。
#       - `training_args`: 模型训练参数，类型为 `Seq2SeqTrainingArguments`。
#       - `stage`: 预处理阶段，取值为 "pt"、"sft"、"rm" 或 "ppo"，使用 `Literal` 类型提示确保只接受这些值。
# 2. 获取列名和模板：
#    - 通过 `list(next(iter(dataset)).keys())` 获取数据集的列名。
#    - 调用 `get_template_and_fix_tokenizer` 函数获取模板，并在需要时调整分词器。
# 3. 验证模板与参数的兼容性：
#    - 如果模板存在且支持有效的 EOS（end-of-sequence）并且设置了 `sft_packing`，则抛出值错误，因为当前模板与 packing 不兼容。
def preprocess_dataset(
    dataset: Union["Dataset", "IterableDataset"],
    tokenizer: "PreTrainedTokenizer",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    stage: Literal["pt", "sft", "rm", "ppo"]
) -> Union["Dataset", "IterableDataset"]:
    column_names = list(next(iter(dataset)).keys())
    template = get_template_and_fix_tokenizer(data_args.template, tokenizer)

    if template is not None and template.efficient_eos and data_args.sft_packing:
        raise ValueError("Current template is incompatible with packing.")

    # 这段代码定义了一个名为 `construct_example` 的函数，用于构建示例。以下是代码的主要逻辑：
    # 1. 函数签名：
    #    - 接受参数 `examples`，这是一个字典，包含不同类型的示例，例如 "prompt"、"response"、"query"、"history" 和 "system"。
    #    - 返回一个生成器（Generator）。
    # 2. 生成器逻辑：
    #    - 通过循环遍历 `range(len(examples["prompt"]))`，迭代数据集中的每个示例。
    #    - 对于每个示例，获取 "prompt" 和 "response" 的值，并将它们赋给 `query` 和 `response` 变量。
    #    - 如果 "query" 存在且非空，则将其添加到 `query` 中。
    #    - 如果 "history" 存在，则将其值赋给 `history` 变量，否则将其设为 `None`。
    #    - 如果 "system" 存在，则将其值赋给 `system` 变量，否则将其设为 `None`。
    #    - 使用 `yield` 语句生成一个包含 `query`、`response`、`history` 和 `system` 的元组。
    # 这个函数的目的似乎是将输入的字典中的各个字段值按照一定规则组合成一个元组，并通过生成器逐个返回。如果你有特定的问题或需要更多的解释，请提出。
    def construct_example(examples: Dict[str, List[Any]]) -> Generator[Any, None, None]:
        for i in range(len(examples["prompt"])):
            query, response = examples["prompt"][i], examples["response"][i]
            query = query + "\n" + examples["query"][i] if "query" in examples and examples["query"][i] else query
            history = examples["history"][i] if "history" in examples else None
            system = examples["system"][i] if "system" in examples else None
            yield query, response, history, system

    # 这段代码定义了一个名为 `preprocess_pretrain_dataset` 的函数，用于预处理预训练数据集。以下是代码的主要逻辑：
    # 1. 函数签名：
    #    - 接受参数 `examples`，这是一个字典，包含不同类型的示例，例如 "prompt"、"response"、"query"、"history" 和 "system"。
    #    - 返回一个字典。
    # 2. 预处理逻辑：
    #    - 检查 `tokenizer` 是否具有 `tokenizer` 属性，并且该属性的值是否是 `tiktoken.Encoding` 类型。如果是，设置 `kwargs` 字典的 `allowed_special` 为 "all"，
    #       这似乎是为 `tiktoken` 分词器特定的参数设置。
    #    - 检查 `tokenizer` 是否具有 `add_bos_token` 和 `add_eos_token` 属性，如果有，则将它们都设置为 `True`，这似乎是为了处理某种特定的分词器（如 LLaMA 分词器）。
    #    - 使用 `tokenizer` 对 "prompt" 字段的值进行分词，并将分词后的结果存储在 `tokenized_examples` 中。
    #    - 将分词后的结果连接成一个长文本，并将结果存储在 `concatenated_examples` 中。
    #    - 计算总文本长度 `total_length`，并设置 `block_size` 为 `data_args.cutoff_len`。
    #    - 将文本按照 `block_size` 切分成多个块，并存储在 `result` 字典中。
    # 这个函数的目的似乎是将输入的字典中的 "prompt" 字段的文本进行分词和预处理，得到适合用于预训练任务的数据。如果你有特定的问题或需要更多的解释，请提出。
    def preprocess_pretrain_dataset(examples: Dict[str, List[Any]]) -> Dict[str, Any]:
        # build grouped texts with format `X1 X2 X3 ...`
        if isinstance(getattr(tokenizer, "tokenizer", None), tiktoken.Encoding):
            kwargs = dict(allowed_special="all") # for tiktoken tokenizer (Qwen)
        else:
            kwargs = dict(add_special_tokens=True)

        if hasattr(tokenizer, "add_bos_token") and hasattr(tokenizer, "add_eos_token"):
            setattr(tokenizer, "add_bos_token", True) # for LLaMA tokenizer
            setattr(tokenizer, "add_eos_token", True)

        tokenized_examples = tokenizer(examples["prompt"], **kwargs)
        concatenated_examples = {k: list(chain(*tokenized_examples[k])) for k in tokenized_examples.keys()}
        total_length = len(concatenated_examples[list(concatenated_examples.keys())[0]])
        block_size = data_args.cutoff_len
        # we drop the small remainder, and if the total_length < block_size, we exclude this batch
        total_length = (total_length // block_size) * block_size
        # split by chunks of cutoff_len
        result = {
            k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        return result

    # 这段代码定义了一个名为 `preprocess_supervised_dataset` 的函数，用于预处理监督学习任务的数据集。以下是代码的主要逻辑：
    # 1. 函数签名：
    #    - 接受参数 `examples`，这是一个字典，包含不同类型的示例，例如 "prompt"、"response"、"query"、"history" 和 "system"。
    #    - 返回一个字典，包含了模型输入的 "input_ids"、"attention_mask" 以及标签 "labels"。
    # 2. 预处理逻辑：
    #    - 构造一个空字典 `model_inputs`，用于存储模型输入的各个部分。
    #    - 使用 `construct_example` 函数对每个示例进行迭代。
    #    - 对每个示例的每个轮次（turn）进行处理：
    #       - 获取源（source）和目标（target）的编码，这使用了 `template.encode_multiturn` 函数。
    #       - 根据 `data_args.cutoff_len` 限制源和目标的长度，确保它们不超过指定的长度。
    #       - 构建输入（`input_ids`）和标签（`labels`）序列，其中源的部分被标记为 `<ignore>`。
    #    - 如果使用了 `template.efficient_eos`，在每个示例的最后添加 EOS（end-of-sequence）标记。
    #    - 如果总长度超过了 `data_args.cutoff_len`，则截断输入和标签。
    #    - 将处理后的输入、注意力掩码和标签添加到 `model_inputs` 中。
    # 这个函数的目的似乎是将输入的字典中的文本序列进行编码，生成用于监督学习训练的模型输入。如果你有特定的问题或需要更多的解释，请提出。
    def preprocess_supervised_dataset(examples: Dict[str, List[Any]]) -> Dict[str, Any]:
        # build inputs with format `<bos> X Y <eos>` and labels with format `<ignore> ... <ignore> Y <eos>`
        # for multiturn examples, we only mask the prompt part in each prompt-response pair.
        model_inputs = {"input_ids": [], "attention_mask": [], "labels": []}

        for query, response, history, system in construct_example(examples):
            input_ids, labels = [], []

            for turn_idx, (source_ids, target_ids) in enumerate(template.encode_multiturn(
                tokenizer, query, response, history, system
            )):
                total_len = len(source_ids) + len(target_ids)
                max_source_len = int(data_args.cutoff_len * (len(source_ids) / total_len))
                max_target_len = int(data_args.cutoff_len * (len(target_ids) / total_len))

                if len(source_ids) > max_source_len:
                    source_ids = source_ids[:max_source_len]
                if len(target_ids) > max_target_len:
                    target_ids = target_ids[:max_target_len]

                if turn_idx != 0 and template.efficient_eos:
                    source_mask = [tokenizer.eos_token_id] + [IGNORE_INDEX] * (len(source_ids) - 1)
                else:
                    source_mask = [IGNORE_INDEX] * len(source_ids)

                input_ids += source_ids + target_ids
                labels += source_mask + target_ids

            if template.efficient_eos:
                input_ids += [tokenizer.eos_token_id]
                labels += [tokenizer.eos_token_id]

            if len(input_ids) > data_args.cutoff_len:
                input_ids = input_ids[:data_args.cutoff_len]
                labels = labels[:data_args.cutoff_len]

            model_inputs["input_ids"].append(input_ids)
            model_inputs["attention_mask"].append([1] * len(input_ids))
            model_inputs["labels"].append(labels)

        return model_inputs

    # 这段代码定义了一个名为 `preprocess_packed_supervised_dataset` 的函数，用于预处理使用 packing（打包）策略的监督学习任务的数据集。以下是代码的主要逻辑：
    # 1. 函数签名：
    #    - 接受参数 `examples`，这是一个字典，包含不同类型的示例，例如 "prompt"、"response"、"query"、"history" 和 "system"。
    #    - 返回一个字典，包含了模型输入的 "input_ids"、"attention_mask" 以及标签 "labels"。
    # 2. 预处理逻辑：
    #    - 构造一个空字典 `model_inputs`，用于存储模型输入的各个部分。
    #    - 使用 `construct_example` 函数对每个示例进行迭代。
    #    - 对每个示例的每个轮次（turn）进行处理：
    #       - 获取源（source）和目标（target）的编码，这使用了 `template.encode_multiturn` 函数。
    #       - 将源和目标的编码添加到 `input_ids` 和 `labels` 中，这里没有对输入进行遮蔽。
    #    - 计算总长度 `total_length`，并设置 `block_size` 为 `data_args.cutoff_len`。
    #    - 将文本按照 `block_size` 切分成多个块，并存储在 `model_inputs` 字典中。
    # 这个函数的目的似乎是将输入的字典中的文本序列进行编码，生成用于监督学习训练的模型输入，同时使用了 packing 策略。如果你有特定的问题或需要更多的解释，请提出。
    def preprocess_packed_supervised_dataset(examples: Dict[str, List[Any]]) -> Dict[str, Any]:
        # build inputs with format `<bos> X Y <eos>` and labels with format `<bos> X Y <eos>`
        # we do not mask the inputs in packed training.
        model_inputs = {"input_ids": [], "attention_mask": [], "labels": []}
        input_ids, labels = [], []
        for query, response, history, system in construct_example(examples):
            for source_ids, target_ids in template.encode_multiturn(tokenizer, query, response, history, system):
                input_ids += source_ids + target_ids
                labels += source_ids + target_ids # TODO: try masking source_ids here

        total_length = len(input_ids)
        block_size = data_args.cutoff_len
        # we drop the small remainder, and if the total_length < block_size, we exclude this batch
        total_length = (total_length // block_size) * block_size
        # split by chunks of cutoff_len
        for i in range(0, total_length, block_size):
            model_inputs["input_ids"].append(input_ids[i: i + block_size])
            model_inputs["attention_mask"].append([1] * block_size)
            model_inputs["labels"].append(labels[i: i + block_size])

        return model_inputs

    # 这段代码定义了一个名为 `preprocess_unsupervised_dataset` 的函数，用于预处理无监督学习任务的数据集。以下是代码的主要逻辑：
    # 1. 函数签名：
    #    - 接受参数 `examples`，这是一个字典，包含不同类型的示例，例如 "prompt"、"response"、"query"、"history" 和 "system"。
    #    - 返回一个字典，包含了模型输入的 "input_ids"、"attention_mask" 以及标签 "labels"。
    # 2. 预处理逻辑：
    #    - 构造一个空字典 `model_inputs`，用于存储模型输入的各个部分。
    #    - 使用 `construct_example` 函数对每个示例进行迭代。
    #    - 对每个示例进行处理：
    #       - 获取单个轮次（turn）的源（source）和目标（target）的编码，这使用了 `template.encode_oneturn` 函数。
    #       - 如果使用了 `template.efficient_eos`，在目标的末尾添加 EOS（end-of-sequence）标记。
    #       - 如果长度超过了 `data_args.cutoff_len`，则截断输入和标签。
    #       - 将处理后的输入、注意力掩码和标签添加到 `model_inputs` 中。
    # 这个函数的目的似乎是将输入的字典中的文本序列进行编码，生成用于无监督学习训练的模型输入。如果你有特定的问题或需要更多的解释，请提出。
    def preprocess_unsupervised_dataset(examples: Dict[str, List[Any]]) -> Dict[str, Any]:
        # build inputs with format `<bos> X` and labels with format `Y <eos>`
        model_inputs = {"input_ids": [], "attention_mask": [], "labels": []}

        for query, response, history, system in construct_example(examples):
            input_ids, labels = template.encode_oneturn(tokenizer, query, response, history, system)

            if template.efficient_eos:
                labels += [tokenizer.eos_token_id]

            if len(input_ids) > data_args.cutoff_len:
                input_ids = input_ids[:data_args.cutoff_len]
            if len(labels) > data_args.cutoff_len:
                labels = labels[:data_args.cutoff_len]

            model_inputs["input_ids"].append(input_ids)
            model_inputs["attention_mask"].append([1] * len(input_ids))
            model_inputs["labels"].append(labels)

        return model_inputs

    # 这段代码定义了一个名为 `preprocess_pairwise_dataset` 的函数，用于预处理成对数据集，其中每个示例包含两个不同的回复。以下是代码的主要逻辑：
    # 1. 函数签名：
    #    - 接受参数 `examples`，这是一个字典，包含不同类型的示例，例如 "prompt"、"response"、"query"、"history" 和 "system"。
    #    - 返回一个字典，包含了模型输入的 "prompt_ids"、"chosen_ids" 和 "rejected_ids"。
    # 2. 预处理逻辑：
    #    - 构造一个空字典 `model_inputs`，用于存储模型输入的各个部分。
    #    - 使用 `construct_example` 函数对每个示例进行迭代。
    #    - 对每个示例进行处理：
    #       - 获取单个轮次（turn）的源（source）和两个目标（target）的编码，这使用了 `template.encode_oneturn` 函数。
    #       - 如果使用了 `template.efficient_eos`，在目标的末尾添加 EOS（end-of-sequence）标记。
    #       - 计算总长度 `total_len`，以确定源和目标的长度截断。
    #       - 根据长度截断源和目标的编码。
    #       - 将处理后的源、目标1和目标2的编码添加到 `model_inputs` 中。
    # 这个函数的目的似乎是将输入的字典中的文本序列进行编码，生成用于处理成对数据集的模型输入。如果你有特定的问题或需要更多的解释，请提出。
    def preprocess_pairwise_dataset(examples):
        # build input pairs with format `<bos> X`, `Y1 <eos>` and `Y2 <eos>`
        model_inputs = {"prompt_ids": [], "chosen_ids": [], "rejected_ids": []}
        for query, response, history, system in construct_example(examples):
            prompt_ids, chosen_ids = template.encode_oneturn(tokenizer, query, response[0], history, system)
            _, rejected_ids = template.encode_oneturn(tokenizer, query, response[1], history, system)

            if template.efficient_eos:
                chosen_ids += [tokenizer.eos_token_id]
                rejected_ids += [tokenizer.eos_token_id]

            total_len = len(prompt_ids) + max(len(chosen_ids), len(rejected_ids))
            max_source_len = int(data_args.cutoff_len * (len(prompt_ids) / total_len))
            max_target_len = int(data_args.cutoff_len * (max(len(chosen_ids), len(rejected_ids)) / total_len))

            if len(prompt_ids) > max_source_len:
                prompt_ids = prompt_ids[:max_source_len]
            if len(chosen_ids) > max_target_len:
                chosen_ids = chosen_ids[:max_target_len]
            if len(rejected_ids) > max_target_len:
                rejected_ids = rejected_ids[:max_target_len]

            model_inputs["prompt_ids"].append(prompt_ids)
            model_inputs["chosen_ids"].append(chosen_ids)
            model_inputs["rejected_ids"].append(rejected_ids)
        return model_inputs

    # 这段代码定义了一个名为 `print_supervised_dataset_example` 的函数，用于打印监督学习数据集中的一个示例。以下是代码的主要逻辑：
    # 1. 函数签名：
    #    - 接受一个参数 `example`，这是一个字典，包含了模型输入的 "input_ids"、"attention_mask" 以及标签 "labels"。
    # 2. 打印逻辑：
    #    - 打印 "input_ids"、"inputs"、"label_ids" 以及 "labels" 的信息。
    #    - 使用 `tokenizer.decode` 函数将 "input_ids" 和 "labels" 解码为文本形式进行打印。
    #    - 在解码 "labels" 时，使用了 `IGNORE_INDEX` 进行过滤，以排除特殊标记。
    # 这个函数的目的是用于在控制台上展示一个监督学习数据集的样例，包括输入文本和相应的标签。如果你有特定的问题或需要更多的解释，请提出。
    def print_supervised_dataset_example(example):
        print("input_ids:\n{}".format(example["input_ids"]))
        print("inputs:\n{}".format(tokenizer.decode(example["input_ids"], skip_special_tokens=False)))
        print("label_ids:\n{}".format(example["labels"]))
        print("labels:\n{}".format(
            tokenizer.decode(list(filter(lambda x: x != IGNORE_INDEX, example["labels"])), skip_special_tokens=False)
        ))

    # 这段代码定义了一个名为 `print_pairwise_dataset_example` 的函数，用于打印成对数据集中的一个示例。以下是代码的主要逻辑：
    # 1. 函数签名：
    #    - 接受一个参数 `example`，这是一个字典，包含了模型输入的 "prompt_ids"、"chosen_ids" 和 "rejected_ids"。
    # 2. 打印逻辑：
    #    - 打印 "prompt_ids"、"prompt"、"chosen_ids"、"chosen"、"rejected_ids" 以及 "rejected" 的信息。
    #    - 使用 `tokenizer.decode` 函数将 "prompt_ids"、"chosen_ids" 和 "rejected_ids" 解码为文本形式进行打印。
    # 这个函数的目的是用于在控制台上展示一个成对数据集的样例，包括输入文本和相应的标签。如果你有特定的问题或需要更多的解释，请提出。
    def print_pairwise_dataset_example(example):
        print("prompt_ids:\n{}".format(example["prompt_ids"]))
        print("prompt:\n{}".format(tokenizer.decode(example["prompt_ids"], skip_special_tokens=False)))
        print("chosen_ids:\n{}".format(example["chosen_ids"]))
        print("chosen:\n{}".format(tokenizer.decode(example["chosen_ids"], skip_special_tokens=False)))
        print("rejected_ids:\n{}".format(example["rejected_ids"]))
        print("rejected:\n{}".format(tokenizer.decode(example["rejected_ids"], skip_special_tokens=False)))

    # 这段代码定义了一个名为 `print_unsupervised_dataset_example` 的函数，用于打印无监督学习数据集中的一个示例。以下是代码的主要逻辑：
    # 1. 函数签名：
    #    - 接受一个参数 `example`，这是一个字典，包含了模型输入的 "input_ids"。
    # 2. 打印逻辑：
    #    - 打印 "input_ids" 和 "inputs" 的信息。
    #    - 使用 `tokenizer.decode` 函数将 "input_ids" 解码为文本形式进行打印。
    # 这个函数的目的是用于在控制台上展示一个无监督学习数据集的样例，包括输入文本。如果你有特定的问题或需要更多的解释，请提出。
    def print_unsupervised_dataset_example(example):
        print("input_ids:\n{}".format(example["input_ids"]))
        print("inputs:\n{}".format(tokenizer.decode(example["input_ids"], skip_special_tokens=False)))

    # 这段代码包含一个条件语句，根据 `stage` 的值选择性地对数据集进行预处理，并使用相应的函数和打印函数。以下是代码的主要逻辑：
    # 1. 条件语句：
    #    - 如果 `stage` 等于 "pt"，则选择仅包含 "prompt" 字段的示例，并使用 `preprocess_pretrain_dataset` 进行预处理，同时使用 `print_unsupervised_dataset_example` 打印样例。
    #    - 如果 `stage` 等于 "sft" 且 `training_args.predict_with_generate` 为 `False`，则选择仅包含 "prompt" 和 "response" 字段的示例，并使用 `preprocess_packed_supervised_dataset`（如果 `data_args.sft_packing` 为 `True`）或 `preprocess_supervised_dataset` 进行预处理，同时使用 `print_supervised_dataset_example` 打印样例。
    #    - 如果 `stage` 等于 "rm"，则选择包含 "prompt" 字段且 "response" 字段长度大于1的示例，并使用 `preprocess_pairwise_dataset` 进行预处理，同时使用 `print_pairwise_dataset_example` 打印样例。
    #    - 对于其他情况，选择仅包含 "prompt" 字段的示例，并使用 `preprocess_unsupervised_dataset` 进行预处理，同时使用 `print_unsupervised_dataset_example` 打印样例。

    if stage == "pt":
        dataset = dataset.filter(lambda example: example["prompt"])
        preprocess_func = preprocess_pretrain_dataset
        print_function = print_unsupervised_dataset_example
    elif stage == "sft" and not training_args.predict_with_generate:
        dataset = dataset.filter(lambda example: example["prompt"] and example["response"])
        preprocess_func = preprocess_packed_supervised_dataset if data_args.sft_packing else preprocess_supervised_dataset
        print_function = print_supervised_dataset_example
    elif stage == "rm":
        dataset = dataset.filter(lambda example: example["prompt"] and len(example["response"]) > 1)
        preprocess_func = preprocess_pairwise_dataset
        print_function = print_pairwise_dataset_example
    else:
        dataset = dataset.filter(lambda example: example["prompt"])
        preprocess_func = preprocess_unsupervised_dataset
        print_function = print_unsupervised_dataset_example

    with training_args.main_process_first(desc="dataset map pre-processing"):
        kwargs = {}
        if not data_args.streaming:
            kwargs = dict(
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on dataset"
            )

        # 2. 使用 `dataset.map` 进行预处理：
        #    - 根据条件语句中选择的 `preprocess_func` 函数对数据集进行映射。
        #    - 使用 `batched=True` 表示以批次处理数据集。
        #    - 使用 `remove_columns=column_names` 删除数据集中的指定列。
        # 3. 打印样例：
        #    - 使用 `print_function` 打印处理后数据集的第一个示例。
        # 4. 返回处理后的数据集。
        # 这个代码段的目的是根据不同的训练阶段（"pt"、"sft"、"rm"、其他）选择不同的预处理逻辑，并在控制台上打印处理后数据集的一个示例。如果你有特定的问题或需要更多的解释，请提出。
        dataset = dataset.map(
            preprocess_func,
            batched=True,            
            remove_columns=column_names,
            **kwargs
        )

        print_function(next(iter(dataset)))
        return dataset
