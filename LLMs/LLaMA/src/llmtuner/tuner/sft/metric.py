
# 这段代码主要是导入了一些 Python 模块和库，并定义了一些数据类（data class）。以下是对代码的逐行解释：
# 1. **导入模块和库：**
#    - 导入 NumPy 库，并使用别名 `np`。
#    - 使用 `dataclass` 装饰器，引入了 Python 3.7+ 提供的 `dataclass` 特性，用于创建简单的数据类。
#    - 导入 `TYPE_CHECKING`，用于在类型检查时导入额外的类型。
#    - 导入一些类型注解所需的模块。
#    - 导入中文分词库 `jieba`，用于中文文本处理。
#    - 导入 Rouge 库，用于计算中文文本的 ROUGE 分数。
#    - 导入 NLTK 库的 `sentence_bleu` 函数，用于计算 BLEU 分数。
#    - 导入 `SmoothingFunction`，用于进行 BLEU 分数的平滑处理。
#    - 从自定义模块 `llmtuner.extras.constants` 导入 `IGNORE_INDEX` 常量。
#    - 如果在类型检查上下文中，导入 `PreTrainedTokenizer` 类型。
# 2. **中文分词处理：**
#    - 导入了中文分词库 `jieba`，用于对中文文本进行分词处理。
# 3. **ROUGE 和 BLEU 相关的函数导入：**
#    - 导入中文文本的 ROUGE 分数计算工具 `Rouge`。
#    - 导入 BLEU 分数计算相关的函数 `sentence_bleu` 和 `SmoothingFunction`。
# 4. **类型注解导入：**
#    - 如果在类型检查的上下文中，导入了 `PreTrainedTokenizer` 类型。
# 这段代码主要用于导入在后续代码中可能用到的各种库、模块和类型。
import numpy as np
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Sequence, Tuple, Union

import jieba
from rouge_chinese import Rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

from llmtuner.extras.constants import IGNORE_INDEX

if TYPE_CHECKING:
    from transformers.tokenization_utils import PreTrainedTokenizer




# 这段代码定义了一个名为 `ComputeMetrics` 的数据类，该类用于计算评估指标。以下是对代码的逐行解释：
# 1. **数据类定义：**
#    - 使用 `@dataclass` 装饰器定义了一个数据类 `ComputeMetrics`。
#    - 类中有一个成员变量 `tokenizer`，类型为 `"PreTrainedTokenizer"`。
@dataclass
class ComputeMetrics:
    r"""
    Wraps the tokenizer into metric functions, used in Seq2SeqPeftTrainer.
    """

    tokenizer: "PreTrainedTokenizer"


    # 2. **`__call__` 方法定义：**
    #    - 定义了 `__call__` 方法，使得对象可以像函数一样被调用。
    #    - 方法接受一个参数 `eval_preds`，类型为 `Sequence[Union[np.ndarray, Tuple[np.ndarray]]]`，返回一个字典，键为评估指标的名称，值为对应的指标值。
    #    - 初始化了一个 `score_dict` 字典，用于存储不同指标的得分。
    # 3. **模型预测后处理：**
    #    - 对模型预测的标签进行后处理，将特殊标记（如 `IGNORE_INDEX`）替换为分词器的填充标记。
    # 4. **解码预测结果和标签：**
    #    - 使用分词器的 `batch_decode` 方法解码模型预测结果和标签。
    def __call__(self, eval_preds: Sequence[Union[np.ndarray, Tuple[np.ndarray]]]) -> Dict[str, float]:
        r"""
        Uses the model predictions to compute metrics.
        """
        preds, labels = eval_preds
        score_dict = {"rouge-1": [], "rouge-2": [], "rouge-l": [], "bleu-4": []}

        preds = np.where(preds != IGNORE_INDEX, preds, self.tokenizer.pad_token_id)
        labels = np.where(labels != IGNORE_INDEX, labels, self.tokenizer.pad_token_id)

        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)


        # 5. **计算 ROUGE 和 BLEU 分数：**
        #    - 使用中文分词库 `jieba` 对预测结果和标签进行分词。
        #    - 使用 ROUGE 库计算 ROUGE 分数。
        # 6. **记录分数到 `score_dict`：**
        #    - 将计算得到的 ROUGE 和 BLEU 分数记录到 `score_dict` 中。
        for pred, label in zip(decoded_preds, decoded_labels):
            hypothesis = list(jieba.cut(pred))
            reference = list(jieba.cut(label))

            if len(" ".join(hypothesis).split()) == 0 or len(" ".join(reference).split()) == 0:
                result = {"rouge-1": {"f": 0.0}, "rouge-2": {"f": 0.0}, "rouge-l": {"f": 0.0}}
            else:
                rouge = Rouge()
                scores = rouge.get_scores(" ".join(hypothesis), " ".join(reference))
                result = scores[0]

            for k, v in result.items():
                score_dict[k].append(round(v["f"] * 100, 4))

            bleu_score = sentence_bleu([list(label)], list(pred), smoothing_function=SmoothingFunction().method3)
            score_dict["bleu-4"].append(round(bleu_score * 100, 4))

        # 7. **计算平均分数并返回：**
        #    - 计算每个指标的平均分数，并将结果返回。
        # 这个类的作用是将模型的预测结果转换为一组评估指标，其中包括 ROUGE-1、ROUGE-2、ROUGE-L 和 BLEU-4。这些指标可以用于评估 Seq2Seq 模型在生成文本任务中的性能。
        return {k: float(np.mean(v)) for k, v in score_dict.items()}
