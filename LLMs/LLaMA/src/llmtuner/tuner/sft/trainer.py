
# 这段代码主要是导入了一些 Python 模块和库，并定义了一些用于模型训练的辅助功能。以下是对代码的逐行解释：
# 1. **导入模块和库：**
#    - 导入了一些基本的 Python 模块和库，包括操作系统模块 `os`、JSON 模块 `json`、PyTorch 模块 `torch`、NumPy 模块 `numpy` 以及 PyTorch 中的神经网络模块 `torch.nn`。
#    - 导入了类型注解所需的 `TYPE_CHECKING`，以及一些来自 Hugging Face Transformers 库的模块和类。
# 2. **导入自定义模块：**
#    - 从自定义模块 `llmtuner.extras.constants` 导入了常量 `IGNORE_INDEX`。
#    - 从自定义模块 `llmtuner.extras.logging` 导入了日志记录器函数 `get_logger`。
# 3. **类型检查导入：**
#    - 如果在类型检查的上下文中，导入了 `PredictionOutput` 类型。
# 4. **创建日志记录器：**
#    - 使用之前导入的 `get_logger` 函数创建了一个日志记录器，该记录器的名称为当前模块的名称。
# 这段代码主要是为后续的模型训练和日志记录做一些准备工作。
import os
import json
import torch
import numpy as np
import torch.nn as nn
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union
from transformers import Seq2SeqTrainer

from llmtuner.extras.constants import IGNORE_INDEX
from llmtuner.extras.logging import get_logger

if TYPE_CHECKING:
    from transformers.trainer import PredictionOutput

logger = get_logger(__name__)



class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    r"""
    Inherits PeftTrainer to compute generative metrics such as BLEU and ROUGE.
    """
# 这段代码定义了一个名为 `CustomSeq2SeqTrainer` 的类，它继承自 `Seq2SeqTrainer` 类。该类主要用于在模型预测时执行一些自定义的操作，如移除生成的标记中的提示部分。以下是对代码的逐行解释：
# 1. **类定义：**
#    - 定义了一个名为 `CustomSeq2SeqTrainer` 的类，它继承自 `Seq2SeqTrainer`。
# 2. **`prediction_step` 方法定义：**
#    - 定义了 `prediction_step` 方法，用于执行模型预测的一步操作。
#    - 方法接受模型 (`model`)、输入 (`inputs`)、是否仅计算预测损失 (`prediction_loss_only`) 和可忽略的键列表 (`ignore_keys`) 作为参数。
#    - 提供了注释，说明该方法的目的是在生成的标记中移除提示部分。
#    - 提示可以在子类中覆盖此方法以注入自定义行为。
    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        r"""
        Removes the prompt part in the generated tokens.

        Subclass and override to inject custom behavior.
        """
        # 3. ** 移除生成标记中的提示部分： **
        # - 如果模型配置参数
        # `predict_with_generate`
        # 为真，则执行以下操作：
        # - 检查分词器的填充方向是否为左侧，并且填充标记是否存在。
        # - 获取输入中提示部分和标签部分的长度。
        # - 如果提示部分长度大于标签部分长度，则将标签部分填充到与提示部分相同的长度。
        # - 如果标签部分长度大于提示部分长度，则将输入部分填充到与标签部分相同的长度，并在需要时更新其他输入张量的填充。
        if self.args.predict_with_generate:
            assert self.tokenizer.padding_side == "left", "This method only accepts left-padded tensor."
            assert self.tokenizer.pad_token_id is not None, "Pad token is required."
            prompt_len, label_len = inputs["input_ids"].size(-1), inputs["labels"].size(-1)
            if prompt_len > label_len:
                inputs["labels"] = self._pad_tensors_to_target_len(inputs["labels"], inputs["input_ids"])
            if label_len > prompt_len:
                inputs["input_ids"] = self._pad_tensors_to_target_len(inputs["input_ids"], inputs["labels"])
                if "attention_mask" in inputs:
                    inputs["attention_mask"] = self._pad_tensors_to_target_len(
                        inputs["attention_mask"], inputs["labels"], pad_token_id=0
                    )
                if "position_ids" in inputs:
                    inputs["position_ids"] = self._pad_tensors_to_target_len(
                        inputs["position_ids"], inputs["labels"], pad_token_id=0
                    )

        # 4. **调用父类的 `prediction_step` 方法：**
        #    - 调用父类 `Seq2SeqTrainer` 的 `prediction_step` 方法，获取损失、生成的标记和标签。
        # 5. **对生成的标记进行后处理：**
        #    - 如果生成的标记存在且模型配置参数 `predict_with_generate` 为真，则执行以下操作：
        #      - 将生成的标记的前部分（长度为提示部分和标签部分中较大的长度）设置为填充标记。
        #      - 对生成的标记进行重新整理（contiguous）操作。
        loss, generated_tokens, labels = super().prediction_step(
            model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
        )
        if generated_tokens is not None and self.args.predict_with_generate:
            generated_tokens[:, :max(prompt_len, label_len)] = self.tokenizer.pad_token_id
            generated_tokens = generated_tokens.contiguous()


        # 6. **返回结果：**
        #    - 返回损失、生成的标记和标签的元组。
        # 这个类主要用于扩展 `Seq2SeqTrainer` 类，在模型预测时执行一些自定义的操作，例如移除生成的标记中的提示部分。
        return loss, generated_tokens, labels


    # 这段代码定义了一个名为 `_pad_tensors_to_target_len` 的方法，用于将输入的张量填充到与目标张量相同的长度。以下是对代码的逐行解释：
    # 1. **方法定义：**
    #    - 定义了一个名为 `_pad_tensors_to_target_len` 的方法，用于将输入的张量填充到与目标张量相同的长度。
    #    - 方法接受源张量 (`src_tensor`)、目标张量 (`tgt_tensor`) 和填充标记的可选值 (`pad_token_id`)。
    # 2. **获取填充标记：**
    #    - 如果提供了填充标记 (`pad_token_id`)，则使用提供的值，否则使用分词器的填充标记。
    # 3. **创建填充后的张量：**
    #    - 使用填充标记乘以与目标张量形状相同的全 1 张量，创建一个填充后的张量。
    # 4. **进行左填充操作：**
    #    - 将源张量的内容按左对齐方式填充到填充后的张量的右侧。
    # 5. **返回填充后的张量：**
    #    - 将填充后的张量进行重新整理（contiguous），并返回结果。
    # 这个方法的主要作用是将源张量填充到与目标张量相同的长度，并且采用左对齐方式。
    def _pad_tensors_to_target_len(
        self,
        src_tensor: torch.Tensor,
        tgt_tensor: torch.Tensor,
        pad_token_id: Optional[int] = None
    ) -> torch.Tensor:
        r"""
        Pads the tensor to the same length as the target tensor.
        """
        pad_token_id = pad_token_id if pad_token_id is not None else self.tokenizer.pad_token_id
        padded_tensor = pad_token_id * torch.ones_like(tgt_tensor)
        padded_tensor[:, -src_tensor.shape[-1]:] = src_tensor # adopt left-padding
        return padded_tensor.contiguous() # in contiguous memory



    # 这段代码定义了一个名为 `save_predictions` 的方法，用于保存模型的预测结果到输出目录 (`output_dir`) 中的一个 JSON 文件。以下是对代码的逐行解释：
    # 1. **方法定义：**
    #    - 定义了一个名为 `save_predictions` 的方法，用于保存模型的预测结果到输出目录 (`output_dir`) 中的一个 JSON 文件。
    #    - 这是一个自定义的行为，不包含在 `Seq2SeqTrainer` 中。
    def save_predictions(
        self,
        predict_results: "PredictionOutput"
    ) -> None:
        r"""
        Saves model predictions to `output_dir`.

        A custom behavior that not contained in Seq2SeqTrainer.
        """
        # 2. **检查进程是否为主进程：**
        #    - 如果当前进程不是全局进程组的第一个进程（主进程），则直接返回，只有主进程执行后续的保存操作。
        # 3. **设置输出文件路径：**
        #    - 构建输出文件路径，文件名为 "generated_predictions.jsonl"，路径为输出目录 (`output_dir`)。
        #    - 使用日志记录器 (`logger`) 输出保存的信息。
        # 4. **处理预测结果：**
        #    - 对预测结果和标签进行处理，将特殊标记（如 `IGNORE_INDEX`）替换为分词器的填充标记。
        # 5. **解码预测结果和标签：**
        #    - 使用分词器的 `batch_decode` 方法解码模型的预测结果和标签，同时跳过特殊标记并清理标记化空格。
        if not self.is_world_process_zero():
            return

        output_prediction_file = os.path.join(self.args.output_dir, "generated_predictions.jsonl")
        logger.info(f"Saving prediction results to {output_prediction_file}")

        preds = np.where(predict_results.predictions != IGNORE_INDEX, predict_results.predictions, self.tokenizer.pad_token_id)
        labels = np.where(predict_results.label_ids != IGNORE_INDEX, predict_results.label_ids, self.tokenizer.pad_token_id)

        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True, clean_up_tokenization_spaces=True)

        # 6. **保存到文件：**
        #    - 打开输出文件，以 UTF-8 编码写入。
        #    - 遍历解码后的预测结果和标签，将它们组织成 JSON 对象，并使用 JSON 序列化写入文件。
        #    - 每个 JSON 对象占一行，整体写入文件。
        # 这个方法的主要作用是将模型的预测结果和相应的标签保存到一个 JSON 文件中，以便后续分析和评估。
        with open(output_prediction_file, "w", encoding="utf-8") as writer:
            res: List[str] = []
            for pred, label in zip(decoded_preds, decoded_labels):
                res.append(json.dumps({"label": label, "predict": pred}, ensure_ascii=False))
            writer.write("\n".join(res))
