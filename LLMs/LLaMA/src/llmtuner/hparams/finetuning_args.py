# 这里导入了 `json` 模块和一些用于类型提示和数据类的模块，包括 `Literal`、`Optional` 和 `dataclass`。这些模块通常在处理 JSON 数据和定义数据类时使用。
# - `json` 模块用于处理 JSON 数据的编码和解码。
# - `Literal` 用于指定某些字段的确切取值，提供更强的类型提示。
# - `Optional` 用于指定字段是可选的。
# - `dataclass` 用于定义数据类，可以更方便地定义和操作类似结构的数据。
# 这些模块的使用表明代码可能涉及到处理配置文件、参数和数据类的定义。
import json
from typing import Literal, Optional
from dataclasses import asdict, dataclass, field

# 这里定义了用于微调（fine-tuning）的参数的数据类 `FinetuningArguments`。以下是一些参数及其用途的简要说明：
# - `finetuning_type`：指定微调的方法，可以是 "lora"、"freeze"、"full" 或 "none"。
# - `num_layer_trainable`：对于部分参数（freeze）微调，指定可训练的层数。
# - `name_module_trainable`：指定部分参数（freeze）微调时可训练的模块名称，有一些预定义的选择。
# - `lora_rank`：LoRA微调的固有维度。
# - `lora_alpha`：LoRA微调的尺度因子（类似于学习率）。
# - `lora_dropout`：LoRA微调的Dropout率。
# - `lora_target`：指定应用LoRA的目标模块名称，可以是多个以逗号分隔的字符串。
# - `additional_target`：指定除了LoRA层之外的其他模块，这些模块将作为可训练并保存在最终检查点中。
# - `resume_lora_training`：指定是否从上次的LoRA权重中恢复训练或在合并它们后创建新权重。
# - `ppo_score_norm`：在PPO训练中使用得分归一化。
# - `dpo_beta`：DPO损失的 beta 参数。
# 这些参数的设置可以影响微调的方式和效果。
@dataclass
class FinetuningArguments:
    r"""
    Arguments pertaining to which techniques we are going to fine-tuning with.
    """
    finetuning_type: Optional[Literal["lora", "freeze", "full", "none"]] = field(
        default="lora",
        metadata={"help": "Which fine-tuning method to use."}
    )
    num_layer_trainable: Optional[int] = field(
        default=3,
        metadata={"help": "Number of trainable layers for partial-parameter (freeze) fine-tuning."}
    )
    name_module_trainable: Optional[Literal["mlp", "self_attn", "self_attention"]] = field(
        default="mlp",
        metadata={"help": "Name of trainable modules for partial-parameter (freeze) fine-tuning. \
                  LLaMA choices: [\"mlp\", \"self_attn\"], \
                  BLOOM & Falcon & ChatGLM2 choices: [\"mlp\", \"self_attention\"], \
                  Qwen choices: [\"mlp\", \"attn\"], \
                  Phi-1.5 choices: [\"mlp\", \"mixer\"], \
                  LLaMA-2, Baichuan, InternLM, XVERSE choices: the same as LLaMA."}
    )
    lora_rank: Optional[int] = field(
        default=8,
        metadata={"help": "The intrinsic dimension for LoRA fine-tuning."}
    )
    lora_alpha: Optional[float] = field(
        default=32.0,
        metadata={"help": "The scale factor for LoRA fine-tuning (similar with the learning rate)."}
    )
    lora_dropout: Optional[float] = field(
        default=0.1,
        metadata={"help": "Dropout rate for the LoRA fine-tuning."}
    )
    lora_target: Optional[str] = field(
        default=None,
        metadata={"help": "Name(s) of target modules to apply LoRA. Use commas to separate multiple modules. \
                  LLaMA choices: [\"q_proj\", \"k_proj\", \"v_proj\", \"o_proj\", \"gate_proj\", \"up_proj\", \"down_proj\"], \
                  BLOOM & Falcon & ChatGLM2 choices: [\"query_key_value\", \"self_attention.dense\", \"mlp.dense\"], \
                  Baichuan choices: [\"W_pack\", \"o_proj\", \"gate_proj\", \"up_proj\", \"down_proj\"], \
                  Qwen choices: [\"c_attn\", \"attn.c_proj\", \"w1\", \"w2\", \"mlp.c_proj\"], \
                  Phi-1.5 choices: [\"Wqkv\", \"out_proj\", \"fc1\", \"fc2\"], \
                  LLaMA-2, InternLM, XVERSE choices: the same as LLaMA."}
    )
    additional_target: Optional[str] = field(
        default=None,
        metadata={"help": "Name(s) of modules apart from LoRA layers to be set as trainable and saved in the final checkpoint."}
    )
    resume_lora_training: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether to resume training from the last LoRA weights or create new weights after merging them."}
    )
    ppo_score_norm: Optional[bool] = field(
        default=False,
        metadata={"help": "Use score normalization in PPO Training."}
    )
    dpo_beta: Optional[float] = field(
        default=0.1,
        metadata={"help": "The beta parameter for the DPO loss."}
    )

    # `__post_init__` 方法是 `dataclass` 提供的一个特殊方法，用于在创建对象后执行一些额外的初始化步骤。在这里，它用于处理 `lora_target` 和 `additional_target` 参数，确保它们都是列表形式的字符串。
    # `save_to_json` 方法用于将参数的内容保存为 JSON 格式的文件。它使用了 `json.dumps` 函数，将对象转换为字典，然后以漂亮的格式写入 JSON 文件。
    def __post_init__(self):
        if isinstance(self.lora_target, str): # support custom target modules/layers of LoRA
            self.lora_target = [target.strip() for target in self.lora_target.split(",")]

        if isinstance(self.additional_target, str):
            self.additional_target = [target.strip() for target in self.additional_target.split(",")]

        assert self.finetuning_type in ["lora", "freeze", "full", "none"], "Invalid fine-tuning method."

    def save_to_json(self, json_path: str):
        r"""Saves the content of this instance in JSON format inside `json_path`."""
        json_string = json.dumps(asdict(self), indent=2, sort_keys=True) + "\n"
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(json_string)

    # `load_from_json` 方法是一个类方法，用于从 JSON 文件中加载参数并创建一个类的实例。
    #     它打开 JSON 文件，读取其中的文本内容，然后使用 `json.loads` 将 JSON 格式的字符串转换为 Python 字典。最后，使用 `cls(**json.loads(text))` 创建并返回类的实例。
    # 这个方法假定 JSON 文件中的键与类的初始化参数相匹配，以便正确地传递给类的 `__init__` 方法。
    @classmethod
    def load_from_json(cls, json_path: str):
        r"""Creates an instance from the content of `json_path`."""
        with open(json_path, "r", encoding="utf-8") as f:
            text = f.read()
        return cls(**json.loads(text))
