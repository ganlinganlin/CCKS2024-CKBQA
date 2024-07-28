
# 这段代码主要用于导入各种模块和库，以及定义一些辅助函数和类。以下是对代码的逐行解释：
# 1. **导入模块和库：**
#    - 导入了一系列基本的 Python 模块和库，包括操作系统模块 `os`、数学模块 `math`、PyTorch 模块 `torch`、`MethodType` 类型来创建方法类型，以及一些类型注解所需的模块。
#    - 导入了 Hugging Face Transformers 库的一些类和函数，如模型配置 (`AutoConfig`)、模型 (`AutoModelForCausalLM`)、分词器 (`AutoTokenizer`)，以及其他一些用于处理字节的配置类 (`BitsAndBytesConfig`) 和预训练模型的基类 (`PreTrainedModel`、`PreTrainedTokenizerBase`)。
#    - 导入了来自 Hugging Face Transformers 库中的 Llama 模型 (`modeling_llama`)。
#    - 导入了用于检查版本的函数 (`check_min_version`、`require_version`)。
#    - 导入了 `trl` 库中的 `AutoModelForCausalLMWithValueHead` 类。
#    - 尝试导入 Transformers 库的 `is_deepspeed_zero3_enabled` 函数，如果导入失败，则从深度学习库 (`transformers.deepspeed`) 中导入该函数。
#
# 2. **导入自定义模块和类：**
#    - 导入了自定义模块 `llmtuner.extras.logging` 中的 `reset_logging` 和 `get_logger` 函数，用于重置日志记录和获取日志记录器。
#    - 导入了自定义模块 `llmtuner.extras.misc` 中的 `count_parameters` 函数，用于计算模型参数数量。
#    - 导入了自定义模块 `llmtuner.extras.patches` 中的 Llama 补丁模块 `llama_patch as LlamaPatches`。
#    - 导入了自定义模块 `llmtuner.extras.save_and_load` 中的 `load_valuehead_params` 函数，用于加载 value head 参数。
#    - 导入了自定义模块 `llmtuner.hparams` 中的 `FinetuningArguments` 类，用于存储微调参数。
#    - 导入了自定义模块 `llmtuner.tuner.core.adapter` 中的 `init_adapter` 函数，用于初始化适配器。
#    - 导入了自定义模块 `llmtuner.tuner.core.utils` 中的 `prepare_model_for_training` 函数，用于准备模型进行训练。
# 3. **日志记录器的初始化：**
#    - 使用之前导入的 `get_logger` 函数创建了一个日志记录器，该记录器的名称为当前模块的名称。
# 这段代码主要是为后续的模型训练和调优准备工作。
import os
import math
import torch
from types import MethodType
from typing import TYPE_CHECKING, Literal, Optional, Tuple

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase
)
from transformers.models.llama import modeling_llama as LlamaModule
from transformers.utils import check_min_version
from transformers.utils.versions import require_version
from trl import AutoModelForCausalLMWithValueHead

try:
    from transformers.integrations import is_deepspeed_zero3_enabled
except ImportError: # https://github.com/huggingface/transformers/releases/tag/v4.33.1
    from transformers.deepspeed import is_deepspeed_zero3_enabled

from llmtuner.extras.logging import reset_logging, get_logger
from llmtuner.extras.misc import count_parameters
from llmtuner.extras.patches import llama_patch as LlamaPatches
from llmtuner.extras.save_and_load import load_valuehead_params
from llmtuner.hparams import FinetuningArguments
from llmtuner.tuner.core.adapter import init_adapter
from llmtuner.tuner.core.utils import prepare_model_for_training

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer
    from llmtuner.hparams import ModelArguments


logger = get_logger(__name__)


check_min_version("4.30.0")
require_version("datasets>=2.12.0", "To fix: pip install datasets>=2.12.0")
require_version("accelerate>=0.21.0", "To fix: pip install accelerate>=0.21.0")
require_version("peft>=0.4.0", "To fix: pip install peft>=0.4.0")
require_version("trl>=0.7.1", "To fix: pip install trl>=0.7.1")



# 这段代码定义了一个名为 `load_model_and_tokenizer` 的函数，该函数用于加载预训练模型和相应的分词器。以下是对代码的逐行解释：
# 1. **函数定义：**
#    - 定义了一个名为 `load_model_and_tokenizer` 的函数。
#    - 函数接受模型参数 (`model_args`)、微调参数 (`finetuning_args`)、可训练标志 (`is_trainable`) 和阶段标志 (`stage`) 作为参数。
#    - 返回一个包含预训练模型和分词器的元组 (`Tuple[PreTrainedModel, "PreTrainedTokenizer"]`)。
# 2. **检查是否可训练并存在检查点目录：**
#    - 如果模型不可训练 (`is_trainable` 为 False) 且在评估时未提供检查点目录 (`model_args.checkpoint_dir` 为 None)，则发出警告并将微调参数设置为 `FinetuningArguments(finetuning_type="none")`。
# 3. **配置参数设置：**
#    - 创建一个配置参数字典 (`config_kwargs`)，包括是否信任远程代码、缓存目录、模型版本、以及是否使用身份验证令牌等配置信息。
# 4. **加载分词器：**
#    - 使用 `AutoTokenizer.from_pretrained` 方法加载预训练分词器。
#    - 使用模型参数中的模型名称或路径 (`model_args.model_name_or_path`)，并传递其他配置参数。
#    - 可选择使用快速分词器 (`use_fast=model_args.use_fast_tokenizer`)。
#    - 配置填充方向为右填充 (`padding_side="right"`)，这是因为在使用 fp16 精度左填充张量进行训练可能导致溢出。
# 5. **选择要加载的模型路径：**
#    - 如果微调类型不是 "lora" 且提供了检查点目录 (`model_args.checkpoint_dir`)，则选择加载检查点目录的第一个模型路径。
#    - 否则，选择加载模型参数中的模型名称或路径 (`model_args.model_name_or_path`)。
# 6. **加载模型配置：**
#    - 使用 `AutoConfig.from_pretrained` 方法加载预训练模型的配置。
#    - 使用选择的模型路径 (`model_to_load`) 和其他配置参数。
# 函数的其余部分没有提供，但通常接下来的步骤将涉及加载预训练模型、进行适应性调整（如果需要）等。这个函数主要负责加载模型和分词器的初始化工作。
def load_model_and_tokenizer(
    model_args: "ModelArguments",
    finetuning_args: "FinetuningArguments",
    is_trainable: Optional[bool] = False,
    stage: Optional[Literal["pt", "sft", "rm", "ppo"]] = "sft"
) -> Tuple[PreTrainedModel, "PreTrainedTokenizer"]:
    r"""
    Loads pretrained model and tokenizer.

    Support both training and inference.
    """
    if (not is_trainable) and model_args.checkpoint_dir is None:
        logger.warning("Checkpoint is not found at evaluation, load the original model.")
        finetuning_args = FinetuningArguments(finetuning_type="none")

    config_kwargs = {
        "trust_remote_code": True,
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        use_fast=model_args.use_fast_tokenizer,
        padding_side="right", # training with left-padded tensors in fp16 precision may cause overflow
        **config_kwargs
    )

    if finetuning_args.finetuning_type != "lora" and model_args.checkpoint_dir is not None:
        model_to_load = model_args.checkpoint_dir[0]
    else:
        model_to_load = model_args.model_name_or_path

    config = AutoConfig.from_pretrained(model_to_load, **config_kwargs)


    # 这部分代码包含了一些模型和配置的修复、调整和设置步骤。以下是对代码的逐行解释：
    # 1. **修复 ChatGLM2 的分词器：**
    #    - 如果模型的类型是 "chatglm"，则使用 `MethodType` 将分词器的 `_pad` 方法设置为 `PreTrainedTokenizerBase._pad`。这可能是为了修复 ChatGLM2 模型的特定问题。
    # 2. **修复 Qwen 模型的配置：**
    #    - 如果模型的类型是 "qwen"，则根据 `model_args.compute_dtype` 设置配置中的浮点精度属性 (`fp16`、`bf16`、`fp32`)。这可能是为了修复 Qwen 模型的特定问题。
    # 3. **设置 RoPE 缩放：**
    #    - 如果指定了 RoPE 缩放 (`model_args.rope_scaling`)，则根据模型类型采取不同的操作。
    #    - 如果模型类型是 Qwen，根据模型是否可训练发出警告或设置一些配置项。
    #    - 如果模型类型是 LLaMA 或 Falcon，设置 RoPE 缩放的类型和因子，并记录相应的信息。
    #    - 如果模型不支持 RoPE 缩放，发出警告。
    # 这部分代码主要用于处理特定模型（ChatGLM2、Qwen、LLaMA、Falcon）的配置修复和设置，以确保模型在训练和推理时能够正确运行。
    # Fix tokenizer (for ChatGLM2)
    if getattr(config, "model_type", None) == "chatglm":
        tokenizer._pad = MethodType(PreTrainedTokenizerBase._pad, tokenizer)

    # Fix config (for Qwen)
    if getattr(config, "model_type", None) == "qwen":
        setattr(config, "fp16", model_args.compute_dtype == torch.float16)
        setattr(config, "bf16", model_args.compute_dtype == torch.bfloat16)
        setattr(config, "fp32", model_args.compute_dtype == torch.float32)

    # Set RoPE scaling
    if model_args.rope_scaling is not None:
        if hasattr(config, "use_dynamic_ntk"): # for Qwen models
            if is_trainable:
                logger.warning("Qwen model does not support RoPE scaling in training.")
            else:
                setattr(config, "use_dynamic_ntk", True)
                setattr(config, "use_logn_attn", True)
                logger.info("Using dynamic NTK scaling.")

        elif hasattr(config, "rope_scaling"): # for LLaMA and Falcon models
            require_version("transformers>=4.31.0", "RoPE scaling requires transformers>=4.31.0")
            if is_trainable:
                if model_args.rope_scaling == "dynamic":
                    logger.warning(
                        "Dynamic NTK may not work well with fine-tuning. "
                        "See: https://github.com/huggingface/transformers/pull/24653"
                    )

                current_max_length = getattr(config, "max_position_embeddings", None)
                if current_max_length and model_args.model_max_length > current_max_length:
                    scaling_factor = float(math.ceil(model_args.model_max_length / current_max_length))
                else:
                    logger.warning("Input length is smaller than max length. Consider increase input length.")
                    scaling_factor = 1.0
            else:
                scaling_factor = 2.0

            setattr(config, "rope_scaling", {"type": model_args.rope_scaling, "factor": scaling_factor})
            logger.info("Using {} scaling strategy and setting scaling factor to {}".format(
                model_args.rope_scaling, scaling_factor
            ))

        else:
            logger.warning("Current model does not support RoPE scaling.")


    # 这部分代码包含了一系列关于模型的配置、修复和设置。以下是对代码的逐行解释：
    # 1. **设置 FlashAttention-2：**
    #    - 如果指定了 `model_args.flash_attn`，则根据模型类型选择 FlashAttention-2 或 Shift Short Attention 进行设置。这可能是为了加速训练和推理。
    # Set FlashAttention-2
    if model_args.flash_attn:
        if getattr(config, "model_type", None) == "llama":
            LlamaModule.LlamaAttention = LlamaPatches.LlamaFlashAttention2
            LlamaModule.LlamaModel._prepare_decoder_attention_mask = (
                LlamaPatches._prepare_decoder_attention_mask
            )
            logger.info("Using FlashAttention-2 for faster training and inference.")
        else:
            logger.warning("Current model does not support FlashAttention-2.")
    elif is_trainable and model_args.shift_attn and getattr(config, "model_type", None) == "llama":
        LlamaModule.LlamaAttention = LlamaPatches.LlamaShiftShortAttention
        logger.warning("Using `--flash_attn` for faster training in large context length.")

    # 2. **量化配置（使用 bitsandbytes 库）：**
    #    - 如果指定了 `model_args.quantization_bit`，则根据不同的量化位数进行配置。支持 8 位和 4 位量化，且根据是否使用 DeepSpeed ZeRO-3 进行检查和设置。
    # Quantization configurations (using bitsandbytes library).
    is_mergeable = True
    if model_args.quantization_bit is not None:
        if is_deepspeed_zero3_enabled():
            raise ValueError("DeepSpeed ZeRO-3 is incompatible with quantization.")

        if model_args.quantization_bit == 8:
            require_version("bitsandbytes>=0.37.0", "To fix: pip install bitsandbytes>=0.37.0")
            config_kwargs["load_in_8bit"] = True
            config_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

        elif model_args.quantization_bit == 4:
            require_version("bitsandbytes>=0.39.0", "To fix: pip install bitsandbytes>=0.39.0")
            config_kwargs["load_in_4bit"] = True
            config_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=model_args.compute_dtype,
                bnb_4bit_use_double_quant=model_args.double_quantization,
                bnb_4bit_quant_type=model_args.quantization_type
            )

        is_mergeable = False
        config_kwargs["device_map"] = {"": int(os.environ.get("LOCAL_RANK", "0"))} if is_trainable else "auto"
        logger.info("Quantizing model to {} bit.".format(model_args.quantization_bit))

    # 3. **加载和准备预训练模型（无 valuehead）：**
    #    - 使用 `AutoModelForCausalLM.from_pretrained` 方法加载预训练模型。
    #    - 使用选择的模型路径 (`model_to_load`)、配置参数 (`config`)、计算数据类型 (`model_args.compute_dtype`) 和其他配置参数。
    # Load and prepare pre-trained models (without valuehead).
    model = AutoModelForCausalLM.from_pretrained(
        model_to_load,
        config=config,
        torch_dtype=model_args.compute_dtype,
        low_cpu_mem_usage=(not is_deepspeed_zero3_enabled()),
        **config_kwargs
    )

    # 4. **设置 shift short attention (S^2-Attn)：**
    #    - 如果模型是可训练的且指定了 `model_args.shift_attn`，并且模型类型是 "llama"，则设置 shift short attention 的比例，并记录相应的信息。
    # 5. **禁用自定义 generate 方法（用于 Qwen 和 Baichuan2）：**
    #    - 如果模型是 `PreTrainedModel` 类型且其 generate 方法不包含 "GenerationMixin"，则将其 generate 方法设置为 `PreTrainedModel.generate`。
    # 6. **修复 LM head（用于 ChatGLM2）：**
    #    - 如果模型类型是 "chatglm"，则将模型的 `lm_head` 属性设置为 `model.transformer.output_layer`。这可能是为了修复 ChatGLM2 模型的 LM head。
    # 这部分代码主要用于设置和修复模型的各种属性，以确保其在训练和推理时正常运行。
    # Set shift short attention (S^2-Attn)
    if is_trainable and model_args.shift_attn:
        if getattr(config, "model_type", None) == "llama":
            setattr(model, "shift_ratio", 0.25)
            logger.info("Using shift short attention proposed by LongLoRA.")
        else:
            logger.warning("Current model does not support shift short attention.")

    # Disable custom generate method (for Qwen and Baichuan2)
    if isinstance(model, PreTrainedModel) and "GenerationMixin" not in str(model.generate.__func__):
        model.generate = MethodType(PreTrainedModel.generate, model)

    # Fix LM head (for ChatGLM2)
    if getattr(config, "model_type", None) == "chatglm":
        setattr(model, "lm_head", model.transformer.output_layer)


    # 这部分代码主要完成以下几个步骤：
    # 1. **注册自动类以保存自定义代码文件：**
    #    - 通过检查配置 (`config`)、模型 (`model`) 和分词器 (`tokenizer`) 是否是预训练模型库的预训练配置、模型和分词器的实例，
    #     并且它们的 `auto_map` 属性中包含相应的自动类，将这些自动类注册到相应的类中。这可能涉及到为了保存自定义代码文件而注册这些类。
    # Register auto class to save the custom code files.
    if isinstance(config, PretrainedConfig) and "AutoConfig" in getattr(config, "auto_map", {}):
        config.__class__.register_for_auto_class()
    if isinstance(model, PreTrainedModel) and "AutoModelForCausalLM" in getattr(config, "auto_map", {}):
        model.__class__.register_for_auto_class()
    if isinstance(tokenizer, PreTrainedTokenizerBase) and "AutoTokenizer" in tokenizer.init_kwargs.get("auto_map", {}):
        tokenizer.__class__.register_for_auto_class()

    # 2. **初始化适配器：**
    #    - 如果模型是可训练的 (`is_trainable`)，则通过 `prepare_model_for_training` 函数初始化模型，其中可能包含一些与训练相关的设置。
    #    - 使用 `init_adapter` 函数初始化适配器。适配器通常用于微调模型。
    # Initialize adapters
    if is_trainable:
        model = prepare_model_for_training(model, model_args.layernorm_dtype, finetuning_args.finetuning_type)
    model = init_adapter(model, model_args, finetuning_args, is_trainable, is_mergeable)
    model = model.train() if is_trainable else model.eval()

    # 3. **准备模型用于 RLHF（Reward Learning from Human Feedback）：**
    #    - 如果当前阶段是 "rm"（评估奖励模型）或 "ppo"（使用 PPO 算法进行强化学习），则使用 `AutoModelForCausalLMWithValueHead.from_pretrained` 函数加载预训练模型，并设置一些属性，包括忽略保存时的某些键。
    #    - 如果阶段是 "rm" 并且存在检查点目录 (`model_args.checkpoint_dir`)，则加载检查点目录中最后一个包含值头的检查点，加载值头的权重到模型中。
    #    - 如果阶段是 "ppo"，则从指定路径 (`model_args.reward_model`) 加载奖励模型，并确保正确加载值头的参数。
    # Prepare model with valuehead for RLHF
    if stage == "rm" or stage == "ppo":
        model = AutoModelForCausalLMWithValueHead.from_pretrained(model)
        model._keys_to_ignore_on_save = None
        reset_logging()
        if stage == "rm" and model_args.checkpoint_dir is not None: # load valuehead weights to evaluate reward model
            logger.warning("Only the last checkpoint containing valuehead will be loaded.")
            if load_valuehead_params(model, model_args.checkpoint_dir[-1]):
                model.v_head.load_state_dict({
                    "summary.weight": getattr(model, "reward_head_weight"),
                    "summary.bias": getattr(model, "reward_head_bias")
                })

        if stage == "ppo": # load reward model
            logger.info("Load reward model from {}".format(model_args.reward_model))
            if getattr(model, "is_peft_model", False):
                model.pretrained_model.load_adapter(model_args.reward_model, "reward")
            assert load_valuehead_params(model, model_args.reward_model), "Reward model is not correctly loaded."

    # 4. **准备模型用于推理：**
    #    - 如果模型不可训练，则将模型的参数固定，使其不再可训练。
    #    - 如果指定了量化位数 (`model_args.quantization_bit`)，则将模型转移到指定的计算精度。
    #    - 计算可训练参数和总参数的数量，并记录日志。
    # 5. **返回模型和分词器：**
    #    - 返回最终配置好的模型和分词器。
    # 这段代码主要用于准备模型和相关组件，以便在后续的训练、评估或推理中使用。
    # Prepare model for inference
    if not is_trainable:
        model.requires_grad_(False) # fix all model params
        model = model.to(model_args.compute_dtype) if model_args.quantization_bit is None else model

    trainable_params, all_param = count_parameters(model)
    logger.info("trainable params: {:d} || all params: {:d} || trainable%: {:.4f}".format(
        trainable_params, all_param, 100 * trainable_params / all_param
    ))

    return model, tokenizer
