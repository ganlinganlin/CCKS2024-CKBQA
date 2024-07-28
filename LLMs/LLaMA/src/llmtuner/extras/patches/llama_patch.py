
# 这段代码导入了一些 PyTorch 相关的库和模块，并引入了一些 FlashAttention 相关的功能。以下是代码的主要逻辑：
# 1. 导入模块：
#    - 导入了 `math`、`torch`、`torch.nn` 等 PyTorch 相关的库。
#    - 导入了 `Optional` 和 `Tuple` 用于类型提示。
#    - 导入了 FlashAttention 的一些功能，例如 `flash_attn_func`、`flash_attn_varlen_func`，以及相关的模块。
# 2. 导入自定义模块：
#    - 导入了 `LlamaAttention`、`apply_rotary_pos_emb`、`repeat_kv` 等自定义的模块，这些模块似乎与 Llama 模型的注意力机制相关。
# 3. 异常处理：
#    - 尝试导入 `flash_attn` 模块，如果导入失败，输出一条信息表明 FlashAttention-2 未安装。此处使用 `try` 和 `except ImportError` 来处理导入异常，
#     确保即使 FlashAttention-2 未安装，代码仍然能够执行。
# 如果你对 FlashAttention 框架或 Llama 模型有特定的问题，或者希望深入了解代码的某个方面，请提出。
import math
import torch
import torch.nn as nn
from typing import Optional, Tuple
from transformers.utils import logging
from transformers.models.llama.modeling_llama import LlamaAttention, apply_rotary_pos_emb, repeat_kv

try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func # type: ignore
    from flash_attn.bert_padding import pad_input, unpad_input # type: ignore
except ImportError:
    print("FlashAttention-2 is not installed, ignore this if you are not using FlashAttention.")


logger = logging.get_logger(__name__)

# 这段代码定义了一个名为 `LlamaShiftShortAttention` 的类，继承自 `LlamaAttention`。这个类覆盖了 `forward` 方法，实现了一种修改过的自注意力机制。
# 主要的修改包括：
# 1. 将输入的 `hidden_states` 分别投影为查询（query）、键（key）和值（value）的状态。
# 2. 对查询、键和值的状态进行维度变换，使其符合自注意力机制的计算需求。
# 3. 引入旋转位置嵌入（rotary position embedding）对查询和键的状态进行位置嵌入。
# 4. 如果存在先前的键值（`past_key_value`），则将当前的键值与先前的键值拼接在一起。
# 5. 如果定义了 `num_key_value_groups`，则对键和值进行扩展，以支持分组操作。
# 6. 如果定义了 `shift_ratio` 且在训练阶段，进行一种特殊的位置偏移操作，通过滚动操作实现。
# 7. 计算注意力权重，考虑了注意力遮罩（attention_mask）。
# 8. 将注意力权重与值相乘，得到最终的自注意力输出。
# 9. 如果设置了 `output_attentions`，则将注意力权重一并返回。
# 这个类的设计主要用于实现一种特殊形式的自注意力机制，其中引入了位置嵌入和位置偏移的概念。如果你有特定的问题或需要更多的解释，请提出。
# Modified from: https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py
class LlamaShiftShortAttention(LlamaAttention):

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        # 10. 通过 `self.q_proj`、`self.k_proj` 和 `self.v_proj` 对输入的 `hidden_states` 进行线性映射，得到查询（query）、键（key）和值（value）的状态。
        # 11. 使用 `view` 方法对查询、键和值的状态进行维度变换，以满足自注意力机制的计算要求。
        # 12. 计算旋转位置嵌入（rotary position embedding）的 `cos` 和 `sin`，并应用于查询和键的状态。
        # 13. 如果存在先前的键值（`past_key_value`），则更新键和值的序列长度，以包括先前的键值的序列长度。
        # 14. 调用 `apply_rotary_pos_emb` 函数，将旋转位置嵌入应用于查询和键的状态，考虑位置编码（`position_ids`）。
        # 15. 代码截断，未完全展示。在这之后，代码执行一系列操作，包括处理键值的拼接、分组、位置偏移、计算注意力权重等步骤。最终，返回自注意力的输出、注意力权重和过去的键值。
        # 这个类的设计主要是为了实现一种特殊的自注意力机制，其中引入了旋转位置嵌入和位置偏移的概念。这些变化使得模型在处理序列数据时能够更好地捕捉序列中的信息。
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]

        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        # 继续阐述代码：
        # 16. 如果存在先前的键值（`past_key_value`），则在维度2上拼接当前计算得到的键值和先前的键值，以便在自注意力中重复使用这些键值。
        # 17. 根据 `use_cache` 的值，更新 `past_key_value` 变量。
        # 18. 如果定义了 `num_key_value_groups`，则对键和值进行扩展，以支持分组操作。这可以通过 `repeat_kv` 函数实现。
        # 19. 如果定义了 `shift_ratio` 且在训练阶段，进行一种特殊的位置偏移操作，通过滚动（`roll`）操作实现。这会在每个头的一半中进行滚动，以改变头之间的相对位置。
        # 20. 计算注意力权重，使用查询、键的转置和除以缩放因子进行计算。
        # 这部分代码主要是在自注意力计算中进行一系列的处理，包括对过去的键值的重用、分组扩展、位置偏移等操作。这些操作的引入可能旨在增强模型对序列中不同部分的关注。
        if past_key_value is not None: # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        if getattr(self, "num_key_value_groups"):
            key_states = repeat_kv(key_states, self.num_key_value_groups)
            value_states = repeat_kv(value_states, self.num_key_value_groups)

        if getattr(self, "shift_ratio", None) and self.training: # shift
            group_size = int(q_len * getattr(self, "shift_ratio"))
            if q_len % group_size > 0:
                raise ValueError("q_len {} should be divisible by group size {}.".format(q_len, group_size))
            num_group = q_len // group_size
            for state in (query_states, key_states, value_states):
                state = state.transpose(1, 2) # output: (bsz, seq_len, n_heads, head_dim)
                state[:, :, self.num_heads//2:] = state[:, :, self.num_heads//2:].roll(-group_size//2, dims=1)
                state = state.reshape(bsz * num_group, group_size, self.num_heads, self.head_dim).transpose(1, 2)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        # 21. 检查计算得到的注意力权重的形状，确保其符合预期的形状 `(bsz, self.num_heads, q_len, kv_seq_len)`。如果形状不符合，抛出 `ValueError`。
        # 22. 如果定义了注意力遮罩（`attention_mask`），检查其形状是否符合预期。注意力遮罩的形状应为 `(bsz, 1, q_len, kv_seq_len)`，否则抛出 `ValueError`。
        # 23. 如果存在注意力遮罩，将注意力权重与遮罩相加。这是为了在注意力计算中考虑遮罩的影响。
        # 24. 将注意力权重进行 softmax 操作，将其转换为概率分布。为了确保精度，将其转换为 `torch.float32` 类型。
        # 25. 使用注意力权重与值进行加权求和，得到最终的自注意力输出。
        # 26. 检查计算得到的自注意力输出的形状，确保其符合预期的形状 `(bsz, self.num_heads, q_len, self.head_dim)`。如果形状不符合，抛出 `ValueError`。
        # 27. 对自注意力输出进行维度变换，以符合后续的计算需求。这涉及到将头的维度移动到最后的维度。
        # 28. 至此，该自注意力机制的前向计算完成，返回计算得到的自注意力输出。
        # 这部分代码主要完成了注意力权重的处理，包括形状的检查、遮罩的考虑、softmax 操作以及最终的自注意力输出的计算。如果有具体问题或需要更多解释，请提问。
        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        # 29. 如果定义了 `shift_ratio` 且在训练阶段，进行注意力输出的逆偏移操作。这是为了将之前应用的位置偏移操作逆转，使得最终的输出与原始序列位置一致。
        # 30. 对注意力输出进行维度变换，以符合后续的计算需求。这涉及到将头的维度移动到最后的维度。
        # 31. 将注意力输出传递给输出投影层 `self.o_proj`，以获得最终的自注意力输出。
        # 32. 如果不需要输出注意力权重，将 `attn_weights` 设置为 `None`。
        # 33. 返回最终的自注意力输出、注意力权重以及可能的过去的键值。
        # 这部分代码主要完成了自注意力输出的后处理，包括逆偏移操作、维度变换以及最终的输出投影。如果有具体问题或需要更多解释，请提问。
        if getattr(self, "shift_ratio", None) and self.training: # shift back
            attn_output.reshape(bsz, q_len, self.num_heads, self.head_dim)
            attn_output[:, :, self.num_heads//2:] = attn_output[:, :, self.num_heads//2:].roll(group_size//2, dims=1)

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


# 这是一个修改版的自注意力层 `LlamaFlashAttention2`，主要用于支持 FlashAttention2。下面是对该层的前向计算的解释：
# 1. 确保不会输出注意力权重，将 `output_attentions` 参数强制设为 `False`。
# 2. 获取输入 `hidden_states` 的形状信息，即 batch size (`bsz`)、序列长度 (`q_len`) 以及隐藏单元维度 (`_`)。
# 3. 对输入进行线性变换，得到查询 (`query_states`)、键 (`key_states`) 和值 (`value_states`) 的表示。
# 4. 为适应 FlashAttention 的输入要求，对查询、键和值的表示进行维度变换，将其形状调整为 `(bsz, seq_len, n_heads, head_dim)`。
#   同时，对维度进行转置操作，以匹配 FlashAttention 的输入要求。
# 5. 获取键值序列的长度 `kv_seq_len`，并在需要的情况下（如果存在 `past_key_value`）进行调整。
# 6. 利用旋转位置嵌入，对查询和键的表示进行位置嵌入的应用。
# 7. 如果存在过去的键值，将当前计算得到的键和值与过去的键值进行拼接。
# 8. 如果需要使用缓存（`use_cache=True`），将当前计算得到的键和值存储为新的过去键值。
# 这部分代码主要完成了 FlashAttention2 自注意力机制的前向计算，包括输入处理、位置嵌入、过去键值的处理等。如果有具体问题或需要更多解释，请提问。
class LlamaFlashAttention2(LlamaAttention):

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        # LlamaFlashAttention2 attention does not support output_attentions
        output_attentions = False

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # FlashAttention requires the input to have the shape (bsz, seq_len, n_heads, head_dim)
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]

        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None: # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        # 这段代码主要进行了如下操作：
        # 1. 如果输入的数据类型是 `torch.float32`，则将查询、键和值的张量类型转换为 `torch.float16`，通过 `to` 方法实现。这个转换可能是为了节省内存或加速计算，但同时也可能导致精度损失。发出了一条警告，提醒用户可能存在的精度问题。
        # 2. 如果存在 `num_key_value_groups` 属性，对键和值进行扩展。`repeat_kv` 函数是一个模型自定义的函数，根据输入的键和值张量，对其维度进行重复扩展。
        # 3. 对查询、键和值的张量进行维度变换，将其形状调整为 `(bsz, n_heads, seq_len, head_dim)` 的形式。这样的形状对于后续计算是更加方便的。
        # 4. 如果存在 `shift_ratio` 并且处于训练模式下，进行注意力的平移操作。对查询、键和值的张量进行平移，其中平移的大小由 `group_size` 决定，`group_size` 的计算基于 `shift_ratio` 和查询序列的长度。
        # 5. 将进行了维度变换和平移操作的张量进行形状调整，以便后续的注意力计算。
        # 总体来说，这段代码主要是为了准备输入数据，包括数据类型的转换、维度的调整、扩展等操作，以便后续的注意力计算。
        # cast to half precision
        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            logger.warning_once("The input hidden states seems to be silently casted in float32.")
            query_states = query_states.to(torch.float16)
            key_states = key_states.to(torch.float16)
            value_states = value_states.to(torch.float16)

        if getattr(self, "num_key_value_groups"):
            key_states = repeat_kv(key_states, self.num_key_value_groups)
            value_states = repeat_kv(value_states, self.num_key_value_groups)

        query_states = query_states.transpose(1, 2) # (bsz, seq_len, n_heads, head_dim)
        key_states = key_states.transpose(1, 2) # (bsz, seq_len, n_heads, head_dim)
        value_states = value_states.transpose(1, 2) # (bsz, seq_len, n_heads, head_dim)

        if getattr(self, "shift_ratio", None) and self.training: # shift
            group_size = int(q_len * getattr(self, "shift_ratio"))
            if q_len % group_size > 0:
                raise ValueError("q_len {} should be divisible by group size {}.".format(q_len, group_size))
            num_group = q_len // group_size
            for state in (query_states, key_states, value_states):
                state[:, :, self.num_heads//2:] = state[:, :, self.num_heads//2:].roll(-group_size//2, dims=1)
                state = state.reshape(bsz * num_group, group_size, self.num_heads, self.head_dim)

        # 这部分代码主要涉及 FlashAttention2 模型在存在注意力掩码 (`attention_mask`) 的情况下的处理过程。具体步骤如下：
        # 1. 如果存在注意力掩码，通过 `unpad_input` 函数去除填充（padding）部分。这里使用了 `flash_attn_varlen_func` 函数，该函数是用于处理变长序列的 FlashAttention 计算。传递给该函数的参数包括未填充的查询、键和值，以及相关的长度信息（`cu_seqlens_q` 和 `cu_seqlens_k` 表示当前序列长度，`max_seqlen_q` 和 `max_seqlen_k` 表示最大序列长度）等。
        # 2. 如果不存在注意力掩码，直接使用 `flash_attn_func` 函数进行 FlashAttention 计算，该函数用于处理无填充的序列。
        # 3. 如果存在 `shift_ratio` 并且处于训练模式下，进行注意力的平移回退操作，与前面的注意力平移相对应。
        # 4. 将得到的注意力输出结果进行形状调整，并传递给输出投影层进行处理。
        # 5. 如果不需要输出注意力权重，将 `attn_weights` 设置为 `None`。
        # 这部分代码完成了 FlashAttention2 在考虑填充的情况下的后续计算和处理，确保了模型能够处理变长序列以及注意力平移的操作。如果有具体问题或需要更多解释，请提问。
        if attention_mask is not None:
            logger.warning_once("Padded sequences are less efficient in FlashAttention.")
            batch_size = query_states.shape[0]
            # -q_len: assumes left padding
            unpadded_q, indices_q, cu_seqlens_q, max_seqlen_q = unpad_input(query_states, attention_mask[:, -q_len:])
            unpadded_k, _, cu_seqlens_k, max_seqlen_k = unpad_input(key_states, attention_mask)
            unpadded_v, _, _, _ = unpad_input(value_states, attention_mask)
            attn_output_unpad = flash_attn_varlen_func(
                unpadded_q,
                unpadded_k,
                unpadded_v,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                dropout_p=0.0,
                softmax_scale=None,
                causal=True,
            )
            attn_output = pad_input(attn_output_unpad, indices_q, batch_size, q_len)
        else:
            attn_output = flash_attn_func(
                query_states, key_states, value_states, 0.0, softmax_scale=None, causal=True
            )

        if getattr(self, "shift_ratio", None) and self.training: # shift back
            attn_output.reshape(bsz, q_len, self.num_heads, self.head_dim)
            attn_output[:, :, self.num_heads//2:] = attn_output[:, :, self.num_heads//2:].roll(group_size//2, dims=1)

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


# 这段代码主要是在 LlamaModel 中的一个方法 `_prepare_decoder_attention_mask` 中，对注意力掩码进行预处理的操作。这个方法的作用是准备解码器的注意力掩码。
# 具体而言，该方法接受以下参数：
# - `attention_mask`：表示输入序列的注意力掩码张量。
# - `input_shape`：表示输入序列的形状张量。
# - `inputs_embeds`：表示嵌入输入序列的张量。
# - `past_key_values_length`：表示过去键值的长度。
# 该方法首先检查注意力掩码是否存在且是否全为真。如果是，表示所有的位置都是有效的，可以直接返回 `None`，这样在训练时使用完整的样本时可以使用更快的调用方式。
# 如果注意力掩码存在但不全为真，表示存在无效的位置，该方法会返回原始的注意力掩码张量。
# 这样的设计可能是因为 FlashAttention 需要一个布尔类型的填充掩码（padding_mask），因此在这里进行了一些特定的处理。
#     在返回的掩码中，如果所有位置都是有效的，可以返回 `None`，否则返回原始的注意力掩码。这种预处理可能是为了满足 FlashAttention 的输入要求。
# Disable the transformation of the attention mask in LlamaModel as flash attention
# takes a boolean padding_mask. Fills in the past kv length for use in forward.
def _prepare_decoder_attention_mask(
    self,
    attention_mask: torch.Tensor,
    input_shape: torch.Tensor,
    inputs_embeds: torch.Tensor,
    past_key_values_length: int
) -> torch.Tensor:
    if attention_mask is not None and torch.all(attention_mask):
        return None  # This uses the faster call when training with full samples

    return attention_mask
