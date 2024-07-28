# 这段代码进行了一系列导入操作，涉及 PyTorch、类型提示、生成器、多线程、`transformers` 库等。以下是一些关键点的解释：
# 1. **`torch` 模块:**
#    - 导入 PyTorch 模块，用于深度学习任务。
# 2. **`typing` 模块:**
#    - 导入与类型提示相关的模块，如 `Any`、`Dict`、`Generator`、`List`、`Optional`、`Tuple`。
# 3. **`threading` 模块:**
#    - 导入了 `Thread` 类，用于创建线程。
# 4. **`transformers` 模块:**
#    - 导入了一些与 Hugging Face Transformers 相关的功能，如 `GenerationConfig`、`TextIteratorStreamer`。
# 5. **`llmtuner.extras.misc` 模块:**
#    - 导入了一些与额外功能相关的模块，如 `dispatch_model`、`get_logits_processor`。
# 6. **`llmtuner.extras.template` 模块:**
#    - 导入了 `get_template_and_fix_tokenizer` 函数。
# 7. **`llmtuner.tuner.core` 模块:**
#    - 导入了一些与调谐器核心相关的函数，如 `get_infer_args`、`load_model_and_tokenizer`。
# 这些导入涉及到深度学习、自然语言处理（NLP）以及模型调谐的一些关键概念。如果你对其中的某一部分有特定的问题，或者需要详细解释，请告诉我，我将尽力提供帮助！
import torch
from typing import Any, Dict, Generator, List, Optional, Tuple
from threading import Thread
from transformers import GenerationConfig, TextIteratorStreamer

from llmtuner.extras.misc import dispatch_model, get_logits_processor
from llmtuner.extras.template import get_template_and_fix_tokenizer
from llmtuner.tuner.core import get_infer_args, load_model_and_tokenizer


# 这是一个 `ChatModel` 类的定义，用于处理聊天模型的初始化和相关操作。以下是对这个类的关键点解释：
class ChatModel:
    # 1. **`__init__` 方法:**
    #    - 构造函数，用于初始化 `ChatModel` 实例。
    #    - 接受一个可选的参数 `args`，默认为 `None`。
    # 2. **参数解析:**
    #    - 使用 `get_infer_args(args)` 函数解析模型、数据和微调的参数。`get_infer_args` 可能用于获取推断时的参数配置。
    # 3. **模型和分词器加载:**
    #    - 使用 `load_model_and_tokenizer` 函数加载模型和分词器。
    #    - `self.model` 存储加载的模型，`self.tokenizer` 存储加载的分词器。
    # 4. **分词器设置:**
    #    - 设置分词器的 `padding_side` 为 "left"，表示在左侧进行填充。
    # 5. **模型分发:**
    #    - 使用 `dispatch_model` 函数，可能用于将模型分发到合适的设备（例如 GPU）。
    # 6. **模板加载:**
    #    - 使用 `get_template_and_fix_tokenizer` 函数加载模板，并校正分词器。
    # 7. **系统提示:**
    #    - 将系统提示存储在 `self.system_prompt` 中。
    # 这个类的初始化过程包括加载模型、分词器，设置分词器参数，加载模板，并存储系统提示。这是一个典型的用于处理聊天模型初始化的类。
    def __init__(self, args: Optional[Dict[str, Any]] = None) -> None:
        model_args, data_args, finetuning_args, self.generating_args = get_infer_args(args)
        self.model, self.tokenizer = load_model_and_tokenizer(model_args, finetuning_args)
        self.tokenizer.padding_side = "left"
        self.model = dispatch_model(self.model)
        self.template = get_template_and_fix_tokenizer(data_args.template, self.tokenizer)
        self.system_prompt = data_args.system_prompt


    # 这是 `ChatModel` 类中的一个方法 `process_args` 的定义，该方法用于处理聊天生成的参数。以下是这个方法的关键点解释：
    # 1. **参数:**
    #    - 接受 `query`、`history`、`system` 和其他任意关键字参数。
    # 2. **系统提示:**
    #    - 如果未提供系统提示 (`system` 参数为 `None`)，则使用类中存储的 `self.system_prompt`。
    def process_args(
        self,
        query: str,
        history: Optional[List[Tuple[str, str]]] = None,
        system: Optional[str] = None,
        **input_kwargs
    ) -> Tuple[Dict[str, Any], int]:
        system = system or self.system_prompt

        # 3. **模板编码:**
        #    - 使用 `self.template.encode_oneturn` 方法，通过给定的输入参数，编码成模型可接受的形式。
        # 4. **输入张量:**
        #    - 将编码后的 prompt 转换为 PyTorch 张量，并放置在模型所在的设备上。
        prompt, _ = self.template.encode_oneturn(
            tokenizer=self.tokenizer, query=query, resp="", history=history, system=system
        )
        input_ids = torch.tensor([prompt], device=self.model.device)
        prompt_length = len(input_ids[0])

        # 5. **处理生成参数:**
        #    - 处理与生成相关的参数，包括 `do_sample`、`temperature`、`top_p`、`top_k`、`repetition_penalty`、`max_length`、`max_new_tokens`。
        do_sample = input_kwargs.pop("do_sample", None)
        temperature = input_kwargs.pop("temperature", None)
        top_p = input_kwargs.pop("top_p", None)
        top_k = input_kwargs.pop("top_k", None)
        repetition_penalty = input_kwargs.pop("repetition_penalty", None)
        max_length = input_kwargs.pop("max_length", None)
        max_new_tokens = input_kwargs.pop("max_new_tokens", None)

        # 6. **生成参数的更新:**
        #    - 更新生成参数，确保其符合模型生成的要求。
        generating_args = self.generating_args.to_dict()
        generating_args.update(dict(
            do_sample=False,
            num_beams = generating_args["num_beams"],
            num_beam_groups = generating_args["num_beams"],
            diversity_penalty = 1.0,
            num_return_sequences=generating_args["num_beams"],
            temperature=temperature or generating_args["temperature"],
            top_p=top_p or generating_args["top_p"],
            top_k=top_k or generating_args["top_k"],
            repetition_penalty=repetition_penalty or generating_args["repetition_penalty"],
            eos_token_id=[self.tokenizer.eos_token_id] + self.tokenizer.additional_special_tokens_ids,
            pad_token_id=self.tokenizer.pad_token_id
        ))

        if max_length:
            generating_args.pop("max_new_tokens", None)
            generating_args["max_length"] = max_length

        if max_new_tokens:
            generating_args.pop("max_length", None)
            generating_args["max_new_tokens"] = max_new_tokens

        # 7. **构建生成参数字典:**
        #    - 将处理后的生成参数构建为字典。
        # 8. **构建生成的关键字参数:**
        #    - 将输入、生成配置和 logits 处理器构建为关键字参数字典。
        gen_kwargs = dict(
            inputs=input_ids,
            generation_config=GenerationConfig(**generating_args),
            logits_processor=get_logits_processor()
        )

        # 9. **返回值:**
        #    - 返回构建好的生成参数和 prompt 的长度。
        # 这个方法的目的是为了根据输入的聊天生成参数，准备好用于调用模型生成的参数。这样的设计使得可以在调用生成方法之前，对参数进行预处理和调整。
        return gen_kwargs, prompt_length

    # 这是 `ChatModel` 类中的一个方法 `chat_beam` 的定义，该方法使用束搜索（beam search）生成聊天响应。以下是这个方法的关键点解释：
    # 1. **`@torch.inference_mode()` 装饰器:**
    #    - 表明这个方法会在推理模式下运行，用于优化推理性能。
    # 2. **参数:**
    #    - 接受 `query`、`history`、`system` 和其他任意关键字参数。
    @torch.inference_mode()
    def chat_beam(
        self,
        query: str,
        history: Optional[List[Tuple[str, str]]] = None,
        system: Optional[str] = None,
        **input_kwargs
    ) -> Tuple[str, Tuple[int, int]]:
        # 3. **生成参数准备:**
        #    - 调用 `self.process_args` 方法，准备生成参数。
        # 4. **生成聊天响应:**
        #    - 使用模型的 `generate` 方法生成聊天响应。
        #    - `return_dict_in_generate=True` 表示生成的输出将以字典的形式返回，包括序列和分数。
        #    - 从生成输出中提取生成的序列和分数。
        # 5. **解码生成的序列:**
        #    - 使用分词器将生成的序列解码为文本。
        gen_kwargs, prompt_length = self.process_args(query, history, system, **input_kwargs)
        generation_output = self.model.generate(**gen_kwargs,return_dict_in_generate=True, output_scores=True)
        outputs = [g[prompt_length:] for g in generation_output['sequences'].tolist()]
        outputs_scores = [s for s in generation_output['sequences_scores'].tolist()]
        response = [self.tokenizer.decode(o, skip_special_tokens=True) for o in outputs]

        # 6. **响应评分:**
        #    - 将生成的响应和相应的分数存储在字典 `response_dict` 中。
        response_dict = {}
        for resp, score in zip(response, outputs_scores):
            if resp not in response_dict or score > response_dict[resp]:
                response_dict[resp] = score

        # 7. **排序和筛选:**
        #    - 将字典转换为元组列表，并按分数进行排序，以获取最高分的响应。
        #    - 注意：在这段代码中，使用了 `reverse=True` 表示降序排序。
        # 8. **返回值:**
        #    - 返回排序后的响应元组列表。
        # 这个方法的主要目的是使用束搜索生成聊天响应，并按照生成的分数进行排序。
        # 将字典转换为元组列表并按得分排序
        sorted_responses = sorted(response_dict.items(), key=lambda x: x[1], reverse=True)
        # response_length = len(outputs)
        return sorted_responses

    # 这是 `ChatModel` 类中的另一个方法 `chat` 的定义，该方法用于生成聊天响应。以下是这个方法的关键点解释：
    # 1. **`@torch.inference_mode()` 装饰器:**
    #    - 表明这个方法会在推理模式下运行，用于优化推理性能。
    @torch.inference_mode()
    def chat(
        self,
        query: str,
        history: Optional[List[Tuple[str, str]]] = None,
        system: Optional[str] = None,
        **input_kwargs
    ) -> Tuple[str, Tuple[int, int]]:
        # 2. **参数:**
        #    - 接受 `query`、`history`、`system` 和其他任意关键字参数。
        # 3. **生成参数准备:**
        #    - 调用 `self.process_args` 方法，准备生成参数。
        # 4. **生成聊天响应:**
        #    - 使用模型的 `generate` 方法生成聊天响应。
        # 5. **解码生成的序列:**
        #    - 使用分词器将生成的序列解码为文本。
        # 6. **响应长度计算:**
        #    - 计算生成的响应的长度。
        # 7. **返回值:**
        #    - 返回生成的聊天响应和一个元组，包含输入的 prompt 长度和生成的响应的长度。
        # 这个方法的主要目的是生成聊天响应，并返回生成的响应文本以及有关生成过程的一些信息，如输入 prompt 的长度和生成的响应的长度。
        gen_kwargs, prompt_length = self.process_args(query, history, system, **input_kwargs)
        generation_output = self.model.generate(**gen_kwargs)
        outputs = generation_output.tolist()[0][prompt_length:]
        response = self.tokenizer.decode(outputs, skip_special_tokens=True)
        response_length = len(outputs)
        return response, (prompt_length, response_length)

    # 这是 `ChatModel` 类中的另一个方法 `stream_chat` 的定义，该方法用于流式生成聊天响应。以下是这个方法的关键点解释：
    # 1. **`@torch.inference_mode()` 装饰器:**
    #    - 表明这个方法会在推理模式下运行，用于优化推理性能。
    @torch.inference_mode()
    def stream_chat(
        self,
        query: str,
        history: Optional[List[Tuple[str, str]]] = None,
        system: Optional[str] = None,
        **input_kwargs
    ) -> Generator[str, None, None]:
        # 2. **参数:**
        #    - 接受 `query`、`history`、`system` 和其他任意关键字参数。
        # 3. **生成参数准备:**
        #    - 调用 `self.process_args` 方法，准备生成参数。
        # 4. **创建 `TextIteratorStreamer` 实例:**
        #    - 创建一个 `TextIteratorStreamer` 实例，该实例用于处理流式生成的文本。
        # 5. **设置 `streamer` 参数:**
        #    - 将 `streamer` 设置为生成参数的一部分。
        gen_kwargs, _ = self.process_args(query, history, system, **input_kwargs)
        streamer = TextIteratorStreamer(self.tokenizer, timeout=60.0, skip_prompt=True, skip_special_tokens=True)
        gen_kwargs["streamer"] = streamer

        # 6. **创建线程并启动:**
        #    - 创建一个线程，目标是调用模型的 `generate` 方法。
        #    - 在线程中启动生成过程。
        thread = Thread(target=self.model.generate, kwargs=gen_kwargs)
        thread.start()

        # 7. **使用生成器 `yield from`:**
        #    - 使用生成器的 `yield from` 语法，将 `streamer` 的输出流式地返回。
        # 这个方法的主要目的是通过启动一个线程，使得生成聊天响应的过程变为异步且流式的。
        yield from streamer
