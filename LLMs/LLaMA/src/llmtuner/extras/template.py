
# 这段代码导入了一些与计算文本标记（tokens）数量相关的模块和函数，以及一些用于数据类的模块。以下是每个导入语句的解释：
# 1. **`import tiktoken`：**
#    - 导入了 `tiktoken` 模块，这是一个用于计算文本标记数量的库。
# 2. **`from dataclasses import dataclass`：**
#    - 从 Python 标准库的 `dataclasses` 模块中导入 `dataclass` 装饰器，用于创建数据类。
# 3. **`from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union`：**
#    - 从 Python 标准库的 `typing` 模块中导入一些用于类型提示的工具。
# 4. **`from llmtuner.extras.logging import get_logger`：**
#    - 从自定义的 `llmtuner.extras.logging` 模块中导入 `get_logger` 函数。这可能是一个用于获取日志记录器的函数。
# 5. **`if TYPE_CHECKING:` 和 `from transformers import PreTrainedTokenizer`：**
#    - 如果在类型检查模式下，导入了 Hugging Face Transformers 库的 `PreTrainedTokenizer` 类。
# 这些导入语句显示了一些用于文本标记数量计算、数据类和日志记录的模块和函数。如果你对其中的某一部分有特定的问题，或者需要更多的解释，请随时告诉我！
import tiktoken
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

from llmtuner.extras.logging import get_logger

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer

logger = get_logger(__name__)


# 这是一个使用 `dataclass` 装饰器定义的名为 `Template` 的数据类。`dataclass` 是 Python 3.7 及更高版本提供的一个功能，用于轻松创建不可变的数据类。以下是 `Template` 类的关键点解释：
# 1. **属性:**
#    - `prefix`: 前缀，表示模板的前缀部分，可以是字符串列表或包含键值对的字典列表。
#    - `prompt`: 提示，表示模板的提示部分，可以是字符串列表或包含键值对的字典列表。
#    - `system`: 系统，表示系统提示的字符串。
#    - `sep`: 分隔符，表示模板的分隔符部分，可以是字符串列表或包含键值对的字典列表。
#    - `stop_words`: 停用词，表示在生成文本时应该被忽略的词语列表。
#    - `use_history`: 布尔值，表示是否使用对话历史信息。
#    - `efficient_eos`: 布尔值，表示是否使用有效的 EOS（End of Sentence）标记。
# 2. **`@dataclass` 装饰器:**
#    - 通过使用 `@dataclass` 装饰器，可以自动为类生成 `__init__`、`__repr__` 和其他特殊方法，简化了数据类的创建。
# 这个数据类的目的是存储模板的各个部分，例如前缀、提示、系统信息等。通过使用数据类，可以方便地创建和管理模板的实例。
@dataclass
class Template:

    prefix: List[Union[str, Dict[str, str]]]
    prompt: List[Union[str, Dict[str, str]]]
    system: str
    sep: List[Union[str, Dict[str, str]]]
    stop_words: List[str]
    use_history: bool
    efficient_eos: bool

    # 这是一个名为 `encode_oneturn` 的方法，它接收一个预训练的分词器 (`tokenizer`)，用户的查询 (`query`)，回答 (`resp`)，对话历史 (`history`) 和系统信息 (`system`)，
    #     然后返回一个包含两个部分的元组：一个代表提示（prompt）的标记 ID 列表，另一个代表响应（response）的标记 ID 列表。
    # 以下是方法的关键点解释：
    # 1. **参数:**
    #    - `tokenizer`: 预训练的分词器，用于将文本转换为标记 ID。
    #    - `query`: 用户的查询文本。
    #    - `resp`: 模型的响应文本。
    #    - `history`: 对话历史，是一个可选的包含元组的列表，每个元组包含用户查询和模型响应。
    #    - `system`: 系统信息，表示系统提示的字符串。
    # 2. **调用 `_format` 方法:**
    #    - 使用 `_format` 方法对输入进行格式化，以获取格式化后的系统信息和历史信息。
    # 3. **调用 `_encode` 方法:**
    #    - 使用 `_encode` 方法对格式化后的系统信息和历史信息进行编码，以获取编码后的标记 ID 对。
    # 4. **构建提示和响应的标记 ID 列表:**
    #    - 将编码后的标记 ID 对按照查询和响应的顺序拼接成一个列表。
    # 5. **返回结果:**
    #    - 返回一个元组，包含两个部分：代表提示的标记 ID 列表和代表响应的标记 ID 列表。
    # 这个方法的主要目的是将用户的查询、模型的响应、对话历史和系统信息转换为模型输入的标记 ID 列表。
    def encode_oneturn(
        self,
        tokenizer: "PreTrainedTokenizer",
        query: str,
        resp: str,
        history: Optional[List[Tuple[str, str]]] = None,
        system: Optional[str] = None
    ) -> Tuple[List[int], List[int]]:
        r"""
        Returns a single pair of token ids representing prompt and response respectively.
        """
        system, history = self._format(query, resp, history, system)
        encoded_pairs = self._encode(tokenizer, system, history)
        prompt_ids = []
        for query_ids, resp_ids in encoded_pairs[:-1]:
            prompt_ids = prompt_ids + query_ids + resp_ids
        prompt_ids, answer_ids = prompt_ids + encoded_pairs[-1][0], encoded_pairs[-1][1]
        return prompt_ids, answer_ids


    # 这是一个名为 `encode_multiturn` 的方法，它与之前的 `encode_oneturn` 方法类似，但返回的是一个列表，其中包含多个元组，每个元组代表一个对话轮次的标记 ID 对。以下是方法的关键点解释：
    # 1. **参数:**
    #    - `tokenizer`: 预训练的分词器，用于将文本转换为标记 ID。
    #    - `query`: 用户的查询文本。
    #    - `resp`: 模型的响应文本。
    #    - `history`: 对话历史，是一个可选的包含元组的列表，每个元组包含用户查询和模型响应。
    #    - `system`: 系统信息，表示系统提示的字符串。
    # 2. **调用 `_format` 方法:**
    #    - 使用 `_format` 方法对输入进行格式化，以获取格式化后的系统信息和历史信息。
    # 3. **调用 `_encode` 方法:**
    #    - 使用 `_encode` 方法对格式化后的系统信息和历史信息进行编码，以获取编码后的标记 ID 对。
    # 4. **返回结果:**
    #    - 返回一个列表，其中每个元素是一个元组，代表一个对话轮次的标记 ID 对。
    # 这个方法的目的是将多轮对话的用户查询、模型的响应、对话历史和系统信息转换为模型输入的标记 ID 列表。
    def encode_multiturn(
        self,
        tokenizer: "PreTrainedTokenizer",
        query: str,
        resp: str,
        history: Optional[List[Tuple[str, str]]] = None,
        system: Optional[str] = None
    ) -> List[Tuple[List[int], List[int]]]:
        r"""
        Returns multiple pairs of token ids representing prompts and responses respectively.
        """
        system, history = self._format(query, resp, history, system)
        encoded_pairs = self._encode(tokenizer, system, history)
        return encoded_pairs

    # 这是一个名为 `_format` 的方法，它用于将输入对齐到标准格式。以下是方法的关键点解释：
    # 1. **参数:**
    #    - `query`: 用户的查询文本。
    #    - `resp`: 模型的响应文本。
    #    - `history`: 对话历史，是一个可选的包含元组的列表，每个元组包含用户查询和模型响应。
    #    - `system`: 系统信息，表示系统提示的字符串。
    # 2. **处理系统信息和对话历史:**
    #    - 如果提供了系统信息 (`system`)，则使用提供的系统信息；否则，使用类属性 `self.system` 中的系统信息。
    #    - 如果启用了使用历史 (`self.use_history`)，并且提供了对话历史 (`history`)，则将其保留；否则，创建一个空的对话历史列表。
    #    - 将用户查询和模型响应作为新的对话历史的一部分添加到对话历史列表中。
    # 3. **返回结果:**
    #    - 返回一个元组，包含两个部分：系统信息 (`system`) 和对话历史列表 (`history`)。
    # 该方法的目的是确保输入的一致性，并将它们对齐到一个标准的格式，以便后续的处理和编码。
    def _format(
        self,
        query: str,
        resp: str,
        history: Optional[List[Tuple[str, str]]] = None,
        system: Optional[str] = None
    ) -> Tuple[str, List[Tuple[str, str]]]:
        r"""
        Aligns inputs to the standard format.
        """
        system = system or self.system # use system if provided
        history = history if (history and self.use_history) else []
        history = history + [(query, resp)]
        return system, history

    # 这是一个名为 `_get_special_ids` 的方法，用于获取特殊标记的标记 ID 列表。以下是方法的关键点解释：
    # 1. **参数:**
    #    - `tokenizer`: 预训练的分词器，用于将特殊标记转换为标记 ID。
    # 2. **获取 BOS（Beginning of Sequence）和 EOS（End of Sequence）标记 ID:**
    #    - 如果分词器支持 BOS 标记且已启用添加 BOS 标记，获取 BOS 标记的标记 ID 列表；否则，使用空列表表示。
    #    - 获取 EOS 标记的标记 ID 列表。
    # 3. **返回结果:**
    #    - 返回一个元组，包含两个部分：BOS 标记的标记 ID 列表 (`bos_ids`) 和 EOS 标记的标记 ID 列表 (`eos_ids`)。
    # 该方法的目的是获取特殊标记（BOS 和 EOS）的标记 ID 列表，以便在后续的编码过程中使用。如果你对其中的某一部分有特定的问题，或者需要更多的解释，请随时告诉我！
    def _get_special_ids(
        self,
        tokenizer: "PreTrainedTokenizer"
    ) -> Tuple[List[int], List[int]]:
        if tokenizer.bos_token_id is not None and getattr(tokenizer, "add_bos_token", True):
            bos_ids = [tokenizer.bos_token_id]
        else: # baichuan, qwen and gpt2 models have no bos token
            bos_ids = []

        if tokenizer.eos_token_id is None:
            raise ValueError("EOS token is required.")

        if self.efficient_eos: # used in baichuan, qwen, chatglm, etc.
            eos_ids = []
        else:
            eos_ids = [tokenizer.eos_token_id]

        return bos_ids, eos_ids

    # 这是一个名为 `_encode` 的方法，用于将格式化的输入编码为 token ID 对。以下是方法的关键点解释：
    # 1. **参数:**
    #    - `tokenizer`: 预训练的分词器，用于将文本转换为 token ID。
    #    - `system`: 系统信息的字符串。
    #    - `history`: 对话历史列表，包含元组，每个元组包含用户查询和模型响应。
    # 2. **获取特殊标记的标记 ID 列表:**
    #    - 调用 `_get_special_ids` 方法获取 BOS（Beginning of Sequence）和 EOS（End of Sequence）标记的标记 ID 列表。
    # 3. **循环遍历对话历史:**
    #    - 对于每一轮对话，根据轮次的不同生成不同的 token ID 对。
    #      - 在第一轮（`turn_idx == 0`）：
    #        - 获取前缀（`prefix`）的标记 ID 列表，包括 BOS 标记、前缀标记和分隔标记。
    #        - 如果前缀标记非空，则添加 BOS 标记、前缀标记和分隔标记；否则，只添加 BOS 标记。
    #      - 在后续轮次：
    #        - 获取分隔标记的标记 ID 列表和 BOS 标记的标记 ID 列表。
    # 4. **将输入文本转换为标记 ID:**
    #    - 调用 `_convert_inputs_to_ids` 方法将前缀、查询和响应文本分别转换为标记 ID 列表。
    # 5. **构建编码对:**
    #    - 将前缀、查询、分隔标记和响应的标记 ID 列表合并成编码对，并添加到 `encoded_pairs` 列表中。
    # 6. **返回结果:**
    #    - 返回一个列表，每个元素是一个元组，包含两个部分：查询的编码（`prefix_ids + query_ids`）和响应的编码（`resp_ids + eos_ids`）。
    # 该方法的目的是根据对话历史和系统信息生成格式化的输入，然后将其转换为 token ID 对。如果你对其中的某一部分有特定的问题，或者需要更多的解释，请随时告诉我！
    def _encode(
        self,
        tokenizer: "PreTrainedTokenizer",
        system: str,
        history: List[Tuple[str, str]]
    ) -> List[Tuple[List[int], List[int]]]:
        r"""
        Encodes formatted inputs to pairs of token ids.
        Turn 0: bos + prefix + sep + query    resp + eos
        Turn t: sep + bos + query             resp + eos
        """
        bos_ids, eos_ids = self._get_special_ids(tokenizer)
        sep_ids = self._convert_inputs_to_ids(tokenizer, context=self.sep)
        encoded_pairs = []
        for turn_idx, (query, resp) in enumerate(history):
            if turn_idx == 0:
                prefix_ids = self._convert_inputs_to_ids(tokenizer, context=self.prefix, system=system)
                if len(prefix_ids) != 0: # has prefix
                    prefix_ids = bos_ids + prefix_ids + sep_ids
                else:
                    prefix_ids = bos_ids
            else:
                prefix_ids = sep_ids + bos_ids

            query_ids = self._convert_inputs_to_ids(tokenizer, context=self.prompt, query=query, idx=str(turn_idx))
            resp_ids = self._convert_inputs_to_ids(tokenizer, context=[resp])
            encoded_pairs.append((prefix_ids + query_ids, resp_ids + eos_ids))
        return encoded_pairs

    # 这是一个名为 `_convert_inputs_to_ids` 的私有方法，用于将上下文（context）转换为 token IDs。以下是方法的关键点解释：
    # 1. **参数:**
    #    - `tokenizer`: 预训练的分词器，用于将文本转换为 token ID。
    #    - `context`: 上下文信息的列表，包含字符串或字典。
    #    - `system`: 系统信息的字符串（可选）。
    #    - `query`: 查询信息的字符串（可选）。
    #    - `idx`: 索引信息的字符串（可选）。
    # 2. **处理特殊标记:**
    #    - 根据分词器的类型，设置不同的参数，以允许或禁用特殊标记。对于 tiktoken 分词器，允许所有特殊标记。
    # 3. **循环遍历上下文:**
    #    - 对于上下文中的每个元素：
    #      - 如果元素是字符串：
    #        - 替换字符串中的特殊标记，如 `{{system}}`、`{{query}}` 和 `{{idx}}`。
    #        - 使用分词器将元素编码为 token ID，并将结果添加到 `token_ids` 列表中。
    #      - 如果元素是字典：
    #        - 获取字典中 "token" 键对应的标记，并将其转换为标记 ID，然后将结果添加到 `token_ids` 列表中。
    #      - 如果元素的类型既不是字符串也不是字典，则引发 ValueError。
    # 4. **返回结果:**
    #    - 返回一个包含所有 token IDs 的列表。
    # 该方法的主要目的是处理上下文中的字符串和字典元素，并将它们转换为相应的 token IDs。如果有任何进一步的疑问或需要更多解释，请随时告诉我！
    def _convert_inputs_to_ids(
        self,
        tokenizer: "PreTrainedTokenizer",
        context: List[Union[str, Dict[str, str]]],
        system: Optional[str] = None,
        query: Optional[str] = None,
        idx: Optional[str] = None
    ) -> List[int]:
        r"""
        Converts context to token ids.
        """
        if isinstance(getattr(tokenizer, "tokenizer", None), tiktoken.Encoding): # for tiktoken tokenizer (Qwen)
            kwargs = dict(allowed_special="all")
        else:
            kwargs = dict(add_special_tokens=False)

        token_ids = []
        for elem in context:
            if isinstance(elem, str):
                elem = elem.replace("{{system}}", system, 1) if system is not None else elem
                elem = elem.replace("{{query}}", query, 1) if query is not None else elem
                elem = elem.replace("{{idx}}", idx, 1) if idx is not None else elem
                if len(elem) != 0:
                    token_ids = token_ids + tokenizer.encode(elem, **kwargs)
            elif isinstance(elem, dict):
                token_ids = token_ids + [tokenizer.convert_tokens_to_ids(elem.get("token"))]
            else:
                raise ValueError("Input must be string or dict[str, str], got {}".format(type(elem)))

        return token_ids


# 这是一个名为 `_encode` 的方法，属于 `Llama2Template` 类。它是用于将格式化的输入编码为 token ID 对的方法。以下是方法的关键点解释：
# 1. **参数:**
#    - `tokenizer`: 预训练的分词器，用于将文本转换为 token ID。
#    - `system`: 系统信息的字符串。
#    - `history`: 包含对话历史的列表，每个元素都是一个包含查询和响应的元组。
# 2. **获取特殊标记:**
#    - 调用 `_get_special_ids` 方法，获取开始（bos）和结束（eos）的特殊标记 ID。
# 3. **循环遍历历史:**
#    - 对于历史中的每个对话轮次：
#      - 如果是第一轮（`turn_idx == 0`）：
#        - 将前缀（prefix）的第一个元素中的 `{{system}}` 替换为系统信息（system）。
#        - 使用 `_convert_inputs_to_ids` 方法将查询（query）编码为 token ID，并将 bos 标记添加到前面。
#        - 使用 `_convert_inputs_to_ids` 方法将响应（resp）编码为 token ID，并将 eos 标记添加到后面。
#      - 如果不是第一轮：
#        - 使用 `_convert_inputs_to_ids` 方法将查询（query）编码为 token ID，并在前面添加 bos 标记。
#        - 使用 `_convert_inputs_to_ids` 方法将响应（resp）编码为 token ID，并在后面添加 eos 标记。
#      - 将编码后的查询和响应组成的元组添加到 `encoded_pairs` 列表中。
# 4. **返回结果:**
#    - 返回一个包含所有对话轮次中查询和响应的 token ID 对的列表。
# 该方法的实现主要是为 Llama2 模板设计的，根据第一轮是否存在分隔符，设置相应的编码方式。如果有其他问题或需要进一步解释，请随时告诉我！
@dataclass
class Llama2Template(Template):

    def _encode(
        self,
        tokenizer: "PreTrainedTokenizer",
        system: str,
        history: List[Tuple[str, str]]
    ) -> List[Tuple[List[int], List[int]]]:
        r"""
        Encodes formatted inputs to pairs of token ids.
        Turn 0: bos + prefix + query    resp + eos
        Turn t: bos + query             resp + eos
        """
        bos_ids, eos_ids = self._get_special_ids(tokenizer)
        encoded_pairs = []
        for turn_idx, (query, resp) in enumerate(history):
            if turn_idx == 0: # llama2 template has no sep_ids
                query = self.prefix[0].replace("{{system}}", system) + query
            query_ids = self._convert_inputs_to_ids(tokenizer, context=self.prompt, query=query)
            resp_ids = self._convert_inputs_to_ids(tokenizer, context=[resp])
            encoded_pairs.append((bos_ids + query_ids, resp_ids + eos_ids))
        return encoded_pairs


templates: Dict[str, Template] = {}

# 这是一个用于注册对话模板的函数 `register_template`，以及一个用于存储模板的字典 `templates`。以下是函数的关键点解释：
# 1. **`templates` 字典:**
#    - `templates` 是一个字典，用于存储注册的对话模板。键是模板的名称（`name`），值是对应的 `Template` 或 `Llama2Template` 对象。
# 2. **`register_template` 函数:**
#    - **参数:**
#      - `name`: 模板的名称。
#      - `prefix`: 包含前缀信息的列表，每个元素可以是字符串或包含标记信息的字典。
#      - `prompt`: 包含提示信息的列表，每个元素可以是字符串或包含标记信息的字典。
#      - `system`: 系统信息的字符串。
#      - `sep`: 包含分隔符信息的列表，每个元素可以是字符串或包含标记信息的字典。
#      - `stop_words`: 包含停用词的可选列表。
#      - `use_history`: 控制是否使用对话历史的可选布尔值，默认为 `True`。
#      - `efficient_eos`: 控制是否使用高效的 EOS 标记的可选布尔值，默认为 `False`。
#    - **根据模板名称选择类:**
#      - 如果模板名称中包含 "llama2"，则选择使用 `Llama2Template` 类；否则，选择使用通用的 `Template` 类。
#    - **创建模板对象:**
#      - 使用选择的类创建对应的模板对象，并使用给定的参数进行初始化。
#    - **将模板对象添加到字典:**
#      - 将模板对象添加到 `templates` 字典中，以模板名称为键。
# 该函数的主要作用是根据输入的参数创建对话模板对象，并将其注册到 `templates` 字典中。
def register_template(
    name: str,
    prefix: List[Union[str, Dict[str, str]]],
    prompt: List[Union[str, Dict[str, str]]],
    system: str,
    sep: List[Union[str, Dict[str, str]]],
    stop_words: Optional[List[str]] = [],
    use_history: Optional[bool] = True,
    efficient_eos: Optional[bool] = False
) -> None:
    template_class = Llama2Template if "llama2" in name else Template
    templates[name] = template_class(
        prefix=prefix,
        prompt=prompt,
        system=system,
        sep=sep,
        stop_words=stop_words,
        use_history=use_history,
        efficient_eos=efficient_eos
    )

# 这个函数的作用是根据给定的名称获取对应的对话模板，并修复与 tokenizer 相关的问题。以下是函数的关键点解释：
# 1. **EOS Token 和 PAD Token 的处理:**
#    - 如果 tokenizer 的 `eos_token_id` 为 `None`，则将 `eos_token` 设置为空字符串，并记录日志。
#    - 如果 tokenizer 的 `pad_token_id` 为 `None`，则将 `pad_token` 设置为与 `eos_token` 相同的值，并记录日志。
# 2. **获取模板对象:**
#    - 使用给定的名称从 `templates` 字典中获取对应的对话模板对象。
#    - 如果名称为 `None`，则返回 `None`。
# 3. **特殊标记的处理:**
#    - 将模板中的停用词添加为 tokenizer 的附加特殊标记，确保它们被包含在 tokenizer 的词汇表中。
# 4. **返回模板对象:**
#    - 返回获取到的对话模板对象。
# 该函数的目的是确保 tokenizer 的特殊标记和对话模板的一致性，并返回用于生成对话的正确模板对象。如果有其他问题或需要进一步解释，请随时告诉我！
def get_template_and_fix_tokenizer(
    name: str,
    tokenizer: "PreTrainedTokenizer"
) -> Template:
    if tokenizer.eos_token_id is None:
        tokenizer.eos_token = "<|endoftext|>"
        logger.info("Add eos token: {}".format(tokenizer.eos_token))

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Add pad token: {}".format(tokenizer.pad_token))

    if name is None:
        return None

    template = templates.get(name, None)
    assert template is not None, "Template {} does not exist.".format(name)
    tokenizer.add_special_tokens(
        dict(additional_special_tokens=template.stop_words),
        replace_additional_special_tokens=False
    )
    return template


# 这个 `register_template` 函数的目的是注册名为 "vanilla" 的对话模板。下面是注册过程中传递的参数说明：
# - **name:** "vanilla"
# - **prefix:** 空列表 `[]`
# - **prompt:** 包含一个字符串 "{{query}}" 的列表
# - **system:** 空字符串 `""`
# - **sep:** 空列表 `[]`
# - **use_history:** `False`
# 这个模板的结构是比较简单的，只包含一个 prompt，即 "{{query}}"。这里使用了 `{{query}}` 占位符，表示在模型生成对话时会将实际的用户查询（query）插入到这个位置。
# 使用 `register_template` 函数注册模板后，可以通过名称 "vanilla" 在其他地方获取并使用这个对话模板。如果有其他问题或需要进一步解释，请随时告诉我！
r"""
Supports language model inference without histories.
"""
register_template(
    name="vanilla",
    prefix=[],
    prompt=[
        "{{query}}"
    ],
    system="",
    sep=[],
    use_history=False
)


# 这个 "default" 模板是一个包含系统说明的基本对话模板。下面是注册过程中传递的参数说明：
# - **name:** "default"
# - **prefix:** 包含一个字符串 "{{system}}" 的列表
# - **prompt:** 包含一个字符串 "Human: {{query}}\nAssistant: " 的列表
# - **system:** 包含对模型系统的说明，描述了一个在用户和人工智能助手之间进行的对话。这个说明会出现在模型的 `{{system}}` 位置。
# - **sep:** 包含一个字符串 "\n" 的列表
# 在这个模板中，`{{query}}` 表示用户的查询，`{{system}}` 表示对话的系统说明。注册 "default" 模板后，你可以在其他地方通过名称 "default" 获取并使用这个对话模板。
# 如果你有其他问题或需要更多的解释，请随时告诉我！
r"""
Default template.
"""
register_template(
    name="default",
    prefix=[
        "{{system}}"
    ],
    prompt=[
        "Human: {{query}}\nAssistant: "
    ],
    system=(
        "A chat between a curious user and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the user's questions."
    ),
    sep=[
        "\n"
    ]
)

# 这个 "llama2" 模板支持 `https://huggingface.co/meta-llama/Llama-2-7b-chat-hf`，`https://huggingface.co/meta-llama/Llama-2-13b-chat-hf`，
#     以及 `https://huggingface.co/meta-llama/Llama-2-70b-chat-hf` 这三个预训练模型。下面是注册过程中传递的参数说明：
# - **name:** "llama2"
# - **prefix:** 包含一个字符串 "<<SYS>>\n{{system}}\n<</SYS>>\n\n" 的列表。其中 `<<SYS>>` 和 `<</SYS>>` 是用于包围系统说明的标记，而 `{{system}}` 会被替换为模型的系统说明。
# - **prompt:** 包含一个字符串 "[INST] {{query}} [/INST] " 的列表。其中 `{{query}}` 表示用户的查询，`[INST]` 和 `[/INST]` 用于包围用户的输入，表示它是指令。
# - **system:** 包含对模型系统的说明，描述了一个帮助、尊重和诚实的助手。在回答用户的问题时，要尽量提供帮助，同时保持安全。模型回答不应包含任何有害、不道德、种族主义、性别歧视、有害、危险或非法内容。
# - **sep:** 包含一个空列表。这意味着在对话中没有特殊的分隔符。
# 注册 "llama2" 模板后，你可以在其他地方通过名称 "llama2" 获取并使用这个对话模板。
r"""
Supports: https://huggingface.co/meta-llama/Llama-2-7b-chat-hf
          https://huggingface.co/meta-llama/Llama-2-13b-chat-hf
          https://huggingface.co/meta-llama/Llama-2-70b-chat-hf
"""
register_template(
    name="llama2",
    prefix=[
        "<<SYS>>\n{{system}}\n<</SYS>>\n\n"
    ],
    prompt=[
        "[INST] {{query}} [/INST] "
    ],
    system=(
        "You are a helpful, respectful and honest assistant. "
        "Always answer as helpfully as possible, while being safe.  "
        "Your answers should not include any harmful, unethical, "
        "racist, sexist, toxic, dangerous, or illegal content. "
        "Please ensure that your responses are socially unbiased and positive in nature.\n\n"
        "If a question does not make any sense, or is not factually coherent, "
        "explain why instead of answering something not correct. "
        "If you don't know the answer to a question, please don't share false information."
    ),
    sep=[]
)


# 这个 "llama2_zh" 模板支持 [`https://github.com/ymcui/Chinese-LLaMA-Alpaca-2`](https://github.com/ymcui/Chinese-LLaMA-Alpaca-2) 和
#     [`https://huggingface.co/ziqingyang/chinese-alpaca-2-7b`](https://huggingface.co/ziqingyang/chinese-alpaca-2-7b) 这两个预训练模型。下面是注册过程中传递的参数说明：
# - **name:** "llama2_zh"
# - **prefix:** 包含一个字符串 "<<SYS>>\n{{system}}\n<</SYS>>\n\n" 的列表。其中 `<<SYS>>` 和 `<</SYS>>` 是用于包围系统说明的标记，而 `{{system}}` 会被替换为模型的系统说明。
# - **prompt:** 包含一个字符串 "[INST] {{query}} [/INST] " 的列表。其中 `{{query}}` 表示用户的查询，`[INST]` 和 `[/INST]` 用于包围用户的输入，表示它是指令。
# - **system:** 包含对模型系统的说明，描述了一个乐于助人的助手。在回答用户的问题时，助手的目标是提供帮助。
# - **sep:** 包含一个空列表。这意味着在对话中没有特殊的分隔符。
# 注册 "llama2_zh" 模板后，你可以在其他地方通过名称 "llama2_zh" 获取并使用这个对话模板。
# 如果你有其他问题或需要更多的解释，请随时告诉我！
r"""
Supports: https://github.com/ymcui/Chinese-LLaMA-Alpaca-2
          https://huggingface.co/ziqingyang/chinese-alpaca-2-7b
"""
register_template(
    name="llama2_zh",
    prefix=[
        "<<SYS>>\n{{system}}\n<</SYS>>\n\n"
    ],
    prompt=[
        "[INST] {{query}} [/INST] "
    ],
    system="You are a helpful assistant. 你是一个乐于助人的助手。",
    sep=[]
)



# 这个 "alpaca" 模板支持 [`https://huggingface.co/tatsu-lab/alpaca-7b-wdiff`](https://huggingface.co/tatsu-lab/alpaca-7b-wdiff) 和
#     [`https://github.com/ymcui/Chinese-LLaMA-Alpaca`](https://github.com/ymcui/Chinese-LLaMA-Alpaca) 这两个预训练模型。以下是注册过程中传递的参数说明：
# - **name:** "alpaca"
# - **prefix:** 包含一个字符串 "{{system}}" 的列表。其中 `{{system}}` 会被替换为模型的系统说明。
# - **prompt:** 包含一个字符串 "### Instruction:\n{{query}}\n\n### Response:\n" 的列表。其中 `{{query}}` 表示用户的查询，"### Instruction:" 和 "### Response:" 用于标识任务说明和生成的响应。
# - **system:** 包含对模型系统的说明，描述了一个需要完成任务的指令。助手的目标是根据指令生成合适的响应。
# - **sep:** 包含一个字符串 "\n\n" 的列表。这表示任务说明和生成的响应之间有一个双换行符的分隔。
# 注册 "alpaca" 模板后，你可以在其他地方通过名称 "alpaca" 获取并使用这个对话模板。
r"""
Supports: https://huggingface.co/tatsu-lab/alpaca-7b-wdiff
          https://github.com/ymcui/Chinese-LLaMA-Alpaca
"""
register_template(
    name="alpaca",
    prefix=[
        "{{system}}"
    ],
    prompt=[
        "### Instruction:\n{{query}}\n\n### Response:\n"
    ],
    system=(
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request."
    ),
    sep=[
        "\n\n"
    ]
)


# 这个 "vicuna" 模板支持 [`https://huggingface.co/lmsys/vicuna-7b-delta-v1.1`](https://huggingface.co/lmsys/vicuna-7b-delta-v1.1)和
#     [`https://huggingface.co/lmsys/vicuna-13b-delta-v1.1`](https://huggingface.co/lmsys/vicuna-13b-delta-v1.1) 这两个预训练模型。以下是注册过程中传递的参数说明：
# - **name:** "vicuna"
# - **prefix:** 包含一个字符串 "{{system}}" 的列表。其中 `{{system}}` 会被替换为模型的系统说明。
# - **prompt:** 包含一个字符串 "USER: {{query}} ASSISTANT: " 的列表。其中 `{{query}}` 表示用户的查询，"USER:" 和 "ASSISTANT:" 用于标识用户和助手的对话。
# - **system:** 包含对模型系统的说明，描述了一个需要完成任务的对话场景。助手的目标是根据用户的查询生成合适的响应。
# - **sep:** 空列表。这表示用户的查询和生成的响应之间没有额外的分隔。
# 注册 "vicuna" 模板后，你可以在其他地方通过名称 "vicuna" 获取并使用这个对话模板。
r"""
Supports: https://huggingface.co/lmsys/vicuna-7b-delta-v1.1
          https://huggingface.co/lmsys/vicuna-13b-delta-v1.1
"""
register_template(
    name="vicuna",
    prefix=[
        "{{system}}"
    ],
    prompt=[
        "USER: {{query}} ASSISTANT: "
    ],
    system=(
        "A chat between a curious user and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the user's questions."
    ),
    sep=[]
)



# "belle" 模板支持 [`https://huggingface.co/BelleGroup/BELLE-LLaMA-EXT-13B`](https://huggingface.co/BelleGroup/BELLE-LLaMA-EXT-13B) 预训练模型。以下是注册过程中传递的参数说明：
# - **name:** "belle"
# - **prefix:** 包含一个字符串 "{{system}}" 的列表。其中 `{{system}}` 会被替换为模型的系统说明。
# - **prompt:** 包含一个字符串 "Human: {{query}}\n\nBelle: " 的列表。其中 `{{query}}` 表示用户的查询，"Human:" 和 "Belle:" 用于标识用户和模型的对话。
# - **system:** 包含对模型系统的说明，描述了一个需要完成任务的对话场景。在这个场景中，用户和 "Belle" 模型进行对话。
# - **sep:** 包含一个字符串 "\n\n" 的列表。这表示用户的查询和生成的响应之间有两个换行符的分隔。
# 注册 "belle" 模板后，你可以在其他地方通过名称 "belle" 获取并使用这个对话模板。
r"""
Supports: https://huggingface.co/BelleGroup/BELLE-LLaMA-EXT-13B
"""
register_template(
    name="belle",
    prefix=[
        "{{system}}"
    ],
    prompt=[
        "Human: {{query}}\n\nBelle: "
    ],
    system="",
    sep=[
        "\n\n"
    ]
)


# "linly" 模板支持 [https://github.com/CVI-SZU/Linly](https://github.com/CVI-SZU/Linly) 项目。以下是注册过程中传递的参数说明：
# - **name:** "linly"
# - **prefix:** 包含一个字符串 "{{system}}" 的列表。其中 `{{system}}` 会被替换为模型的系统说明。
# - **prompt:** 包含一个字符串 "User: {{query}}\nBot: " 的列表。其中 `{{query}}` 表示用户的查询，"User:" 和 "Bot:" 用于标识用户和模型的对话。
# - **system:** 包含对模型系统的说明，描述了一个需要完成任务的对话场景。在这个场景中，用户和 "Bot" 模型进行对话。
# - **sep:** 包含一个字符串 "\n" 的列表。这表示用户的查询和生成的响应之间有一个换行符的分隔。
# 注册 "linly" 模板后，你可以在其他地方通过名称 "linly" 获取并使用这个对话模板。
r"""
Supports: https://github.com/CVI-SZU/Linly
"""
register_template(
    name="linly",
    prefix=[
        "{{system}}"
    ],
    prompt=[
        "User: {{query}}\nBot: "
    ],
    system="",
    sep=[
        "\n"
    ]
)


# "billa" 模板支持 [https://github.com/Neutralzz/BiLLa](https://github.com/Neutralzz/BiLLa) 项目。以下是注册过程中传递的参数说明：
# - **name:** "billa"
# - **prefix:** 包含一个字符串 "{{system}}" 的列表。其中 `{{system}}` 会被替换为模型的系统说明。
# - **prompt:** 包含一个字符串 "Human: {{query}}\nAssistant: " 的列表。其中 `{{query}}` 表示用户的查询，"Human:" 和 "Assistant:" 用于标识用户和模型的对话。
# - **system:** 包含对模型系统的说明，描述了一个需要完成任务的对话场景。在这个场景中，用户和助手模型进行对话。
# - **sep:** 包含一个字符串 "\n" 的列表。这表示用户的查询和生成的响应之间有一个换行符的分隔。
# 注册 "billa" 模板后，你可以在其他地方通过名称 "billa" 获取并使用这个对话模板。
r"""
Supports: https://github.com/Neutralzz/BiLLa
"""
register_template(
    name="billa",
    prefix=[
        "{{system}}"
    ],
    prompt=[
        "Human: {{query}}\nAssistant: "
    ],
    system="",
    sep=[
        "\n"
    ]
)


# "ziya" 模板支持 [https://huggingface.co/IDEA-CCNL/Ziya-LLaMA-13B-v1](https://huggingface.co/IDEA-CCNL/Ziya-LLaMA-13B-v1) 项目。以下是注册过程中传递的参数说明：
# - **name:** "ziya"
# - **prefix:** 包含一个字符串 "{{system}}" 的列表。其中 `{{system}}` 会被替换为模型的系统说明。
# - **prompt:** 包含三个部分的列表。第一个部分是一个字典 `{"token": "<human>"}`，表示用户的标识；第二个部分是一个字符串 ":{{query}}\n"，表示用户的查询；第三个部分是一个字典 `{"token": "<bot>"}`，表示模型的标识。
# - **system:** 包含对模型系统的说明，描述了一个需要完成任务的对话场景。在这个场景中，用户和助手模型进行对话。
# - **sep:** 包含一个字符串 "\n" 的列表。这表示用户的查询和生成的响应之间有一个换行符的分隔。
# 注册 "ziya" 模板后，你可以在其他地方通过名称 "ziya" 获取并使用这个对话模板。
r"""
Supports: https://huggingface.co/IDEA-CCNL/Ziya-LLaMA-13B-v1
"""
register_template(
    name="ziya",
    prefix=[
        "{{system}}"
    ],
    prompt=[
        {"token": "<human>"},
        ":{{query}}\n",
        {"token": "<bot>"},
        ":"
    ],
    system="",
    sep=[
        "\n"
    ]
)


# "aquila" 模板支持 [https://huggingface.co/qhduan/aquilachat-7b](https://huggingface.co/qhduan/aquilachat-7b) 项目。以下是注册过程中传递的参数说明：
# - **name:** "aquila"
# - **prefix:** 包含一个字符串 "{{system}}" 的列表。其中 `{{system}}` 会被替换为模型的系统说明。
# - **prompt:** 包含两个部分的列表。第一个部分是一个字符串 "Human: {{query}}"，表示用户的查询；第二个部分是一个字符串 "###Assistant: "，表示模型的标识。
# - **system:** 包含对模型系统的说明，描述了一个需要完成任务的对话场景。在这个场景中，用户和助手模型进行对话。
# - **sep:** 包含一个字符串 "###" 的列表。这表示用户的查询和生成的响应之间有一个 "###" 的分隔。
# 注册 "aquila" 模板后，你可以在其他地方通过名称 "aquila" 获取并使用这个对话模板。
r"""
Supports: https://huggingface.co/qhduan/aquilachat-7b
"""
register_template(
    name="aquila",
    prefix=[
        "{{system}}"
    ],
    prompt=[
        "Human: {{query}}###Assistant: "
    ],
    system=(
        "A chat between a curious human and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the human's questions."
    ),
    sep=[
        "###"
    ]
)


# "intern" 模板支持 [https://huggingface.co/internlm/internlm-chat-7b](https://huggingface.co/internlm/internlm-chat-7b) 项目。以下是注册过程中传递的参数说明：
# - **name:** "intern"
# - **prefix:** 包含一个字符串 "{{system}}" 的列表。其中 `{{system}}` 会被替换为模型的系统说明。
# - **prompt:** 包含三个部分的列表。第一个部分是一个字符串 ":{{query}}"，表示用户的查询；第二个部分是一个字典 `{"token": "<eoh>"}`，表示 "End of Human Utterance" 标记；第三个部分是一个字符串 ":\n"，表示模型的标识。
# - **system:** 包含对模型系统的说明，描述了一个需要完成任务的对话场景。在这个场景中，用户和助手模型进行对话。
# - **sep:** 包含两个部分的列表。第一个部分是一个字典 `{"token": "<eoa>"}`，表示 "End of Assistant Utterance" 标记；第二个部分是一个字符串 "\n"，表示用户的查询和生成的响应之间的分隔。
# - **stop_words:** 包含一个字符串 "<eoa>" 的列表。这是要添加到 tokenizer 作为特殊标记的停用词。
# - **efficient_eos:** 设置为 `True`，表示在模型的生成中使用有效的 EOS（End of Sequence）策略。
# 注册 "intern" 模板后，你可以在其他地方通过名称 "intern" 获取并使用这个对话模板。
r"""
Supports: https://huggingface.co/internlm/internlm-chat-7b
"""
register_template(
    name="intern",
    prefix=[
        "{{system}}"
    ],
    prompt=[
        "<|User|>:{{query}}",
        {"token": "<eoh>"},
        "\n<|Bot|>:"
    ],
    system="",
    sep=[
        {"token": "<eoa>"},
        "\n"
    ],
    stop_words=[
        "<eoa>"
    ],
    efficient_eos=True
)


# "baichuan" 模板支持 [https://huggingface.co/baichuan-inc/Baichuan-13B-Chat](https://huggingface.co/baichuan-inc/Baichuan-13B-Chat) 项目。以下是注册过程中传递的参数说明：
# - **name:** "baichuan"
# - **prefix:** 包含一个字符串 "{{system}}" 的列表。其中 `{{system}}` 会被替换为模型的系统说明。
# - **prompt:** 包含三个部分的列表。第一个部分是一个字典 `{"token": "<reserved_102>"}`，表示用户的 Token；第二个部分是一个字符串 "{{query}}"，表示用户的查询；第三个部分是一个字典 `{"token": "<reserved_103>"}`，表示助手模型的 Token。
# - **system:** 包含对模型系统的说明，描述了一个需要完成任务的对话场景。在这个场景中，用户和助手模型进行对话。
# - **sep:** 包含一个空列表，表示在用户的查询和生成的响应之间没有分隔符。
# - **efficient_eos:** 设置为 `True`，表示在模型的生成中使用有效的 EOS（End of Sequence）策略。
# 注册 "baichuan" 模板后，你可以在其他地方通过名称 "baichuan" 获取并使用这个对话模板。
r"""
Supports: https://huggingface.co/baichuan-inc/Baichuan-13B-Chat
"""
register_template(
    name="baichuan",
    prefix=[
        "{{system}}"
    ],
    prompt=[
        {"token": "<reserved_102>"}, # user token
        "{{query}}",
        {"token": "<reserved_103>"}  # assistant token
    ],
    system="",
    sep=[],
    efficient_eos=True
)


# "baichuan2" 模板支持 [https://huggingface.co/baichuan-inc/Baichuan2-7B-Chat](https://huggingface.co/baichuan-inc/Baichuan2-7B-Chat) 和
# [https://huggingface.co/baichuan-inc/Baichuan2-13B-Chat](https://huggingface.co/baichuan-inc/Baichuan2-13B-Chat) 两个项目。以下是注册过程中传递的参数说明：
# - **name:** "baichuan2"
# - **prefix:** 包含一个字符串 "{{system}}" 的列表。其中 `{{system}}` 会被替换为模型的系统说明。
# - **prompt:** 包含三个部分的列表。第一个部分是一个字典 `{"token": "<reserved_106>"}`，表示用户的 Token；第二个部分是一个字符串 "{{query}}"，表示用户的查询；第三个部分是一个字典 `{"token": "<reserved_107>"}`，表示助手模型的 Token。
# - **system:** 包含对模型系统的说明，描述了一个需要完成任务的对话场景。在这个场景中，用户和助手模型进行对话。
# - **sep:** 包含一个空列表，表示在用户的查询和生成的响应之间没有分隔符。
# - **efficient_eos:** 设置为 `True`，表示在模型的生成中使用有效的 EOS（End of Sequence）策略。
# 注册 "baichuan2" 模板后，你可以在其他地方通过名称 "baichuan2" 获取并使用这个对话模板。
r"""
Supports: https://huggingface.co/baichuan-inc/Baichuan2-7B-Chat
          https://huggingface.co/baichuan-inc/Baichuan2-13B-Chat
"""
register_template(
    name="baichuan2",
    prefix=[
        "{{system}}"
    ],
    prompt=[
        {"token": "<reserved_106>"}, # user token
        "{{query}}",
        {"token": "<reserved_107>"}  # assistant token
    ],
    system="",
    sep=[],
    efficient_eos=True
)


# "starchat" 模板支持两个项目：[https://huggingface.co/HuggingFaceH4/starchat-alpha](https://huggingface.co/HuggingFaceH4/starchat-alpha)和
#     [https://huggingface.co/HuggingFaceH4/starchat-beta](https://huggingface.co/HuggingFaceH4/starchat-beta)。以下是注册过程中传递的参数说明：
# - **name:** "starchat"
# - **prefix:** 包含两个部分的列表。第一个部分是一个字典 `{"token": ""}`，表示一个空 Token；第二个部分是一个字符串 "\n{{system}}"，表示系统说明的换行。
# - **prompt:** 包含五个部分的列表。前两个部分是一个字典 `{"token": ""}` 和一个字符串 "\n{{system}}"，表示系统说明的空 Token 和换行。第三个部分是一个字典 `{"token": ""}`，表示一个空 Token。第四个部分是一个字符串 "\n{{query}}"，表示用户的查询。最后一个部分是一个字典 `{"token": ""}`，表示一个空 Token。
# - **system:** 包含对模型系统的说明，描述了一个需要完成任务的对话场景。在这个场景中，用户和助手模型进行对话。
# - **sep:** 包含两个部分的列表。第一个部分是一个字典 `{"token": ""}`，表示一个空 Token；第二个部分是一个字符串 "\n"，表示在用户的查询和生成的响应之间的换行。
# - **stop_words:** 包含一个空字符串的列表，表示不使用停用词。
# - **efficient_eos:** 设置为 `True`，表示在模型的生成中使用有效的 EOS（End of Sequence）策略。
# 注册 "starchat" 模板后，你可以在其他地方通过名称 "starchat" 获取并使用这个对话模板。
r"""
Supports: https://huggingface.co/HuggingFaceH4/starchat-alpha
          https://huggingface.co/HuggingFaceH4/starchat-beta
"""
register_template(
    name="starchat",
    prefix=[
        {"token": "<|system|>"},
        "\n{{system}}",
    ],
    prompt=[
        {"token": "<|user|>"},
        "\n{{query}}",
        {"token": "<|end|>"},
        "\n",
        {"token": "<|assistant|>"}
    ],
    system="",
    sep=[
        {"token": "<|end|>"},
        "\n"
    ],
    stop_words=[
        "<|end|>"
    ],
    efficient_eos=True
)


# "chatml" 模板支持项目 [https://huggingface.co/Qwen/Qwen-7B-Chat](https://huggingface.co/Qwen/Qwen-7B-Chat)。以下是注册过程中传递的参数说明：
# - **name:** "chatml"
# - **prefix:** 包含两个部分的列表。第一个部分是一个字典 `{"token": ""}`，表示一个空 Token；第二个部分是一个字符串 "system\n{{system}}"，表示系统说明的空 Token 和换行。
# - **prompt:** 包含七个部分的列表。前两个部分是一个字典 `{"token": ""}` 和一个字符串 "user\n{{query}}"，表示用户的空 Token 和查询。
#       第三个部分是一个字典 `{"token": ""}`，表示一个空 Token。第四个部分是一个字符串 "\n"，表示在用户的查询和生成的响应之间的换行。
#       第五个部分是一个字典 `{"token": ""}`，表示一个空 Token。第六个部分是一个字符串 "assistant\n"，表示助手的说明。最后一个部分是一个字典 `{"token": ""}`，表示一个空 Token。
# - **system:** 包含对模型系统的说明，描述了一个需要完成任务的对话场景。在这个场景中，用户和助手模型进行对话。
# - **sep:** 包含两个部分的列表。第一个部分是一个字典 `{"token": ""}`，表示一个空 Token；第二个部分是一个字符串 "\n"，表示在用户的查询和生成的响应之间的换行。
# - **stop_words:** 包含一个空字符串的列表，表示不使用停用词。
# - **efficient_eos:** 设置为 `True`，表示在模型的生成中使用有效的 EOS（End of Sequence）策略。
# 注册 "chatml" 模板后，你可以在其他地方通过名称 "chatml" 获取并使用这个对话模板。
r"""
Supports: https://huggingface.co/Qwen/Qwen-7B-Chat
"""
register_template(
    name="chatml",
    prefix=[
        {"token": "<|im_start|>"},
        "system\n{{system}}"
    ],
    prompt=[
        {"token": "<|im_start|>"},
        "user\n{{query}}",
        {"token": "<|im_end|>"},
        "\n",
        {"token": "<|im_start|>"},
        "assistant\n"
    ],
    system="You are a helpful assistant.",
    sep=[
        {"token": "<|im_end|>"},
        "\n"
    ],
    stop_words=[
        "<|im_end|>"
    ],
    efficient_eos=True
)


# "chatglm2" 模板支持项目 [https://huggingface.co/THUDM/chatglm2-6b](https://huggingface.co/THUDM/chatglm2-6b)。以下是注册过程中传递的参数说明：
# - **name:** "chatglm2"
# - **prefix:** 包含三个部分的列表。第一个部分是一个字典 `{"token": "[gMASK]"}`，表示一个标记化为 "[gMASK]" 的特殊 Token。
#     第二个部分是一个字典 `{"token": "sop"}`，表示一个标记化为 "sop" 的特殊 Token。第三个部分是一个字符串 "{{system}}"，表示对系统的说明。
# - **prompt:** 包含三个部分的列表。第一个部分是一个字符串 "[Round {{idx}}]"，表示轮数标记。
#     第二个部分是一个字符串 "\n\n问：{{query}}\n\n答："，表示在用户的查询和生成的响应之间的换行，以及问题的标记。第三个部分是一个空字符串，表示生成的响应。
# - **system:** 包含对模型系统的说明，描述了一个需要完成任务的对话场景。在这个场景中，用户和助手模型进行对话。
# - **sep:** 包含一个字符串 "\n\n" 的列表，表示在用户的查询和生成的响应之间的换行。
# - **efficient_eos:** 设置为 `True`，表示在模型的生成中使用有效的 EOS（End of Sequence）策略。
# 注册 "chatglm2" 模板后，你可以在其他地方通过名称 "chatglm2" 获取并使用这个对话模板。
# 如果有其他问题或需要更多的帮助，请随时告诉我！
r"""
Supports: https://huggingface.co/THUDM/chatglm2-6b
"""
register_template(
    name="chatglm2",
    prefix=[
        {"token": "[gMASK]"},
        {"token": "sop"},
        "{{system}}"
    ],
    prompt=[
        "[Round {{idx}}]\n\n问：{{query}}\n\n答："
    ],
    system="",
    sep=[
        "\n\n"
    ],
    efficient_eos=True
)


# "chatglm3" 模板支持项目 [https://huggingface.co/THUDM/chatglm3-6b](https://huggingface.co/THUDM/chatglm3-6b)。以下是注册过程中传递的参数说明：
# - **name:** "chatglm3"
# - **prefix:** 包含三个部分的列表。第一个部分是一个字典 `{"token": "[gMASK]"}`，表示一个标记化为 "[gMASK]" 的特殊 Token。
#     第二个部分是一个字典 `{"token": "sop"}`，表示一个标记化为 "sop" 的特殊 Token。第三个部分是一个字符串 "{{system}}"，表示对系统的说明。
# - **prompt:** 包含三个部分的列表。第一个部分是一个字符串 "[Round {{idx}}]"，表示轮数标记。
#     第二个部分是一个字符串 "\n\n问：{{query}}\n\n答："，表示在用户的查询和生成的响应之间的换行，以及问题的标记。第三个部分是一个空字符串，表示生成的响应。
# - **system:** 包含对模型系统的说明，描述了一个需要完成任务的对话场景。在这个场景中，用户和助手模型进行对话。
# - **sep:** 包含一个字符串 "\n\n" 的列表，表示在用户的查询和生成的响应之间的换行。
# - **efficient_eos:** 设置为 `True`，表示在模型的生成中使用有效的 EOS（End of Sequence）策略。
# 注册 "chatglm3" 模板后，你可以在其他地方通过名称 "chatglm3" 获取并使用这个对话模板。
r"""
Supports: https://huggingface.co/THUDM/chatglm3-6b
"""
register_template(
    name="chatglm3",
    prefix=[
        {"token": "[gMASK]"},
        {"token": "sop"},
        "{{system}}"
    ],
    prompt=[
        "[Round {{idx}}]\n\n问：{{query}}\n\n答："
    ],
    system="",
    sep=[
        "\n\n"
    ],
    efficient_eos=True
)

# "xverse" 模板支持项目 [https://huggingface.co/xverse/XVERSE-13B-Chat](https://huggingface.co/xverse/XVERSE-13B-Chat)。以下是注册过程中传递的参数说明：
# - **name:** "xverse"
# - **prefix:** 包含一个字符串 "{{system}}" 的列表，表示对模型系统的说明。
# - **prompt:** 包含一个字符串 "Human: {{query}}\n\nAssistant: " 的列表，表示用户的查询和生成的响应之间的格式，以及生成的响应。
# - **system:** 包含对模型系统的说明，描述了一个需要完成任务的对话场景。在这个场景中，用户和助手模型进行对话。
# - **sep:** 包含一个空列表，表示没有特定的分隔符。
# 注册 "xverse" 模板后，你可以在其他地方通过名称 "xverse" 获取并使用这个对话模板。
r"""
Supports: https://huggingface.co/xverse/XVERSE-13B-Chat
"""
register_template(
    name="xverse",
    prefix=[
        "{{system}}"
    ],
    prompt=[
        "Human: {{query}}\n\nAssistant: "
    ],
    system="",
    sep=[]
)


