import time
from enum import Enum
from pydantic import BaseModel, Field
from typing import List, Optional

# 这段代码定义了一系列 Pydantic 模型，用于描述聊天模型 API 中的各种请求和响应。以下是每个模型的简要解释：

# 1. **`Role` 枚举:**
#    - 定义了三种角色：`USER`、`ASSISTANT` 和 `SYSTEM`，用于表示消息的发送者。
class Role(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"

# 2. **`Finish` 枚举:**
#    - 定义了两种结束类型：`STOP` 和 `LENGTH`，用于表示聊天生成的结束原因。
class Finish(str, Enum):
    STOP = "stop"
    LENGTH = "length"

# 3. **`ModelCard` 模型:**
#    - 描述了一个模型卡片，包括模型的一些基本信息，如 `id`、`object`、`created` 等。
class ModelCard(BaseModel):
    id: str
    object: Optional[str] = "model"
    created: Optional[int] = Field(default_factory=lambda: int(time.time()))
    owned_by: Optional[str] = "owner"
    root: Optional[str] = None
    parent: Optional[str] = None
    permission: Optional[list] = []

# 4. **`ModelList` 模型:**
#    - 描述了一个模型卡片列表，包括 `object` 类型和 `data`（包含多个模型卡片的列表）。
class ModelList(BaseModel):
    object: Optional[str] = "list"
    data: Optional[List[ModelCard]] = []


# 5. **`ChatMessage` 模型:**
#    - 描述了一个聊天消息，包括 `role`（发送者角色）和 `content`（消息内容）。
class ChatMessage(BaseModel):
    role: Role
    content: str

# 6. **`DeltaMessage` 模型:**
#    - 描述了一个消息变更，包括 `role` 和 `content`。
class DeltaMessage(BaseModel):
    role: Optional[Role] = None
    content: Optional[str] = None

# 7. **`ChatCompletionRequest` 模型:**
#    - 描述了一个聊天完成的请求，包括模型标识、消息列表和一些生成选项，如 `do_sample`、`temperature` 等。
class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    do_sample: Optional[bool] = True
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    n: Optional[int] = 1
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False

# 8. **`ChatCompletionResponseChoice` 模型:**
#    - 描述了一个聊天完成的响应选择，包括选择的索引、相应的消息和结束原因。
class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Finish

# 9. **`ChatCompletionResponseStreamChoice` 模型:**
#    - 描述了一个聊天完成的响应流选择，包括选择的索引、消息的变更和结束原因。
class ChatCompletionResponseStreamChoice(BaseModel):
    index: int
    delta: DeltaMessage
    finish_reason: Optional[Finish] = None

# 10. **`ChatCompletionResponseUsage` 模型:**
#    - 描述了聊天完成响应的使用情况，包括提示令牌数、完成令牌数和总令牌数。
class ChatCompletionResponseUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


# 11. **`ChatCompletionResponse` 模型:**
#    - 描述了一个聊天完成的响应，包括 `id`、`object`、`created`、模型标识、选择列表和使用情况。
class ChatCompletionResponse(BaseModel):
    id: Optional[str] = "chatcmpl-default"
    object: Optional[str] = "chat.completion"
    created: Optional[int] = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionResponseChoice]
    usage: ChatCompletionResponseUsage


# 12. **`ChatCompletionStreamResponse` 模型:**
#    - 描述了一个聊天完成的流响应，包括 `id`、`object`、`created`、模型标识、选择流列表。
class ChatCompletionStreamResponse(BaseModel):
    id: Optional[str] = "chatcmpl-default"
    object: Optional[str] = "chat.completion.chunk"
    created: Optional[int] = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionResponseStreamChoice]
