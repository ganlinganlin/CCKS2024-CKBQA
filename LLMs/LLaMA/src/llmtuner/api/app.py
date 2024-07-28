import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from sse_starlette import EventSourceResponse
from typing import List, Tuple

from llmtuner.extras.misc import torch_gc
from llmtuner.chat import ChatModel
from llmtuner.api.protocol import (
    Role,
    Finish,
    ModelCard,
    ModelList,
    ChatMessage,
    DeltaMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionStreamResponse,
    ChatCompletionResponseChoice,
    ChatCompletionResponseStreamChoice,
    ChatCompletionResponseUsage
)
# 这段代码看起来是一个使用 [FastAPI](https://fastapi.tiangolo.com/) 构建的 Web API 服务。以下是一些关键点的解释：
# 1. **导入模块:**
#    - `uvicorn` 用于运行 FastAPI 应用程序。
#    - `FastAPI` 是一个现代、快速（基于 Starlette 和 Pydantic）的 Web 框架。
#    - `HTTPException` 是 FastAPI 中用于处理 HTTP 异常的类。
#    - `CORSMiddleware` 用于处理跨域资源共享（CORS）。
#    - `asynccontextmanager` 是用于创建异步上下文管理器的工具。
#    - `EventSourceResponse` 用于支持 [Server-Sent Events (SSE)](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events)。
#    - `List` 和 `Tuple` 是用于定义类型提示的。
# 2. **导入项目中的其他模块:**
#    - `torch_gc` 来自 `llmtuner.extras.misc` 模块。
#    - `ChatModel` 来自 `llmtuner.chat` 模块。
#    - 各种协议消息（`Role`、`Finish`、`ModelCard` 等）和请求/响应类型来自 `llmtuner.api.protocol` 模块。
# 这段代码显示了一些模块和库的导入，为构建一个使用 FastAPI 的 Web API 服务做好了准备。如果你有特定的问题或者需要更多的解释，请告诉我！


# 这段代码使用 `asynccontextmanager` 装饰器定义了一个异步上下文管理器 `lifespan`，用于在 FastAPI 应用程序的生命周期中执行一些操作。
#     在这个上下文管理器中，使用了 `torch_gc()` 函数，可能是用于收集 GPU 内存的操作。
# 解释这段代码的关键点：
# 1. `@asynccontextmanager` 装饰器:
#    - 表明 `lifespan` 函数是一个异步上下文管理器。
# 2. `async def lifespan(app: FastAPI)`:
#    - 定义了 `lifespan` 函数，该函数接受一个 `FastAPI` 应用程序实例作为参数。
# 3. `yield`:
#    - 在异步上下文管理器中，`yield` 之前的代码会在进入上下文时执行，而 `yield` 之后的代码会在退出上下文时执行。
# 4. `torch_gc()`:
#    - 在退出上下文时，调用了 `torch_gc()` 函数，可能是用于执行一些与 PyTorch 的 GPU 内存管理相关的操作。
# 这样的设计可能是为了确保在 FastAPI 应用程序的生命周期中，特别是在应用程序关闭时，执行一些资源管理或清理操作。如果你对 `torch_gc()` 或者这个上下文管理器的使用有其他问题或需要更多的解释，请随时告诉我！
@asynccontextmanager
async def lifespan(app: FastAPI): # collects GPU memory
    yield
    torch_gc()

# 这段代码创建了一个 FastAPI 应用程序，并定义了两个端点用于模型列表和聊天完成。以下是关键点的解释：
# 1. **`create_app` 函数:**
#    - 接受一个 `ChatModel` 实例作为参数，返回一个 FastAPI 应用程序实例。
#    - 使用了 `lifespan` 上下文管理器。
def create_app(chat_model: ChatModel) -> FastAPI:
    app = FastAPI(lifespan=lifespan)

    # 2. **中间件:**
    #    - 添加了跨域资源共享（CORS）中间件，允许所有来源访问。
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 3. **`list_models` 端点:**
    #    - GET 请求，返回一个包含一个模型卡片的模型列表。
    @app.get("/v1/models", response_model=ModelList)
    async def list_models():
        model_card = ModelCard(id="gpt-3.5-turbo")
        return ModelList(data=[model_card])

    # 4. **`create_chat_completion` 端点:**
    #    - POST 请求，接受 `ChatCompletionRequest`，生成聊天完成响应。
    #    - 如果请求是流式的，使用 `predict` 函数生成流式响应。
    @app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
    async def create_chat_completion(request: ChatCompletionRequest):
        if len(request.messages) < 1 or request.messages[-1].role != Role.USER:
            raise HTTPException(status_code=400, detail="Invalid request")

        query = request.messages[-1].content
        prev_messages = request.messages[:-1]
        if len(prev_messages) > 0 and prev_messages[0].role == Role.SYSTEM:
            system = prev_messages.pop(0).content
        else:
            system = None

        history = []
        if len(prev_messages) % 2 == 0:
            for i in range(0, len(prev_messages), 2):
                if prev_messages[i].role == Role.USER and prev_messages[i+1].role == Role.ASSISTANT:
                    history.append([prev_messages[i].content, prev_messages[i+1].content])

        if request.stream:
            generate = predict(query, history, system, request)
            return EventSourceResponse(generate, media_type="text/event-stream")

        response, (prompt_length, response_length) = chat_model.chat(
            query, history, system,
            do_sample=request.do_sample,
            temperature=request.temperature,
            top_p=request.top_p,
            max_new_tokens=request.max_tokens
        )

        usage = ChatCompletionResponseUsage(
            prompt_tokens=prompt_length,
            completion_tokens=response_length,
            total_tokens=prompt_length+response_length
        )

        choice_data = ChatCompletionResponseChoice(
            index=0,
            message=ChatMessage(role=Role.ASSISTANT, content=response),
            finish_reason=Finish.STOP
        )

        return ChatCompletionResponse(model=request.model, choices=[choice_data], usage=usage)

    # 5. **`predict` 函数:**
    #    - 用于生成流式响应，通过 `chat_model.stream_chat` 逐步生成聊天响应。
    #    - 使用 `EventSourceResponse` 支持 Server-Sent Events。
    async def predict(query: str, history: List[Tuple[str, str]], system: str, request: ChatCompletionRequest):
        choice_data = ChatCompletionResponseStreamChoice(
            index=0,
            delta=DeltaMessage(role=Role.ASSISTANT),
            finish_reason=None
        )
        chunk = ChatCompletionStreamResponse(model=request.model, choices=[choice_data])
        yield chunk.json(exclude_unset=True, ensure_ascii=False)

        for new_text in chat_model.stream_chat(
            query, history, system,
            do_sample=request.do_sample,
            temperature=request.temperature,
            top_p=request.top_p,
            max_new_tokens=request.max_tokens
        ):
            if len(new_text) == 0:
                continue

            choice_data = ChatCompletionResponseStreamChoice(
                index=0,
                delta=DeltaMessage(content=new_text),
                finish_reason=None
            )
            chunk = ChatCompletionStreamResponse(model=request.model, choices=[choice_data])
            yield chunk.json(exclude_unset=True, ensure_ascii=False)

        choice_data = ChatCompletionResponseStreamChoice(
            index=0,
            delta=DeltaMessage(),
            finish_reason=Finish.STOP
        )
        chunk = ChatCompletionStreamResponse(model=request.model, choices=[choice_data])
        yield chunk.json(exclude_unset=True, ensure_ascii=False)
        yield "[DONE]"

    return app

# 6. **`if __name__ == "__main__":` 部分:**
#    - 在主程序中创建了一个 `ChatModel` 实例和一个 FastAPI 应用程序实例。
#    - 使用 `uvicorn.run` 运行应用程序。
# 整体上，这个脚本创建了一个 FastAPI 应用程序，提供了列出模型和生成聊天完成响应的功能。如果你有任何关于这个代码的特定问题或需要更多的解释，请告诉我！
if __name__ == "__main__":
    chat_model = ChatModel()
    app = create_app(chat_model)
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=1)

