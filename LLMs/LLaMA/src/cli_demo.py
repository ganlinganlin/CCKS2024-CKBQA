from llmtuner import ChatModel

# 这段代码看起来是一个简单的命令行界面（CLI）应用程序，使用了 `ChatModel` 类来生成对话响应。下面是一些主要的点解释：
# 1. `from llmtuner import ChatModel`:
#    - 导入了 `ChatModel` 类，这可能是一个用于处理聊天对话的类。
# 2. `def main():`:
#    - 定义了主函数，其中创建了 `ChatModel` 的实例，并初始化了一个空的历史记录列表。
def main():
    chat_model = ChatModel()
    history = []
    print("Welcome to the CLI application, use `clear` to remove the history, use `exit` to exit the application.")

    # 3. `while True:` 循环:
    #    - 这是一个无限循环，用户可以不断输入对话。
    # 4. `try` 块:
    #    - 尝试获取用户输入，如果检测到 `UnicodeDecodeError`，则打印错误消息并继续下一次循环。
    # 5. `if query.strip() == "exit":`:
    #    - 如果用户输入 "exit"，则退出循环，结束应用程序。
    # 6. `if query.strip() == "clear":`:
    #    - 如果用户输入 "clear"，则清空历史记录列表并打印相应消息。
    # 7. 对话生成部分:
    #    - 通过调用 `chat_model.stream_chat(query, history)`，生成助手的响应，然后将对话打印出来。
    # 8. `history` 更新:
    #    - 更新历史记录，将用户输入和助手响应添加到历史记录列表中。
    # 这个CLI应用程序的基本逻辑是用户输入一条消息，然后通过 `ChatModel` 生成助手的回复，
    #   并将整个对话历史记录保存在 `history` 列表中。用户可以输入 "exit" 退出应用程序，输入 "clear" 清空历史记录。
    while True:
        try:
            query = input("\nUser: ")
        except UnicodeDecodeError:
            print("Detected decoding error at the inputs, please set the terminal encoding to utf-8.")
            continue
        except Exception:
            raise

        if query.strip() == "exit":
            break

        if query.strip() == "clear":
            history = []
            print("History has been removed.")
            continue

        print("Assistant: ", end="", flush=True)

        response = ""
        for new_text in chat_model.stream_chat(query, history):
            print(new_text, end="", flush=True)
            response += new_text
        print()

        history = history + [(query, response)]


if __name__ == "__main__":
    main()
