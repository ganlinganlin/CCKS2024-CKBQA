import sys
import logging

# 这是一个自定义的`LoggerHandler`类，继承自Python的`logging.Handler`。这个类有以下几个特点：
# 1. **reset方法**: 将log字符串重置为空字符串。
# 2. **emit方法**: 在这个方法中，处理每一条log记录。在这里，如果记录的名称是"httpx"，则直接返回，否则将log记录的格式化字符串添加到log属性中。
# 这个类的作用是捕获日志记录，将其格式化并存储在`self.log`属性中。在你的应用程序中，你可以创建一个`LoggerHandler`的实例，将其添加到你的日志记录器中，然后在需要的时候获取或处理`self.log`属性的值。
class LoggerHandler(logging.Handler):

    def __init__(self):
        super().__init__()
        self.log = ""

    def reset(self):
        self.log = ""

    def emit(self, record):
        if record.name == "httpx":
            return
        log_entry = self.format(record)
        self.log += log_entry
        self.log += "\n\n"

# 这个`reset_logging`函数用于移除根日志器的基本配置。在Python中，根日志器（root logger）是默认的日志器，当你在应用程序中使用`logging`模块时，它会被自动创建。
# 这个函数的作用是移除根日志器上已有的所有处理器（handlers）和过滤器（filters），从而清空其配置。这样做的目的可能是在某些情况下，你想要重新配置日志记录系统，或者你想要更灵活地管理日志的配置。
def reset_logging():
    r"""
    Removes basic config of root logger
    """
    root = logging.getLogger()
    list(map(root.removeHandler, root.handlers))
    list(map(root.removeFilter, root.filters))

# 这个`get_logger`函数用于获取一个已配置的日志器（logger）。函数接受一个`name`参数，该参数是日志器的名称，通常用于标识日志记录的来源。
# 函数的实现步骤如下：
# 1. 创建一个`Formatter`对象，用于指定日志记录的格式，包括时间戳、日志级别、日志器名称和消息内容。
# 2. 创建一个`StreamHandler`对象，将日志信息输出到标准输出流（stdout）。
# 3. 创建一个具有指定名称的日志器。
# 4. 将创建的`StreamHandler`对象添加到日志器中。
# 5. 将日志器的日志级别设置为`INFO`级别。
# 最终，该函数返回已配置的日志器，可以使用这个日志器进行日志记录。
def get_logger(name: str) -> logging.Logger:
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S"
    )
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)

    return logger
