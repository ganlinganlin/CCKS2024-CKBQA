
# 这段代码导入了一些必要的库和模块，并定义了一些类型提示。以下是对代码的简要说明：
# 1. `gc`：Python的垃圾回收模块，用于管理内存。
# 2. `torch`：PyTorch库，用于深度学习模型的构建和训练。
# 3. `TYPE_CHECKING`：用于在类型提示中检查类型的标志。在运行时会被视为`False`，但在类型检查时会被视为`True`，以避免循环导入。
# 4. `Tuple`：用于表示元组类型的类。
# 5. `InfNanRemoveLogitsProcessor`：来自transformers库的类，用于处理logits中的无穷大（Inf）和无效值（NaN）。
# 6. `LogitsProcessorList`：来自transformers库的类，用于组合多个logits处理器。
# 这些库和模块通常在深度学习模型的训练和推理过程中使用。
import gc
import torch
from typing import TYPE_CHECKING, Tuple
from transformers import InfNanRemoveLogitsProcessor, LogitsProcessorList

if TYPE_CHECKING:
    from transformers.modeling_utils import PreTrainedModel

# 这段代码定义了一个`AverageMeter`类，用于计算和存储平均值和当前值。以下是该类的主要方法和属性：
# - `__init__(self)`: 类的初始化方法，在创建类的实例时调用。调用`reset`方法进行初始化。
# - `reset(self)`: 重置所有计数器，将`val`、`avg`、`sum`和`count`都设为零。
# - `update(self, val, n=1)`: 更新计数器，传入当前值`val`和可选的权重`n`。更新`sum`和`count`，计算并更新平均值`avg`。
# 该类通常在训练过程中用于跟踪损失值或其他指标的平均值。通过调用`update`方法来更新值，然后可以通过访问`avg`属性来获取平均值。
# 例如，可以使用`AverageMeter`来跟踪训练过程中的平均损失：
# 这使得在训练过程中能够轻松地计算和跟踪平均损失值。
class AverageMeter:
    r"""
    Computes and stores the average and current value.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# 这个函数用于计算模型中的可训练参数数量和总参数数量。以下是该函数的主要功能：
# - `count_parameters(model: torch.nn.Module) -> Tuple[int, int]`: 接受一个`torch.nn.Module`类型的模型作为输入，返回一个包含两个整数的元组，分别表示可训练参数的数量和总参数的数量。
# 在函数内部，通过遍历模型的所有参数，使用`param.numel()`获取每个参数的元素数量，然后根据一些条件进行处理：
# - 如果使用了 DS Zero 3 并且权重是空的，使用`param.ds_numel`获取元素数量。
# - 如果参数是来自 BitsAndBytes 库的 4 位线性层（4bit linear layers），将元素数量乘以 2。
# - 累加可训练参数数量和总参数数量。
# 最终返回一个包含可训练参数数量和总参数数量的元组。
def count_parameters(model: torch.nn.Module) -> Tuple[int, int]:
    r"""
    Returns the number of trainable parameters and number of all parameters in the model.
    """
    trainable_params, all_param = 0, 0
    for param in model.parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        # Due to the design of 4bit linear layers from bitsandbytes, multiply the number of parameters by 2
        if param.__class__.__name__ == "Params4bit":
            num_params = num_params * 2

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params

    return trainable_params, all_param


# 以下是这两个函数的功能说明：
# 1. `get_logits_processor() -> LogitsProcessorList`:
#    - 返回一个`LogitsProcessorList`对象，其中包含一个`InfNanRemoveLogitsProcessor`实例。
#    - `InfNanRemoveLogitsProcessor`用于处理模型输出的 logits，移除其中的无穷大（inf）和 NaN（Not a Number）。
# 2. `torch_gc() -> None`:
#    - 用于收集 GPU 内存。
#    - 使用`gc.collect()`进行 Python 内存垃圾回收。
#    - 如果检测到 CUDA 可用，使用`torch.cuda.empty_cache()`清空 CUDA 缓存，然后使用`torch.cuda.ipc_collect()`收集 CUDA 内存。这有助于释放由 PyTorch 占用的 GPU 内存。
# 这两个函数可以在训练期间定期调用，以确保及时释放不再使用的 GPU 内存，从而提高程序的内存利用效率。
def get_logits_processor() -> LogitsProcessorList:
    logits_processor = LogitsProcessorList()
    logits_processor.append(InfNanRemoveLogitsProcessor())
    return logits_processor


def torch_gc() -> None:
    r"""
    Collects GPU memory.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

# 这个`dispatch_model`函数的作用是将预训练模型分发到 GPU 上，以实现平衡的内存使用。以下是该函数的主要步骤：
# 1. **检查模型是否已经以8位或4位加载（Quantized）：**
#    - 如果模型已经以8位或4位加载，就不做任何处理，直接返回原模型。
# 2. **检查 GPU 数量：**
#    - 如果当前系统有多个 GPU，就使用 [accelerate](https://github.com/huggingface/accelerate) 库中的函数来进行模型分发和内存平衡。
# 3. **获取平衡的内存分配：**
#    - 通过调用 `get_balanced_memory` 函数获取模型的平衡内存分配。
# 4. **确保 tied weights 被 tied：**
#    - 在创建设备映射之前，确保 tied weights 被 tied。
# 5. **推断自动设备映射：**
#    - 使用 `infer_auto_device_map` 函数推断出自动的设备映射。
# 6. **调用 `dispatch_model` 函数：**
#    - 使用 `dispatch_model` 函数将模型分发到 GPU 上。
# 如果系统只有一个 GPU，就直接使用 PyTorch 的 `cuda()` 函数将模型移到 GPU 上。
def dispatch_model(model: "PreTrainedModel") -> "PreTrainedModel":
    r"""
    Dispatches a pre-trained model to GPUs with balanced memory.
    Borrowed from: https://github.com/huggingface/transformers/blob/v4.31.0/src/transformers/modeling_utils.py#L2803
    """
    if getattr(model, "is_loaded_in_8bit", False) or getattr(model, "is_loaded_in_4bit", False): # do nothing
        return model

    if torch.cuda.device_count() > 1:
        from accelerate import dispatch_model
        from accelerate.utils import infer_auto_device_map, get_balanced_memory

        if model._no_split_modules is None:
            raise ValueError("The model class needs to implement the `_no_split_modules` attribute.")

        kwargs = {"dtype": model.dtype, "no_split_module_classes": model._no_split_modules}
        max_memory = get_balanced_memory(model, **kwargs)
        # Make sure tied weights are tied before creating the device map.
        model.tie_weights()
        device_map = infer_auto_device_map(model, max_memory=max_memory, **kwargs)
        return dispatch_model(model, device_map)
    else:
        return model.cuda()
