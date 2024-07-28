
# 这段代码看起来是一个自定义的 Trainer Callback，用于记录训练过程中的一些信息到日志文件。
#     Trainer Callbacks 在 Hugging Face Transformers 库中用于在训练过程中添加额外的逻辑，例如记录日志、保存模型等。
# 这个 Callback 的主要作用是在训练的不同阶段记录一些信息到日志文件中，例如每个 epoch 结束后记录当前的学习率、每个 step 结束后记录训练速度等。
#     这样的信息记录有助于在训练过程中监控模型的性能和训练的进展。
# 以下是对主要函数的简要解释：
# - `on_epoch_end`：在每个 epoch 结束时调用，记录当前的学习率。
# - `on_step_end`：在每个 step 结束时调用，记录训练速度（steps per second）。
# 这个 Callback 使用了自定义的日志文件名 `LOG_FILE_NAME`，并且在训练结束时打印一些总结信息。
import os
import json
import time
from typing import TYPE_CHECKING
from datetime import timedelta

from transformers import TrainerCallback
from transformers.trainer_utils import has_length, PREFIX_CHECKPOINT_DIR

from llmtuner.extras.constants import LOG_FILE_NAME
from llmtuner.extras.logging import get_logger

if TYPE_CHECKING:
    from transformers import TrainingArguments, TrainerState, TrainerControl


logger = get_logger(__name__)

# 这个 `SavePeftModelCallback` 是一个自定义的 Trainer Callback，它在保存 checkpoint 之后（`on_save` 事件）和训练结束时（`on_train_end` 事件）执行一些特定逻辑。
# 主要逻辑如下：
# - `on_save`：在每次保存 checkpoint 后触发。它会检查当前模型是否是 PEFT 模型（通过检查模型是否有 `is_peft_model` 属性），
#     如果是，则保存其预训练模型的权重到与当前 checkpoint 相关的文件夹中。这里使用 `model.pretrained_model` 获取预训练模型。
# - `on_train_end`：在整个训练过程结束后触发。类似地，它检查模型是否是 PEFT 模型，如果是，则保存整个 PEFT 模型的权重到指定的 `output_dir` 中。
# 这个 Callback 的作用是在 PEFT 模型训练的过程中，除了保存整个模型的 checkpoint 外，还额外保存了预训练模型的权重。这在某些场景下可能是有用的，例如需要在训练后继续对预训练模型进行微调。
# 如果你有任何具体问题或需要更详细的解释，请告诉我。
class SavePeftModelCallback(TrainerCallback):

    def on_save(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        r"""
        Event called after a checkpoint save.
        """
        if args.should_save:
            output_dir = os.path.join(args.output_dir, "{}-{}".format(PREFIX_CHECKPOINT_DIR, state.global_step))
            model = kwargs.pop("model")
            if getattr(model, "is_peft_model", False):
                getattr(model, "pretrained_model").save_pretrained(output_dir)

    def on_train_end(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        r"""
        Event called at the end of training.
        """
        if args.should_save:
            model = kwargs.pop("model")
            if getattr(model, "is_peft_model", False):
                getattr(model, "pretrained_model").save_pretrained(args.output_dir)



# `LogCallback` 是一个用于记录训练过程中信息的自定义 Trainer Callback。以下是它的主要功能：
# - 在 `on_train_begin` 事件中，它会在训练开始时被调用。在这里，它记录了训练的开始时间 `start_time`，总步数 `max_steps`，并检查是否存在之前的日志文件，如果存在则删除。
# - 在 `on_train_end` 事件中，它会在训练结束时被调用。在这里，它将 `in_training` 设为 `False`，并将 `cur_steps` 和 `max_steps` 重置为 0。
# - 它还有一个 `timing` 方法用于计算训练过程中的时间信息，包括已经经过的时间 `elapsed_time` 和剩余时间 `remaining_time`。
# 这个 Callback 的主要目的是在训练开始和结束时记录一些时间相关的信息，并且在训练过程中，通过 `timing` 方法计算已经经过的时间和剩余时间。这些信息可能有助于实时监控训练进度。
class LogCallback(TrainerCallback):

    def __init__(self, runner=None):
        self.runner = runner
        self.in_training = False
        self.start_time = time.time()
        self.cur_steps = 0
        self.max_steps = 0
        self.elapsed_time = ""
        self.remaining_time = ""

    def timing(self):
        cur_time = time.time()
        elapsed_time = cur_time - self.start_time
        avg_time_per_step = elapsed_time / self.cur_steps if self.cur_steps != 0 else 0
        remaining_time = (self.max_steps - self.cur_steps) * avg_time_per_step
        self.elapsed_time = str(timedelta(seconds=int(elapsed_time)))
        self.remaining_time = str(timedelta(seconds=int(remaining_time)))

    def on_train_begin(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        r"""
        Event called at the beginning of training.
        """
        if state.is_local_process_zero:
            self.in_training = True
            self.start_time = time.time()
            self.max_steps = state.max_steps
            if os.path.exists(os.path.join(args.output_dir, LOG_FILE_NAME)):
                logger.warning("Previous log file in this folder will be deleted.")
                os.remove(os.path.join(args.output_dir, LOG_FILE_NAME))

    # 这个 Callback 中的三个方法的功能如下：
    # 1. `on_train_end`: 在训练结束时调用。如果是本地进程的第一个进程（`state.is_local_process_zero` 为真），则将 `in_training` 设为 `False`，
    #     并将 `cur_steps` 和 `max_steps` 重置为 0。这个方法的主要目的是在训练结束时进行一些清理工作。
    # 2. `on_substep_end`: 在梯度累积的过程中，在每个子步骤结束时调用。如果是本地进程的第一个进程，且 `self.runner` 不为空且 `self.runner.aborted` 为真，
    #     则设置 `control.should_epoch_stop` 和 `control.should_training_stop` 为真。这个方法的目的是在 `self.runner` 被中止时停止训练。
    # 3. `on_step_end`: 在每个训练步骤结束时调用。如果是本地进程的第一个进程，更新 `cur_steps` 为当前全局步数，然后调用 `timing` 方法计算已经经过的时间和剩余时间。
    #     如果 `self.runner` 不为空且 `self.runner.aborted` 为真，则同样设置 `control.should_epoch_stop` 和 `control.should_training_stop` 为真。
    #     这个方法的目的是在每个训练步骤结束时更新步数和时间，并在 `self.runner` 被中止时停止训练。
    # 这个 Callback 的作用主要是在训练的不同阶段进行一些操作，例如在训练结束时清理、在子步骤结束时处理中止操作、在每个步骤结束时更新时间和处理中止操作。
    def on_train_end(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        r"""
        Event called at the end of training.
        """
        if state.is_local_process_zero:
            self.in_training = False
            self.cur_steps = 0
            self.max_steps = 0

    def on_substep_end(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        r"""
        Event called at the end of an substep during gradient accumulation.
        """
        if state.is_local_process_zero and self.runner is not None and self.runner.aborted:
            control.should_epoch_stop = True
            control.should_training_stop = True

    def on_step_end(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        r"""
        Event called at the end of a training step.
        """
        if state.is_local_process_zero:
            self.cur_steps = state.global_step
            self.timing()
            if self.runner is not None and self.runner.aborted:
                control.should_epoch_stop = True
                control.should_training_stop = True

    # 这个 Callback 中的方法功能如下：
    # 1. `on_evaluate`: 在评估阶段结束后调用。如果是本地进程的第一个进程且不处于训练中，将 `cur_steps` 和 `max_steps` 重置为 0。这个方法的目的是在评估结束后进行一些清理工作。
    # 2. `on_predict`: 在成功预测后调用。如果是本地进程的第一个进程且不处于训练中，将 `cur_steps` 和 `max_steps` 重置为 0。这个方法的目的是在预测结束后进行一些清理工作。
    # 3. `on_log`: 在记录最后的日志后调用。如果不是本地进程的第一个进程，则直接返回。对于第一个进程，将当前步数、总步数、损失等信息写入 JSON 日志文件中。
    # 4. `on_prediction_step`: 在每个预测步骤后调用。如果是本地进程的第一个进程，且评估数据加载器具有长度（`has_length(eval_dataloader)` 为真）且不处于训练中，
    #     则更新当前步数 `cur_steps`，并通过 `timing` 方法计算已经经过的时间和剩余时间。
    # 这个 Callback 主要用于在不同的训练阶段进行一些操作，如在评估阶段和预测阶段结束后进行清理工作，记录日志等。
    def on_evaluate(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        r"""
        Event called after an evaluation phase.
        """
        if state.is_local_process_zero and not self.in_training:
            self.cur_steps = 0
            self.max_steps = 0

    def on_predict(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", *other, **kwargs):
        r"""
        Event called after a successful prediction.
        """
        if state.is_local_process_zero and not self.in_training:
            self.cur_steps = 0
            self.max_steps = 0

    def on_log(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs) -> None:
        r"""
        Event called after logging the last logs.
        """
        if not state.is_local_process_zero:
            return

        logs = dict(
            current_steps=self.cur_steps,
            total_steps=self.max_steps,
            loss=state.log_history[-1].get("loss", None),
            eval_loss=state.log_history[-1].get("eval_loss", None),
            predict_loss=state.log_history[-1].get("predict_loss", None),
            reward=state.log_history[-1].get("reward", None),
            learning_rate=state.log_history[-1].get("learning_rate", None),
            epoch=state.log_history[-1].get("epoch", None),
            percentage=round(self.cur_steps / self.max_steps * 100, 2) if self.max_steps != 0 else 100,
            elapsed_time=self.elapsed_time,
            remaining_time=self.remaining_time
        )
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, "trainer_log.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(logs) + "\n")

    def on_prediction_step(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        r"""
        Event called after a prediction step.
        """
        eval_dataloader = kwargs.pop("eval_dataloader", None)
        if state.is_local_process_zero and has_length(eval_dataloader) and not self.in_training:
            if self.max_steps == 0:
                self.max_steps = len(eval_dataloader)
            self.cur_steps += 1
            self.timing()
