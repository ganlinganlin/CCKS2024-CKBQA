# Inspired by: https://github.com/huggingface/transformers/blob/v4.29.2/examples/pytorch/summarization/run_summarization.py


# 这段代码主要是导入了一些 Python 模块和自定义模块。以下是对代码的逐行解释：
# 1. **导入模块和函数：**
#    - 从 `llmtuner.dsets` 模块导入 `get_dataset`、`preprocess_dataset` 和 `split_dataset` 函数。
#    - 从 `llmtuner.extras.constants` 模块导入 `IGNORE_INDEX` 常量。
#    - 从 `llmtuner.extras.misc` 模块导入 `get_logits_processor` 函数。
#    - 从 `llmtuner.extras.ploting` 模块导入 `plot_loss` 函数。
#    - 从 `llmtuner.tuner.core` 模块导入 `load_model_and_tokenizer` 函数。
#    - 从 `llmtuner.tuner.sft.metric` 模块导入 `ComputeMetrics` 类。
#    - 从 `llmtuner.tuner.sft.trainer` 模块导入 `CustomSeq2SeqTrainer` 类。
# 2. **条件导入：**
#    - 如果 `TYPE_CHECKING` 为真，即在类型检查的上下文中，导入额外的类型，包括 `TrainerCallback` 和一些自定义参数的类型。
# 这段代码主要是为了在后续的代码中使用这些模块、函数和类型。这些模块和函数看起来涉及到了自然语言处理（NLP）模型的微调、数据处理、训练等方面的功能。
from typing import TYPE_CHECKING, Optional, List
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainingArguments

from llmtuner.dsets import get_dataset, preprocess_dataset, split_dataset
from llmtuner.extras.constants import IGNORE_INDEX
from llmtuner.extras.misc import get_logits_processor
from llmtuner.extras.ploting import plot_loss
from llmtuner.tuner.core import load_model_and_tokenizer
from llmtuner.tuner.sft.metric import ComputeMetrics
from llmtuner.tuner.sft.trainer import CustomSeq2SeqTrainer

if TYPE_CHECKING:
    from transformers import TrainerCallback
    from llmtuner.hparams import ModelArguments, DataArguments, FinetuningArguments, GeneratingArguments




# 这段代码定义了一个名为 `run_sft` 的函数，该函数用于执行序列到序列（Seq2Seq）任务的微调（fine-tuning）操作。以下是对代码的逐行解释：
# 1. **函数定义：**
#    - 这是一个函数定义，名为 `run_sft`，接受一系列参数，包括模型参数（`model_args`）、数据参数（`data_args`）、
#    微调参数（`finetuning_args`）、生成参数（`generating_args`）以及回调函数列表（`callbacks`）。
def run_sft(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    generating_args: "GeneratingArguments",
    callbacks: Optional[List["TrainerCallback"]] = None
):

    # 2. **获取数据集并加载模型和分词器：**
    #    - 调用 `get_dataset` 函数获取数据集。
    #    - 调用 `load_model_and_tokenizer` 函数加载模型和分词器，其中 `stage="sft"` 表示在微调（fine-tuning）阶段。
    # 3. **预处理数据集：**
    #    - 调用 `preprocess_dataset` 函数对数据集进行预处理。
    dataset = get_dataset(model_args, data_args)
    model, tokenizer = load_model_and_tokenizer(model_args, finetuning_args, training_args.do_train, stage="sft")
    dataset = preprocess_dataset(dataset, tokenizer, data_args, training_args, stage="sft")

    # 4. **处理生成阶段的特殊设置：**
    if training_args.predict_with_generate:
        tokenizer.padding_side = "left" # use left-padding in generation# 在生成时使用左填充


    # 5. **设置数据收集器（DataCollator）：**
    #    - 创建一个用于 Seq2Seq 模型的数据收集器。
    # 6. **覆盖 Seq2SeqTrainer 的解码参数：**
    #    - 将一些解码参数覆盖到 `training_args`。
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        label_pad_token_id=IGNORE_INDEX if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    )

    # Override the decoding parameters of Seq2SeqTrainer
    training_args_dict = training_args.to_dict()
    training_args_dict.update(dict(
        generation_max_length=training_args.generation_max_length or data_args.cutoff_len,
        generation_num_beams=data_args.eval_num_beams or training_args.generation_num_beams
    ))
    training_args = Seq2SeqTrainingArguments(**training_args_dict)


    # 7. **初始化 Seq2SeqTrainer：**
    #    - 创建一个自定义的 Seq2SeqTrainer 实例。
    # 8. **为 `model.generate` 设置关键字参数：**
    #    - 设置用于 `model.generate` 的关键字参数。
    # Initialize our Trainer
    trainer = CustomSeq2SeqTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=callbacks,
        compute_metrics=ComputeMetrics(tokenizer) if training_args.predict_with_generate else None,
        **split_dataset(dataset, data_args, training_args)
    )

    # Keyword arguments for `model.generate`
    gen_kwargs = generating_args.to_dict()
    gen_kwargs["eos_token_id"] = [tokenizer.eos_token_id] + tokenizer.additional_special_tokens_ids
    gen_kwargs["pad_token_id"] = tokenizer.pad_token_id
    gen_kwargs["logits_processor"] = get_logits_processor()


    # 9. **训练阶段：**
    #    - 如果设置进行训练，调用 `trainer.train` 进行训练，并保存一些训练相关的指标。
    # 10. **评估阶段：**
    # 11. **预测阶段：**
    #     - 如果设置进行预测，调用 `trainer.predict` 进行预测，并保存相关的指标和预测结果。
    # 这个函数主要负责运行序列到序列（Seq2Seq）任务的微调，包括训练、评估和预测阶段，并对各个阶段的指标进行记录和保存。
    # Training
    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()
        trainer.save_model()
        if trainer.is_world_process_zero() and model_args.plot_loss:
            plot_loss(training_args.output_dir, keys=["loss", "eval_loss"])

    # Evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate(metric_key_prefix="eval", **gen_kwargs)
        if training_args.predict_with_generate: # eval_loss will be wrong if predict_with_generate is enabled
            metrics.pop("eval_loss", None)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Predict
    if training_args.do_predict:
        predict_results = trainer.predict(dataset, metric_key_prefix="predict", **gen_kwargs)
        if training_args.predict_with_generate: # predict_loss will be wrong if predict_with_generate is enabled
            predict_results.metrics.pop("predict_loss", None)
        trainer.log_metrics("predict", predict_results.metrics)
        trainer.save_metrics("predict", predict_results.metrics)
        trainer.save_predictions(predict_results)
