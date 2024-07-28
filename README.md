

# CCKS2024——开放领域知识图谱问答评测

### 任务描述
- https://tianchi.aliyun.com/competition/entrance/532197/information
- 本任务属于开放领域的中文知识图谱自然语言问答任务，简称CKBQA （Chinese Knowledge Base Question Answering）。即输入一句中文问题，问答系统从给定知识库中选择若干实体或属性值作为该问题的答案。问题均为客观事实型，不包含主观因素。理解并回答问题的过程中可能需要进行实体识别、关系抽取、语义解析等子任务。这些任务的训练可以使用额外的公开的语料资源，但是最终的答案必须来自给定的知识库。
### 任务目标
- 输入，输入文件包含若干行中文问句。
- 输出，输出文件每一行对应一个问题的答案列表，列表内元素以制表符\t分隔。
- 输入样例

  q1: 故宫附近有哪些豪华酒店？

  q2: 《射雕英雄传》里黄蓉的爸爸用什么武器？
- 输出样例

  <北京王府井希尔顿酒店>\t<北京励骏酒店>\t<北京国际饭店>\t<北京东方君悦大酒店>

  <玉箫>

##  1. General Setup 

[//]: # (<h2>General Setup</h2>)
###  1.1 Environment Setup
- Ubuntu 22.04、CUDA 11.7
- Create a environment
  ```
  cd CCKS2024-CKBQA
  conda create -n kbqa388 python=3.8.8
  conda activate kbqa388
  pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
  pip install -r requirement.txt
  ```
- Download the flash-attention from [flash_attn-2.3.5+cu117torch1.13cxx11abiTRUE-cp38-cp38-linux_x86_64.whl](https://github.com/Dao-AILab/flash-attention/releases?page=2) to `princeton-nlp/`
  ```
  cd princeton-nlp
  pip install flash_attn-2.3.5+cu117torch1.13cxx11abiFALSE-cp38-cp38-linux_x86_64.whl
  mv nltk_data ../../
  ```


### 1.2 Large Language Models (LLMs)

- Download the LLMs from [LLaMa2-7b](https://huggingface.co/meta-llama/Llama-2-7b-hf) to `meta-llama/Llama-2-7b-hf/`
- Fine-tuning datasets for large language models
  ```
  cd CCKS2024-CKBQA
  python llm_datasets.py
  ```

## 2. Fine-tuning LLMs

### 2.1 Train and test LLMs for Logical Form Generation


- Train LLMs for Logical Form Generation (The checkpoint data will be saved as `Reading/logical-form-generation/checkpoint/`)
- Beam-setting LLMs for Logical Form Generation (The generated_predictions.jsonl will be saved as `Reading/logical-form-generation/`)
- Data processing (The beam_test_gen_statistics.json and beam_test_top_k_predictions.json will be saved as `Reading/logical-form-generation/test/`)
  ```
  cd CCKS2024-CKBQA
  CUDA_VISIBLE_DEVICES=0 nohup python -u LLMs/LLaMA/src/train_bash.py --stage sft --model_name_or_path meta-llama/Llama-2-7b-hf --do_train  --dataset_dir LLMs/data_tianchi --dataset train --template llama2  --finetuning_type lora --lora_target q_proj,v_proj --output_dir Reading/logical-form-generation/checkpoint --overwrite_cache --per_device_train_batch_size 4 --gradient_accumulation_steps 4  --lr_scheduler_type cosine --logging_steps 10 --save_steps 1000 --learning_rate 5e-5  --num_train_epochs 100.0  --plot_loss  --fp16 >> Sexpr_train_LLaMA2-7b_LoRA_epoch100.txt 2>&1 &
  ```
  ```
  CUDA_VISIBLE_DEVICES=1 nohup python -u LLMs/LLaMA/src/beam_output_eva.py --model_name_or_path meta-llama/Llama-2-7b-hf --dataset_dir LLMs/data_tianchi --dataset test --template llama2 --finetuning_type lora --checkpoint_dir Reading/logical-form-generation/checkpoint --num_beams 8 >> Sexpr_predbeam8_LLaMA2-7b_LoRA_epoch100.txt 2>&1 &
  ```




## 5. Retrieval & Evaluation

### 5.1 **Evaluate KBQA result with Retrieval**

- gStore http api 
  ```
  cd CCKS2024-CKBQA/python-api/example
  python POST-example.py
  ```


