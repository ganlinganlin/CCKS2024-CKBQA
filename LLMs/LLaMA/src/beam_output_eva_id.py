
# 这段代码看起来是一个评估脚本，用于评估聊天模型在给定数据集上的性能。以下是一些关键点的解释：
# 1. **导入模块:**
#    - 通过 `from llmtuner import ChatModel` 导入聊天模型。
#    - `import json` 用于处理 JSON 数据。
#    - `from tqdm import tqdm` 用于在循环中显示进度条。
#    - `import random` 用于随机打乱数据。
#    - `import re` 用于正则表达式操作。
#    - `import os` 用于文件路径和目录操作。
#    - 从 `llmtuner.tuner.core` 中导入了 `get_infer_args` 函数，该函数用于获取推断时的参数。
from llmtuner import ChatModel
import json
from tqdm import tqdm
import random
import re
import os
from llmtuner.tuner.core import get_infer_args

# model_args, data_args, _, generating_args = get_infer_args()
# print(str(generating_args.num_beams))
# print(generating_args.num_beams)

def main():
    # 2. **获取推断参数:**
    #    - 使用 `get_infer_args` 函数获取推断所需的模型参数、数据参数等。
    # 3. **初始化聊天模型:**
    #    - 创建了 `ChatModel` 类的实例，用于生成聊天响应。

    model_args, data_args, _, generating_args = get_infer_args()
    chat_model = ChatModel()
    output_data = []
    # 4. **循环读取数据:**
    #    - 打开给定数据集的 JSON 文件，循环遍历每个数据项。
    #    - 将数据项中的指令和输入合并为查询（`query`）。
    #    - 使用 `chat_model.chat_beam(query)` 生成聊天响应。
    # 5. **检查匹配:**
    #    - 对每个生成的响应进行匹配，检查是否与标签匹配。
    #    - 统计匹配的行数和将要匹配的行数。


    with open(os.path.join(data_args.dataset_dir,data_args.dataset,'examples.json'), 'r', encoding='utf-8') as f:
        json_data = json.load(f)
        # random.shuffle(json_data)
        total_lines = 0
        matched_lines = 0
        will_matched_lines = 0

        # 2. 读取每一行
        for data in tqdm(json_data):
            total_lines += 1
            query = data['instruction']+data['input']
            predict = chat_model.chat_beam(query)
            # print('1',predict)
            predict = [p[0] for p in predict]
            # print('2',predict)
            output_data.append({'question':data['input'],'label':data['output'],'predict':predict})
            for p in predict:
                # 4. 检查"predict_label"和"predict"的值是否相等
                if data['output'] == p:
                    matched_lines += 1
                    break
            for p in predict:
                # 5. 检查"predict_label"和"predict"的值是否相等
                if re.sub(r'\[.*?\]', '', data['output']) == re.sub(r'\[.*?\]', '', p):
                    will_matched_lines += 1
                    break


    # 5. 计算相等的行的数量
    print(f"Total lines: {total_lines}")
    print(f"Matched lines: {matched_lines}")
    print(f"Will Matched lines: {will_matched_lines}")

    # 6. 计算相等行的占比
    percentage = (matched_lines / total_lines) * 100
    print(f"Percentage of matched lines: {percentage:.2f}%")
    # 6. 计算相等行的占比
    will_percentage = (will_matched_lines / total_lines) * 100
    print(f"Percentage of will matched lines: {will_percentage:.2f}%")

    # 6. **输出结果:**
    #    - 打印总行数、匹配行数、将要匹配行数以及匹配行的百分比和将要匹配行的百分比。
    # 7. **保存输出:**
    #    - 将输出数据保存为 JSONLines 文件。
    # 8. **文件路径和目录操作:**
    #    - 构建了输出文件的路径，并在必要时创建目录。

    output_jsonl = 'evaluation_beam{}_{}/generated_predictions.jsonl'.format(str(generating_args.num_beams),str(data_args.dataset))
    output_dir = os.path.join(os.path.dirname(model_args.checkpoint_dir[0]), output_jsonl)
    # output_dir = os.path.join(os.path.dirname(model_args.checkpoint_dir[0]),'evaluation_beam/generated_predictions.jsonl')
    if not os.path.exists(os.path.dirname(output_dir)):
        os.makedirs(os.path.dirname(output_dir))
    with open(output_dir, 'w') as f:
        for item in output_data:
            json_string = json.dumps(item)
            f.write(json_string + '\n')


if __name__ == "__main__":
    main()
    # 总体而言，这个脚本用于评估聊天模型在给定数据集上的性能，包括行匹配和百分比。
