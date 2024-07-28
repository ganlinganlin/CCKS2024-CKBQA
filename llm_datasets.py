import json
import os
import re
import sys
import json
import os


def load_data(train_dir, test_dir):
    train_question_list = []  # 问题列表
    sexpr_list = []  # 逻辑形式列表
    answer_list = []  # 答案列表
    valid_question_list = []  # 测试集问题列表

    k = 0
    train = open(train_dir, "r", encoding='utf-8')
    valid = open(test_dir, "r", encoding='utf-8')

    valid1 = valid.readline()  # 读取第一行
    while valid1:
        valid_question_list.append(valid1)  # 列表增加
        valid1 = valid.readline()  # 读取第一行

    train1 = train.readline()  # 读取第一行
    train2 = train.readline()  # 读取第二行
    train3 = train.readline()  # 读取第三行
    train4 = train.readline()  # 读取第四行
    while train2:
        # txt_data = eval(line) # 可将字符串变为元组
        train_question_list.append(train1)  # 列表增加
        sexpr_list.append(train2)  # 列表增加
        answer_list.append(train3)  # 列表增加
        train1 = train.readline()  # 读取第一行
        train2 = train.readline()  # 读取第二行
        train3 = train.readline()  # 读取第三行
        train4 = train.readline()  # 读取第四行
        k = k + 1
    print('训练集问题：', train_question_list[0:6])
    print('训练集问题数量：', len(train_question_list))
    print('逻辑形式数量：', len(sexpr_list))
    print('答案数量：', len(answer_list))
    print('测试集问题数量：', len(valid_question_list))
    return train_question_list, sexpr_list, answer_list, valid_question_list


def load_kb_data(train_dir, test_dir):
    train_question_sim_data = []
    test_question_sim_data = []

    k = 0
    train_sim = open(train_dir, "r", encoding='utf-8')
    valid_sim = open(test_dir, "r", encoding='utf-8')

    valid1 = valid_sim.readline()  # 读取第一行
    while valid1:
        test_question_sim_data.append(valid1)  # 列表增加
        valid1 = valid_sim.readline()  # 读取第一行

    test1 = train_sim.readline()  # 读取第一行
    while test1:
        train_question_sim_data.append(test1)  # 列表增加
        test1 = train_sim.readline()  # 读取第一行
        k = k + 1
    print('训练集问题：', train_question_sim_data[0:6])
    print('训练集知识库数量：', len(train_question_sim_data))
    print('测试集知识库数量：', len(test_question_sim_data))

    return train_question_sim_data, test_question_sim_data



def llm_datasets(train_dir, test_dir, train_question_list, train_output_list, valid_question_list):
    instruction = 'Generate a Logical Form that retrieves the information corresponding to the given question. \n'
    json_train_data = []
    json_test_data = []
    for i in range(len(train_question_list)):

        input = 'Question: { ' + train_question_list[i][:-1] + ' }'
        output = train_output_list[i][:-1] + '\t'
        json_train_data.append({"instruction": instruction, "input": input, "output": output, "history": []})
        # print(json_train_data)
    for i in range(len(valid_question_list)):
        input = 'Question: { ' + valid_question_list[i][:-1] + ' }'
        output = ''
        json_test_data.append({"instruction": instruction, "input": input, "output": output, "history": []})


    if not os.path.exists(os.path.dirname(train_dir)):
        os.makedirs(os.path.dirname(train_dir))
    with open(train_dir, "w", encoding="utf-8") as file:
        json.dump(json_train_data, file, ensure_ascii=False)
    print(train_dir+'已保存')

    if not os.path.exists(os.path.dirname(test_dir)):
        os.makedirs(os.path.dirname(test_dir))
    with open(test_dir, "w", encoding="utf-8") as file:
        json.dump(json_test_data, file, ensure_ascii=False)
    print(test_dir + '已保存')

train_question_list, sexpr_list, answer_list, valid_question_list = load_data("./train.txt","./valid_ques.txt")
# train_question_sim_data, test_question_sim_data = load_kb_data("reward-model-datasets/question_sim_train.txt","reward-model-datasets/question_sim_valid.txt")
llm_datasets('LLMs/data_tianchi/train/examples.json', 'LLMs/data_tianchi/test/examples.json', train_question_list, sexpr_list, valid_question_list)












