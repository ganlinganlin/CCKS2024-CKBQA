# -*- coding: UTF-8 -*-
"""
# Filename: POST-example.py
# Author: suxunbin
# Last Modified: 2019-5-15 18:26
# Description: a simple POST-example of python API
"""
import sys
import os
sys.path.append('../src')
import GstoreConnector
import ast
from pathlib import Path

# before you run this example, make sure that you have started up ghttp service (using bin/ghttp port)
IP = "127.0.0.1"
Port = 9000
httpType = "ghttp"
username = "root"
password = "123456"
# sparql = "select ?x where \
#                  { \
#                      ?x  <rdf:type> <ub:UndergraduateStudent>. \
#                      ?y    <ub:name> <Course1>. \
#                      ?x    <ub:takesCourse>  ?y. \
#                      ?z   <ub:teacherOf>    ?y. \
#                      ?z    <ub:name> <FullProfessor1>. \
#                      ?z    <ub:worksFor>    ?w. \
#                      ?w    <ub:name>    <Department0>. \
#                  }"
# filename = "res.txt"
#
# # start a gc with given IP, Port, username and password
# gc =  GstoreConnector.GstoreConnector(IP, Port, username, password, http_type=httpType)



pwd_dir = os.getcwd()
print(pwd_dir)
parent_dir = os.path.dirname(pwd_dir)
parent_dir = os.path.dirname(parent_dir)
print(parent_dir)

predict_answer_list = []
predict_answer_k = 0
f = open(parent_dir + "/Reading/logical-form-generation/evaluation_beam8_test/generated_predictions.jsonl", "r",encoding='utf-8')
predict_answer_line1 = f.readline() # 读取第一行
predict_answer_line1 = ast.literal_eval(predict_answer_line1)
# print(predict_answer_line1,type(predict_answer_line1))
predict_answer = predict_answer_line1['predict'][0]
# print(predict_answer)
while predict_answer:
    predict_answer_list.append(predict_answer) # 列表增加
    predict_answer_line1 = f.readline()  # 读取第一行
    if len(predict_answer_line1) != 0:
        predict_answer_line1 = ast.literal_eval(predict_answer_line1)
        predict_answer = predict_answer_line1['predict'][0]
    else:
        predict_answer = False
    predict_answer_k = predict_answer_k + 1
print('predict_answer_k：',predict_answer_k)
predict_answer_list1 = predict_answer_list[0:200]
predict_answer_list2 = predict_answer_list[200:400]
predict_answer_list3 = predict_answer_list[400:600]
predict_answer_list4 = predict_answer_list[600:800]
predict_answer_list5 = predict_answer_list[800:1000]
predict_answer_list6 = predict_answer_list[1000:len(predict_answer_list)]


# #Llama3-Chinese-8B
# sexpr_list = []
# sexpr_k = 0
# f = open(parent_dir + "/Reading/00_result-sexpr.txt", "r",encoding='utf-8')
# sexpr_line1 = f.readline() # 读取第一行
# while sexpr_line1:
#     sexpr_list.append(sexpr_line1) # 列表增加
#     sexpr_line1 = f.readline()  # 读取第一行
#     sexpr_k = sexpr_k + 1
# # print(predict_answer_list)
# print('sexpr_k：',sexpr_k)
# sexpr_list1 = sexpr_list[0:200]
# sexpr_list2 = sexpr_list[200:400]
# sexpr_list3 = sexpr_list[400:600]
# sexpr_list4 = sexpr_list[600:800]
# sexpr_list5 = sexpr_list[800:1000]
# sexpr_list6 = sexpr_list[1000:len(sexpr_list)]


gc = GstoreConnector.GstoreConnector("pkubase.gstore.cn", 20024, "root", "123456")
list1 = [
# 'select?x where { <世界上第一台电子多用途计算机>	<重量>?x. }',
'select?y where { <凉宫春日_（轻小说《凉宫春日系列》中女主角）> <社团> ?y. }',
'select ?x where { <商朝> <灭亡> ?x. }\n',
'select ?y where { <亚瑟·叔本华> <信仰> ?y. }\n',
'select ?x where { ?x <灭亡> <商朝>. }\n',
'select ?y where { ?y <信仰> <亚瑟·叔本华>. }\n',
'select ?x where { <商朝> <灭亡> ?x. }',
'select ?y where { <亚瑟·叔本华> <信仰> ?y. }',
'select ?x where { <2008年北京奥运会> <口号> ?x. }',
'select ?x where { <大连理工大学> <校歌> ?x . }',
]

res_list = []
answer_right = 0
answer_wrong = 0
question_k = 0
for sparql_i in predict_answer_list:
    question_k = question_k +1
    print('问题：', question_k)

    res = gc.query("pkubase", "json", sparql_i)
    # print(res)
    user_dict = ast.literal_eval(res)
    print(user_dict, type(user_dict))

    # print(user_dict['results']['bindings'])
    if user_dict['StatusCode']:
        head_vars = 'error'
        answer_str = 'null01\t'
        res_list.append(answer_str)
        answer_wrong = answer_wrong + 1
    else:
        head_vars = user_dict['head']['vars'][0]

        if len(user_dict['results']['bindings']) == 0:
            print(len(user_dict['results']['bindings']))
            answer_str = 'null02\t'
            answer_wrong = answer_wrong + 1
        else:
            answer_str = ''
            for answer in user_dict['results']['bindings']:
                # print(answer)
                if answer[head_vars]['type'] == 'uri':
                    answer_str = answer_str + str('<') + str(answer[head_vars]['value']) + str('>') + '\t'
                else:
                    answer_str = answer_str + str('"') + str(answer[head_vars]['value']) + str('"') + '\t'

            answer_right = answer_right + 1
            print('已找到答案的数量：', answer_right)
        res_list.append(answer_str)

print('未找到答案的数量：',answer_wrong)
print('已找到答案的数量：', answer_right)


result_test_file = open(parent_dir + '/Result/河工大智慧图谱_result.txt','a')
for i in range(len(res_list)) :
    result_test_file.write(res_list[i] + '\n')
result_test_file.close()
print(parent_dir + '/Result/河工大智慧图谱_result.txt已保存')