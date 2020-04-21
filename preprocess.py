import json
import os
import re
import itertools
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np




from utils import *
src_base = os.path.join(".", "work")

def suball(string):
    pattern1 = re.compile(r"\(.*?\)", re.S)
    pattern2 = re.compile(r"\[.*?\]", re.S)
    pattern3 = re.compile(r' ')
    pattern4 = re.compile(r"『.*?』")
    res = pattern4.sub("", string)
    res1 = pattern1.sub("", res)
    res2 = pattern2.sub("", res1)
    res3 = pattern3.sub("", res2)
    return res3

def sub_info(string, token=""):
    pattern1 = re.compile(r' ')
    pattern2 = re.compile(r"『.*?』")
    pattern3 = re.compile(r'『.*?(?=,)')
    res = pattern1.sub("", string)
    res1 = pattern2.sub(token, res)
    res2 = pattern3.sub(token, res1)
    return res2

def get_goals_keywords(raw_goals):
    goals_key = [suball(item["goal"]) for item in train_data]
    split_goal = [item.split("-->") for item in raw_goals]
    key_word = [suball(goal) for goals in split_goal for goal in goals]
    return goals_key, split_goal, key_word

def sub_test_info(string, token=""):
    pattern1 = re.compile(r' ')
    pattern2 = re.compile(r"『.*?』")
    pattern3 = re.compile(r'『.*?(?=,)')
    pattern4 = re.compile(r"-->\.\.\.\.\.\.")

    res = pattern1.sub("", string)
    res1 = pattern2.sub(token, res)
    res2 = pattern3.sub(token, res1)
    res3 = pattern4.sub("",res2)
    return res3

def goals_sub(input):
    pattern = re.compile(r"\[[0-9]+\]", re.S)
    return pattern.sub("", input)



if __name__ == '__main__':
    tgt_base = set_base(__file__)
    filename = os.path.join(src_base, "train.txt")
    train_data = read_origin_data(filename)

    raw_goals = [item["goal"] for item in train_data]

    # 按照关键字级别进行处理
    goals_key, split_goal, key_word = get_goals_keywords(raw_goals)

    goals_len = [len(item) for item in split_goal]
    prob, left, rectangle = plt.hist(x=goals_len,
                                     color="steelblue",
                                     edgecolor="black",
                                     # density=True,
                                     stacked=True
                                     )
    plt.show()
    key_set = set(key_word)
    print("The number of key words: ", len(set(key_word)))
    print("The number of key words path: ", len(set(goals_key)))



    # 中颗粒度模板级别的goals分析
    goals_tmp = [sub_info(item, token="<info>") for item in raw_goals]
    # 单独的模板数量
    unq_goals_tmp_set = set(goals_tmp)
    unq_goals_tmp_list = list(unq_goals_tmp_set)

    # 建立模板和id之间的映射关系
    goals_dict = {item : i for i, item in enumerate(unq_goals_tmp_list)}
    goals_dict_re = {i : item for i, item in enumerate(unq_goals_tmp_list)}


    goals_ids = [goals_dict.get(item) for item in goals_tmp]

    goals_stat = {}
    for item in goals_ids:
        goals_stat[item] = goals_stat.get(item, 0) + 1

    stat_items = goals_stat.items()
    stat_items = sorted(stat_items, key=lambda x: x[1])
    threadthold = 10

    fliename = os.path.join(tgt_base, "view_goals_sort.txt")
    with open(fliename, 'w',encoding='utf-8') as f:
        for key,val in stat_items:
            item = goals_dict_re.get(key)
            f.write(str(val) + ": "+ item + '\n\n')


    fliename = os.path.join(tgt_base, "less_glob.txt")
    with open(fliename, 'w',encoding='utf-8') as f:
        for key,val in stat_items:
            if val < threadthold:
                item = goals_dict_re.get(key)
                f.write(str(val) + ": "+ item + '\n\n')

    fliename = os.path.join(tgt_base, "goals_path.txt")
    with open(fliename, 'w',encoding='utf-8') as f:
        for item in sorted(goals_dict_re.values()):
            f.write(item + '\n')


    # 寻找特殊的标记，即
    # 1: [1]
    # 寒暄(Bot主动，根据给定的 < info > 寒暄，第一句问候要带User名字，聊天内容不要与 < info > 矛盾，聊天要自然，不要太生硬)-->[2]
    # 提问(Bot主动问最喜欢谁的新闻？User回答: 最喜欢 < info > 的新闻)-->[3]
    # 新闻推荐(Bot主动，推荐 < info > 的新闻 < info >, User接受。需要聊2轮)-->[4]
    # 再见
    #
    # 1: [1]
    # 寒暄(Bot主动，根据给定的 < info > 寒暄，第一句问候要带User名字，聊天内容不要与 < info > 矛盾，聊天要自然，不要太生硬)-->[2]
    # 提问(Bot主动，最喜欢谁的新闻？User回答: 最喜欢 < info > 的新闻)-->[3]
    # 新闻推荐(Bot主动，推荐 < info > 的新闻 < info >。, User接受。需要聊2轮)-->[4]
    # 再见
    # 这两个的原句

    sp_item = [stat_items[2][0], stat_items[1][0]]
    sp_content = [goals_dict_re[item] for item in sp_item]

    sp_item_id = [i for i, item in enumerate(goals_tmp) if item in sp_content]
    sp_exm = [raw_goals[i] for i in sp_item_id]

    edit_dis = np.array([normal_leven(i, j) for i, j in  itertools.product(unq_goals_tmp_list, unq_goals_tmp_list, repeat=1)])
    res = list(itertools.product(unq_goals_tmp_list, unq_goals_tmp_list, repeat=1))

    # 处理测试集的goals
    test_file = os.path.join(src_base, "test_1.txt")
    test_data = read_origin_data(test_file)

    raw_goals_test = [item["goal"] for item in test_data]
    goals_tmp_test = [sub_test_info(item,token='<info>') for item in raw_goals_test]

    goals_tmp_test_item = [item.split("-->") for item in goals_tmp_test]

    goals_tmp_list_uniq = [item.split("-->") for item in unq_goals_tmp_list] #模板goals序列的list表示
    goals_item_uniq = set([i for item in goals_tmp_list_uniq for i in item])
    for item in goals_tmp_test_item:
        for i in item:
            if i not in goals_item_uniq:
                print(i)

    goals_item_uniq_not_num = set([goals_sub(i) for item in goals_tmp_list_uniq for i in item])
    fliename = os.path.join(tgt_base, "goals_item.txt")
    with open(fliename, 'w',encoding='utf-8') as f:
        for item in sorted(list(goals_item_uniq_not_num)):
            f.write(item + "\n")


    # 搜索方法进行查找
    res = []
    for item in goals_tmp_test_item:
        # 子集包含法查找
        tmp = []
        for tgt in goals_tmp_list_uniq:
            if set(item) <= set(tgt):
                tmp.append(tgt)
        res.append(tmp)
    res_len = [len(item) for item in res]
    not_found = np.sum((np.array(res_len) == 0))
    double_res = np.sum((np.array(res_len) > 2))
    print("The number of res is :",Counter(res_len))

    ### 检测多个结果之间的相似性(最小编辑距离)
    multi_res_relevant = []
    for item in res:
        tmp = [normal_leven(a, b) for a, b in itertools.combinations(item, 2)]
        multi_res_relevant.append(tmp)
    multi_res_relevant_max = [max(item) for item in multi_res_relevant if item != []]
    print(Counter(multi_res_relevant_max))
