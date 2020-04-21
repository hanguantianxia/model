# author:hjt
# contact: hanguantianxia@sina.com
# datetime:2020/4/7 22:53
# software: PyCharm

"""
This is the program
"""
import os
from collections import namedtuple

import matplotlib.pyplot as plt

from utils import time_clock


@time_clock
def read_train_data(filename, limit=None) :
    """
    read the train.txt and get the list of all work
    :param filename:
    :return:
    """
    corpus = []  # 存储所有对话的列表
    with open(filename, 'r', encoding='utf-8') as f:
        data = list()  # 存储一个对话的列表
        for line in f:
            if line == '\n':
                corpus.append(data)
                data = list()
                continue
            raw_data = line.strip().split('\t')

            processed_data = []
            for idx, item in enumerate(raw_data):
                try:
                    temp = eval(item)
                except:
                    temp = item
                finally:
                    if idx == 2 and type(temp)==int:
                        temp = str(temp)
                    processed_data.append(temp)
            data.append(processed_data)
            if limit is not None and len(corpus) > int(limit):
                break
    return corpus

Dialogue = namedtuple("Dialogue", ["role", 'responce', 'new_goal', 'current_goal', 'first_goal',
                                   'final_goal', 'goal_seq','knowledge','situation','usr_prof','history',"next_goal"])

def compute_knowledge_len(dial_tuple):
    goal_seq, knowledge, situation, usr_prof = dial_tuple.goal_seq, dial_tuple.knowledge, dial_tuple.situation, dial_tuple.usr_prof
    goal_seq_len = sum((len(item) -1 for item in goal_seq))
    kg_len = sum(( len(SPO.split()) for SPO in knowledge))
    s_len = 5
    usr_prof_len = sum((len(value) for _, value in usr_prof.items()))
    res = goal_seq_len + kg_len + s_len + usr_prof_len
    return res

if __name__ == '__main__':
    train_file = os.path.join(".", "work", "train.txt")
    train_data = read_train_data(train_file)

    knowledge_len = []
    conv_len = []
    for dial in train_data:
        tmp_len = 0
        cur_len = 0
        for dial_turn in dial:
            dial_tuple = Dialogue(*dial_turn)
            knowledge_len.append(compute_knowledge_len(dial_tuple))
            cur_len = max((len(dial_tuple.current_goal.split()),cur_len))
            tmp_len += len(str(dial_tuple.responce).split())
        conv_length = tmp_len + cur_len
        conv_len.append(conv_length)

    prob, left, rectangle = plt.hist(x=knowledge_len,
                                     color="steelblue",
                                     edgecolor="black",
                                     # density=True,
                                     stacked=True
                                     )
    plt.show()