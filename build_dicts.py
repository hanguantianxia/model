import code
import re
import copy
import json
from utils import json_saver

Segment_tpye = [
"S", "P", "O",                                      # knowledge seg
"date", "workday", "time", "location", "theme",     # situation                  # goal_seq
"current_goal",                                     # current goal
"user", "bot",                                      # conversation role
 # 'active poi',                                      # usr profile
 # 'active news',
 # 'active star',
 # 'active movie',
 # 'active food',
 # 'active music',
 # 'negative movie',
 # 'negative music',
 # 'name',
 # 'addr',
 # 'age',
 # 'sex',
 # 'reject',
 # 'profession',
'active 兴趣点',
 'active 新闻',
 'active 明星',
 'active 电影',
 'active 美食',
 'active 音乐',
 'negative 电影',
 'negative 音乐',
 '姓名',
 '居住地',
 '年龄区间',
 '性别',
 '拒绝',
 '职业状态'
]


def get_dict(file_path):
    utt_words = set()
    spo_p_words = set()
    goal_type_words = set()
    goal_entity_words = set()

    with open(file_path, "r", encoding='utf8') as train_data:
        train_samples = train_data.readlines()
        for sample in train_samples:
            if sample == '\n':
                continue
            else:

                sample = sample.split('\t')
                u_tmp = (sample[1]).split()
                goal_tmp = eval(sample[6])
                spo_tmp = eval(sample[7])
                cur_goal_tmp = sample[3].split()

                for w in u_tmp:
                    utt_words.add(w)
                for w in cur_goal_tmp:
                    utt_words.add(w)
                for triple in spo_tmp:
                    spo_p_words.add(triple[1])
                #code.interact(local= locals())
                for t, e in goal_tmp:
                    goal_type_words.add(t)
                    for e_i in e:
                        goal_entity_words.add(e_i)
    train_data.close()



    conv_word_dict = dict()
    conv_word_dict["[PAD]"] = len(conv_word_dict)
    conv_word_dict["[UNK]"] = len(conv_word_dict)
    conv_word_dict["[MASK]"] = len(conv_word_dict)
    conv_word_dict["[CLS]"] = len(conv_word_dict)
    conv_word_dict["[SEP]"] = len(conv_word_dict)
    conv_word_dict["<info>"] = len(conv_word_dict)

    # 不能用数字作为键,保存的时候会直接转化为str
    for i in range(100):
        conv_word_dict["space_%d"%i]= len(conv_word_dict)

    for idx, word in enumerate(utt_words):
        if word not in conv_word_dict:
            conv_word_dict[word] = len(conv_word_dict)


    p_word_dict = dict()
    p_word_dict["[UNK]"] = len(p_word_dict)

    for idx, word in enumerate(spo_p_words):
        p_word_dict[word] = len(p_word_dict)

    goal_type_dict = dict()
    goal_type_dict["[UNK]"] = len(goal_type_dict)
    for word in goal_type_words:
        goal_type_dict[word] = len(goal_type_dict)

    goal_entity_dict = dict()
    goal_entity_dict["[UNK]"] = len(goal_entity_dict)

    for idx, word in enumerate(goal_entity_words):
        goal_entity_dict[word] = len(goal_entity_dict)


    return conv_word_dict, p_word_dict, goal_type_dict, goal_entity_dict


def get_entities_dict():
    with open('work/entities_set', 'r', encoding='utf8') as f:
        entities = f.readlines()
        entities = eval(entities[0])

        entities_dict = dict()
        entities_dict["[UNK]"] = len(entities_dict)
        for idx, word in enumerate(entities):
            entities_dict[word] = len(entities_dict)

    return entities_dict


def file_saver(file_path, obj):
    with open(file_path, "w", encoding='utf8') as f:
        json.dump(obj, f)

def merge_vocab(vocab:dict, entity:dict, is_copy=False):
    if is_copy:
        word_dict = copy.deepcopy(vocab)
    else:
        word_dict = vocab

    for idx, key in enumerate(entity.keys()):
        if key not in word_dict:
            word_dict[key] = len(word_dict)
    return word_dict


if __name__ == "__main__":

    conv_word_dict, p_word_dict, goal_type_dict, goal_entity_dict = get_dict('work/train.txt')
    entities_dict = get_entities_dict()
    conv_word_dict = merge_vocab(conv_word_dict, entities_dict)

    Segment_dict = {"[PAD]":0,"[UNK]":1}

    for idx, item in enumerate(Segment_tpye):
        Segment_dict[item] = len(Segment_dict)

    for i in range(10):
        Segment_dict["goal_type"+str(i)] = len(Segment_dict)
        Segment_dict["goal_entity"+str(i)] = len(Segment_dict)

    file_saver("./work/Segment_dict.json", Segment_dict)
    file_saver("work/conv_word_dict.txt", conv_word_dict)
    file_saver("work/p_word_dict.txt", p_word_dict)
    file_saver("work/goal_type_dict.txt", goal_type_dict)
    file_saver("work/goal_entity_dict.txt", goal_entity_dict)
    file_saver("work/entities_dict.txt", entities_dict)








