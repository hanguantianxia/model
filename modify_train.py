import code

file_path = './train.txt'
f = open('./train_1.txt', 'w')

def str_tuple(*items, split_char='\t', split_string='\n'):
    """

    :param items:
    :return:
    """
    str_list = [str(item) for item in items]
    res = split_char.join(str_list)
    return res

with open(file_path, "r") as train_data:
    train_samples = train_data.readlines()
    spo_p_words = set()
    c = 1
    for sample in train_samples:
        if sample== '\n':
            f.write('\n')
        else:
            sample = sample.split('\t')
            u_b = sample[0]
            utt = sample[1]
            s_new_goal = sample[2]
            cur_goal = sample[3]
            first_goal = sample[4]
            final_goal = sample[5]
            all_goal = sample[6]
            knowledge = eval(sample[7])
            situation = sample[8]
            u_profile = sample[9]

            new_knowledge = []
            for kg in knowledge:
                if len(kg[1].split()) == 1 or kg[1] == '喜欢 的 新闻':
                    if kg[1].startswith('2018'):
                        kg[1] = '天气'
                    new_knowledge.append(kg)
                else:
                    new_kg = []
                    p = kg[1].split()
                    p_pre = p[:-1]
                    p_next = p[-1]
                    new_kg.append(kg[0] + ('').join(p_pre))
                    new_kg.append(p_next)
                    new_kg.append(kg[2])
                    new_knowledge.append(new_kg)

            tgt_str = str_tuple(u_b,utt,s_new_goal, cur_goal, first_goal, final_goal, all_goal,new_knowledge, situation, u_profile)
            f.write(tgt_str)

train_data.close()
f.close()


