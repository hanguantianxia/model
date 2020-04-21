import code
import re


def get_dict(file_path):
    utt_words = set()
    spo_p_words = set()
    goal_type_words = set()
    goal_entity_words = set()

    with open(file_path, "r") as train_data:
        train_samples = train_data.readlines()
        for sample in train_samples:
            if sample == '\n':
                continue
            else:

                sample = sample.split('\t')
                u_tmp = (sample[1]).split()
                goal_tmp = eval(sample[6])
                spo_tmp = eval(sample[7])

                for w in u_tmp:
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
    for i in range(100):
        conv_word_dict[i]= i
    for idx, word in enumerate(utt_words):
        conv_word_dict[word] = idx + 100
    conv_word_dict["UNK"] = len(conv_word_dict)
    conv_word_dict["CLS"] = len(conv_word_dict)

    p_word_dict = dict()
    for idx, word in enumerate(spo_p_words):
        p_word_dict[word] = idx
    p_word_dict["UNK"] = len(p_word_dict)

    goal_type_dict = dict()
    for idx, word in enumerate(goal_type_words):
        goal_type_dict[word] = idx
    goal_type_dict["UNK"] = len(goal_type_dict)

    goal_entity_dict = dict()
    for idx, word in enumerate(goal_entity_words):
        goal_entity_dict[word] = idx
    goal_entity_dict["UNK"] = len(goal_entity_dict)


    return conv_word_dict, p_word_dict, goal_type_dict, goal_entity_dict


def get_entities_dict():
    with open('./entities_set', 'r') as f:
        entities = f.readlines()
        entities = eval(entities[0])

        entities_dict = dict()
        for idx, word in enumerate(entities):
            entities_dict[word] = idx
        entities_dict["UNK"] = len(entities_dict)

    return entities_dict


def file_saver(file_path, obj):
    with open(file_path, "w") as f:
        f.write(str(obj))
        f.close()


if __name__ == "__main__":

    conv_word_dict, p_word_dict, goal_type_dict, goal_entity_dict = get_dict('./train.txt')
    entities_dict = get_entities_dict()

    file_saver("./process/conv_word_dict.txt", conv_word_dict)
    file_saver("./process/p_word_dict.txt", p_word_dict)
    file_saver("./process/goal_type_dict.txt", goal_type_dict)
    file_saver("./process/goal_entity_dict.txt", goal_entity_dict)
    file_saver("./process/entities_dict.txt", entities_dict)








