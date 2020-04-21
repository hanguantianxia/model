import os
import json
import time

from collections import namedtuple
from typing import List, Tuple, Dict

import numpy as np
import paddle.fluid as fluid

from utils import read_train_data, time_clock


Raw_data = namedtuple("Raw_data", ['role', 'dialogue', 'new_role', 'current_goal', 'first_goal',
                                   'final_goal', 'goal_seq', 'knowledge', 'situation', 'user_profile','history', 'next_goal'])
MASK_label = namedtuple("MASK_label", ['origin_token', "Segment_type", "pos"])

from train_data_demo import read_train_data
# from ERNIE.ernie.ernie_encoder_mine import Sent_Embed

KNOWLEDGE_MAX_LEN = 412
SEQ_MAX_LEN = 512
DATA_DIR = os.path.join(".","work")


def read_json(filename):
    """

    :param filename:
    :return:
    """
    with open(filename, 'r', encoding='utf8') as f:
        data = json.load(f)
    return data

class Dataset():
    """
    This module is the general module for dataset, it can finish following works:
    1. preprocess the raw work
    2. generate the padded index  work for trainning
    3. it do for train, dev
    4. need to another method to address test dataset


    """

    def __init__(self, data_path=os.path.join(DATA_DIR,"train.txt"),
                 vocab_path=os.path.join(DATA_DIR, "conv_word_dict.txt"),
                 SPO_vocab_path=os.path.join(DATA_DIR, "p_word_dict.txt"),
                 entities_vocab_path=os.path.join(DATA_DIR, "entities_dict.txt"),
                 segment_vacab_path=os.path.join(DATA_DIR, "Segment_dict.json"),
                 goaltype_vocab_path=os.path.join(DATA_DIR, "goal_type_dict.txt"),
                 goals_str_path=os.path.join(DATA_DIR, "goals_dict.json"),
                 CLS_token='[CLS]',
                 SEP_token='[SEP]',
                 UNK_token='[UNK]',
                 PAD_token='[PAD]',
                 MASK_token='[MASK]',
                 limit=None):
        """
        Todo:
        1. get dataset
        2. get the dict for word, entities, segment type
        3. get
        :param data_path:
        """
        # get raw dataset
        self.raw_dataset = read_train_data(data_path, limit=limit)

        self.word2idx = read_json(vocab_path)
        self.idx2word = {value:key for key, value in self.word2idx.items()}

        self.p_dict = read_json(SPO_vocab_path)
        self.id2p = {value:key for key, value in self.p_dict.items()}

        self.entity2id = read_json(entities_vocab_path)
        self.id2entity = {value:key for key, value in self.entity2id.items()}

        self.segtype2id = read_json(segment_vacab_path)
        self.id2segtype = {value:key for key, value in self.segtype2id.items()}

        self.goaltype2id = read_json(goaltype_vocab_path)
        self.id2goaltype = {value:key for key, value in self.goaltype2id.items()}

        self.goal2str = read_json(goals_str_path)



        # the special token
        self.CLS_id = self.word2idx[CLS_token]
        self.SEP_id = self.word2idx[SEP_token]
        self.PAD_id = self.word2idx[PAD_token]
        self.UNK_id = self.word2idx[UNK_token]
        self.MASK_id = self.word2idx[MASK_token]

        self.CLS_token = CLS_token
        self.SEP_token = SEP_token
        self.PAD_token = PAD_token
        self.UNK_token = UNK_token
        self.MASK_token = MASK_token

        # special token
        self.situation_type = ["date", "workday", "time", "location", "theme"]
        self.situation_sp_token = '<date>'
        self.kg_type = ["S", "P", "O"]
        self.task_type = 'gtsp'

        # the list without dialogue info
        self.flatten_dataset = [dial for item in self.raw_dataset for dial in item]
        self.bot_dial = [item for item in self.flatten_dataset if Raw_data(*item).role == "bot"]
        self.whole_dial = [item[-1] for item in self.raw_dataset]


        # work type
        self.int_type = np.int64

    def get_params(self,is_print=True):
        vocab_size = len(self.word2idx)
        goal_type_num = len(self.goaltype2id)
        goal_entity_num = len(self.entity2id)
        knowledge_s_num = len(self.entity2id)
        knowledge_p_num = len(self.p_dict)
        seg_num = len(self.segtype2id)

        params = {
            "VOCAB_SIZE":vocab_size,
            "GOAL_TYPE_NUM":goal_type_num,
            "GOAL_ENTITY_NUM":goal_entity_num,
            "KNOWLEDGE_S_NUM":knowledge_s_num,
            "KNOWLEDGE_P_NUM":knowledge_p_num,
            "TYPE_VOCAB_SIZE":seg_num
        }
        if is_print:
            for key, val in params.items():
                print(key,val)
        return params

    def _get_info(self, info, role,knowledge_size=KNOWLEDGE_MAX_LEN):
        """
        Extract information from the raw work
        :param info:
        :param role:
        :param knowledge_size:
        :return:
        """
        token_list = []
        segment_list = []
        if role == 'situation':
            for idx, item in enumerate(info):
                if item != []:
                    tmp = item[0]
                    token_list.append(tmp.strip() if idx!=0 else self.situation_sp_token)
                    segment_list.append(self.situation_type[idx])

        elif role == 'user_profile':
            for key, value in info.items():
                for item in value:
                    token_list.append(item)
                    segment_list.append(key)

        elif role == 'knowledge':
            sp_p = ["评论", "新闻"]
            sp_kg = [kg for kg in info if kg[1] in sp_p]
            # deal with the normal knowledge
            for kg in info:
                if kg[1] not in sp_p:
                    token_list.append(kg[0])
                    segment_list.append("S")
                    token_list.append(kg[1])
                    segment_list.append("P")
                    for word in kg[2].split():
                        token_list.append(word.strip())
                        segment_list.append("O")
                if len(token_list) > knowledge_size:
                    break
            # deal with special knowledge like news and issue
            np.random.seed(int(time.time()))
            random_id = np.random.permutation(len(sp_kg))

            for id in random_id:
                kg = sp_kg[id]
                tmp_token = [kg[0], kg[1]]
                tmp_seg = ["S", "P"]
                o_list = [item.strip() for item in kg[2].split()]
                tmp_seg = tmp_seg + ["O"] * len(o_list)
                tmp_token.extend(o_list)
                if len(tmp_token)  + len(token_list) > knowledge_size:
                    break
                token_list.extend(tmp_token)
                segment_list.extend(tmp_seg)

        elif role == 'goal_seq':
            new_goal_seq = [info[0], info[-2], info[-1]]
            num_list = list(range(len(info)))
            num_list = [num_list[0], num_list[-2], num_list[-1]]
            for i in range(len(info)-len(new_goal_seq)):
                new_goal_seq.append(info[1 + i])
                num_list.append(1+i)

            for idx, item in enumerate(new_goal_seq):
                token_list.append(item[0])
                segment_list.append("goal_type"+str(num_list[idx]))
                for entity in item[-1]:
                    if entity != '':
                        token_list.append(entity)
                        segment_list.append("goal_entity"+str(num_list[idx]))

        assert len(token_list) == len(segment_list), "token_list must have the same length as segment list"
        return token_list, segment_list

    def get_enc_item(self, raw_item,knowledge_size=KNOWLEDGE_MAX_LEN, max_length=SEQ_MAX_LEN):
        """
        pick the encoder need items in the raw item

        :return tokens item, type_idx_list
        """
        raw_item = Raw_data(*raw_item)
        need_info = [raw_item.situation, raw_item.user_profile, raw_item.knowledge, raw_item.goal_seq]
        role_list = ['situation',  'user_profile', 'knowledge', 'goal_seq']

        token_list = []
        segment_list = []
        for info, role in zip(need_info,role_list):
            tokens, segment = self._get_info(info, role, knowledge_size)
            token_list.extend(tokens)
            segment_list.extend(segment)

        assert len(token_list) == len(segment_list), "token_list and segment_list must have the same length"
        assert len(token_list) <= max_length, "the length of token list large than %d, please cheak! "%max_length
        return token_list, segment_list

    def sample_one(self, data):
        try:
            return [data[int(np.random.randint(0, len(data),1))]]
        except:
            return  []

    def mask_enc(self, token_list, segment_list)-> Tuple[List, List, List]:
        """
         Given the strategy, mask the encoder input, output hte mask_token_list, label, mask_pos_list
        :param token_list:
        :param segment_list:
        :return: Tuple[mask_token_list, label, mask_pos]
        mask_token_list: 4 types of task

        """
        S_prob = 0.18
        P_prob = 0.12
        Goal_type_prob = 0.35
        Goal_entities_prob = 0.35


        new_token_list = token_list.copy()
        P_index = []
        S_index = []
        entities_index = []
        goal_type_index = []
        for idx ,item in enumerate(segment_list):
            if item =="P":
                P_index.append(idx)
            elif item =="S":
                S_index.append(idx)
            elif item.find("goal_type") != -1:
                goal_type_index.append(idx)
            elif item.find("goal_entity") != -1:
                entities_index.append(idx)

        # sample from index work
        np.random.seed(int(time.time()))
        mask_S_index = [item for item in S_index if np.random.rand() < S_prob]
        mask_P_index = [item for item in P_index if np.random.rand() < P_prob]
        mask_goal_type_index = [item for item in goal_type_index if np.random.rand() < Goal_type_prob]
        mask_goal_entity_index = [item for item in entities_index if np.random.rand() < Goal_entities_prob]

        # cheak wheather there are a empty sample, if true, sample one from the origin work.
        if not mask_S_index:
            mask_S_index = self.sample_one(S_index)
        if not mask_P_index:
            mask_P_index = self.sample_one(P_index)
        if not mask_goal_type_index:
            mask_goal_type_index = self.sample_one(goal_type_index)
        if not mask_goal_entity_index:
            mask_goal_entity_index = self.sample_one(entities_index)

        mask_idx_list = mask_S_index + mask_P_index + mask_goal_type_index + mask_goal_entity_index

        label_list = []
        for mask_idx in mask_idx_list:
            label_list.append(MASK_label(token_list[mask_idx], segment_list[mask_idx], mask_idx))
            new_token_list[mask_idx] = self.MASK_token
        return new_token_list, label_list, mask_idx_list

    def convert_seg_token_into_id(self, tokens_list, segment_list, pad=True, max_len=SEQ_MAX_LEN):
        """
        convert tokens to ids and pad the token list with [pad]

        :param tokens_list:
        :param segment_list:
        :param pad: need pad
        :param max_len: pad max length
        :return: Tuple [mask_token_list, segment_list, pos_list, seq_len]
        """
        assert len(tokens_list) == len(segment_list), "tokens_list and segment_list must have the same length"
        seq_len = len(tokens_list)
        mask_token_list = [self.word2idx.get(item, self.UNK_id) for item in tokens_list]
        new_segment_list = [self.segtype2id.get(item) for item in segment_list]
        pos_list = list(range(1, seq_len+1))
        if pad:
            mask_token_list += [self.PAD_id] * (max_len - seq_len)
            new_segment_list += [self.PAD_id] * (max_len - seq_len)
            pos_list += [self.PAD_id] * (max_len - seq_len)

        assert len(mask_token_list) == len(new_segment_list), "tokens_list and segment_list must have the same length"
        assert len(mask_token_list) == max_len, "tokens_list must have been padded to the max length."
        return mask_token_list, new_segment_list, pos_list, seq_len

    def _get_label(self,Segment_type, origin_token, cmd):
        """

        :param Segment_type:
        :param origin_token:
        :param cmd:
        :return:
        """
        if cmd == 'g':
            if Segment_type.find("goal_type") != -1:
                return self.goaltype2id.get(origin_token, self.goaltype2id[self.UNK_token])
        elif cmd == 't':
            if Segment_type.find("goal_entity") != -1:
                return self.entity2id.get(origin_token, self.entity2id[self.UNK_token])
        elif cmd =='s':
            if Segment_type.find("S") != -1:
                return self.entity2id.get(origin_token, self.entity2id[self.UNK_token])
        elif cmd=='p':
            if Segment_type.find("P") != -1:
                return self.p_dict.get(origin_token, self.p_dict[self.UNK_token])

    def _parse_label(self, label:List[List[MASK_label]],task_cmd='gtsp'):
        """
        parse the label
        :param label:
        :param cmd: the commander for LM task, g=goal_type t=goal_entity, s=knowledge_s, p=knowledge_p
        use cmd to iter
        :return:
        """

        task = []
        for cmd in task_cmd:
            # different task
            assert cmd in self.task_type, "command must in %s." % self.task_type

            tmp_pos = []
            tmp_label = []
            for batch_id, batch in enumerate(label):
                for item in batch:
                    tmp = self._get_label(item.Segment_type, item.origin_token,cmd)
                    if tmp is not None:
                        tmp_pos.append(np.array([[batch_id, item.pos]], dtype=self.int_type))
                        tmp_label.append(tmp)
            if tmp_pos:
                tmp_pos = np.concatenate(tmp_pos, axis=0)
                tmp_label = np.array(tmp_label, dtype=self.int_type)
            else:
                # if there are empty task input the nan as the input to distinguish them from the normal label
                tmp_pos = np.array([0, 0])
                tmp_label = np.array([0])
                print("there are an emtpy example in knowledge s ")
            task.append((tmp_pos, tmp_label))

        assert len(task) == len(task_cmd)
        return task

    def enc_input_mask(self, seq_len, max_len=512):
        return [1]*seq_len + [0] * (max_len - seq_len)

    def generate_enc_batch(self, batch_size=32, pad=True,
                           max_len=SEQ_MAX_LEN, shuffle=True):
        """
        encoder input batch generator function

        :param batch_size:
        :param max_len:
        :param pad:
        :return:
        """
        if shuffle:
            id_list = np.random.permutation(len(self.flatten_dataset))
        else:
            id_list = list(range(len(self.flatten_dataset)))

        batch = []

        for item in id_list:
            raw_item = self.flatten_dataset[item]
            tokens_list, segment_list = self.get_enc_item(raw_item)
            mask_token_list, label, mask_pos = self.mask_enc(tokens_list, segment_list)
            mask_token_id_list, segment_list, pos_list, seq_len = \
                self.convert_seg_token_into_id(mask_token_list, segment_list, pad, max_len)
            batch.append([mask_token_id_list, segment_list, pos_list, seq_len, label])
            assert len(mask_token_id_list) == len(segment_list), "Error in this!"
            if len(batch) == batch_size:

                token_arr = np.concatenate([np.array(item[0], dtype=self.int_type).reshape(1, max_len) for item in batch])
                segment_arr = np.concatenate([np.array(item[1], dtype=self.int_type).reshape(1, max_len) for item in batch])
                pos_list_arr = np.concatenate([np.array(item[2], dtype=self.int_type).reshape(1, max_len) for item in batch])
                seq_len_arr = np.array([self.enc_input_mask(item[3],max_len) for item in batch], dtype=self.int_type).reshape(batch_size,max_len,1)

                label = [item[4] for item in batch]
                label = self._parse_label(label)


                goal_type_task, goal_entity_task, knowledge_s_task, knowledge_p_task = label
                goal_type_pos, goal_type_label = goal_type_task
                goal_entity_pos, goal_entity_label = goal_entity_task
                knowledge_s_pos, knowledge_s_label = knowledge_s_task
                knowledge_p_pos, knowledge_p_label = knowledge_p_task


                yield token_arr, segment_arr, pos_list_arr, seq_len_arr,\
                      goal_type_pos, goal_entity_pos, knowledge_s_pos,knowledge_p_pos,\
                      goal_type_label, goal_entity_label, knowledge_s_label, knowledge_p_label

                batch = []

    #############################################################################
    # decoder dataset generator
    #
    #
    #############################################################################
    def convert_token_to_id(self, tokens_list, pad=True, max_len=SEQ_MAX_LEN):
        """

        :param tokens_list:
        :param pad:
        :param max_len:
        :return:
        """
        seq_len = len(tokens_list)
        assert max_len > seq_len, "The token list is longer than the max_len.%d > %d"%(seq_len, max_len)

        token_id_list = [self.word2idx.get(item, self.UNK_id) for item in tokens_list]
        if pad:
            token_id_list = token_id_list + ["[PAD]"] * (max_len - len(token_id_list))

        return token_id_list, seq_len

    def convert_segment_to_id(self, segment_list, pad=True, max_len=SEQ_MAX_LEN):
        """

        :param segment_list:
        :param pad:
        :param max_len:
        :return:
        """
        seq_len = len(segment_list)
        assert max_len < seq_len, "The token list is longer than the max_len.%d > %d"%(seq_len, max_len)

        segment_id_list = [self.segtype2id.get(item, self.segtype2id.get(self.UNK_token)) for item in segment_list]
        if pad:
            segment_id_list = segment_id_list + ["[PAD]"] * (max_len - len(segment_id_list))

        return segment_id_list, seq_len

    def _get_lm_mask(self, range_list, mask_len=SEQ_MAX_LEN):
        """
        get the LM mask for decode training
        :param range_list:
        :return: lm mask and its lenght
        """
        tmp = np.zeros((1, mask_len))
        tmp[0, range_list[0]:range_list[1]] = 1
        return tmp, range_list[1] - range_list[0]

    def _get_lm_mask_stack(self, range_list:np.array, mask_len=SEQ_MAX_LEN):
        """

        :param range_list:
        :param mask_len:
        :return:
        """
        tmp = np.zeros((1, mask_len))
        seq_len = 0
        for range_item in range_list:
            item_output =  self._get_lm_mask(range_item)
            tmp += item_output[0]
            seq_len += item_output[1]

        return tmp, seq_len

    def get_dec_item(self, raw_item):
        """
        get the token list[current_goal,]
        :param raw_item:
        :return:
        """
        token_list = []
        segment_list = []

        raw_item = Raw_data(*raw_item)
        role, cur_goal, dial, history = [raw_item.role, raw_item.current_goal, raw_item.dialogue, raw_item.history]

        # load current goal
        cur_goal_list = cur_goal.split() + ["[SEP]"]
        token_list.extend(cur_goal_list)
        segment_list.extend(["current_goal"]*len(cur_goal_list))

        # load history
        for item in history:
            tmp = []
            tmp.append('[BOS]')
            tmp.extend(item[1].split())
            tmp.append('[CLS]')
            token_list.extend(tmp)
            segment_list.extend([item[0].lower()]*len(tmp))
        assert len(token_list) == len(segment_list)

        # load responce

        response =  dial.split() + ["[CLS]"]

        dial_list = ["[BOS]"] + response
        base = len(token_list)

        token_list.extend(dial_list)
        segment_list.extend([role.lower()]*len(dial_list))

        # return position position range list
        res_range_pos = [base, base+len(response)]

        # return next_goal
        next_goal = raw_item.next_goal
        if next_goal == []:
            next_goal = self.UNK_token
        else:
            next_goal = next_goal[0]

        next_goal_pos = len(token_list)-1

        assert len(token_list) == len(segment_list)
        # assert len(response) == len(res_pos)
        return token_list, segment_list, response, res_range_pos, next_goal, next_goal_pos

    def dec_self_attn_mask(self, seq_len, res_len, max_len=SEQ_MAX_LEN):
        """
        get the attn mask for self attention in decoder
        :param token_len:
        :param res_len:
        :return: numpy array of shape [1, max_len, max_len]
        """
        cg_his_len = seq_len - res_len
        cg_his_mask = np.array([self.enc_input_mask(cg_his_len, max_len)], dtype='int64')
        input_mask = np.array([self.enc_input_mask(seq_len, max_len)], dtype='int64')

        input_mask = np.dot(input_mask.T, cg_his_mask)
        input_mask[cg_his_len:cg_his_len+res_len,cg_his_len:cg_his_len+res_len]=np.tri(res_len, res_len)

        input_mask = input_mask.reshape(1,max_len, max_len)
        return input_mask

    def generate_dec_batch(self, batch_size=32, pad=True,
                           max_len=SEQ_MAX_LEN, shuffle=True):
        """

        :param batch_size:
        :param pad:
        :param max_len:
        :param shuffle:
        :return:
        """
        # only train role==bot dialogue
        dataset = self.bot_dial
        if shuffle:
            id_list = np.random.permutation(len(dataset))
        else:
            id_list = list(range(len(dataset)))

        batch = []
        for item in id_list:
            raw_item = dataset[item]
            try:
                token_list, segment_list, response, res_range_pos, next_goal, next_goal_pos = self.get_dec_item(raw_item)
            except:
                continue

            # convert to id
            token_id_list, segment_id_list, pos_list, seq_len = self.convert_seg_token_into_id(
                token_list, segment_list,pad, max_len)
            response_id_list, _ = self.convert_token_to_id(response, pad=False)
            next_goal_id = self.goaltype2id.get(next_goal)

            # get decoder self attention mask
            slf_attn_mask_arr = self.dec_self_attn_mask(seq_len, len(response)+1, SEQ_MAX_LEN)

            # convert to array
            token_id_arr = np.array([token_id_list], dtype=np.int64)
            segment_id_arr = np.array([segment_id_list], dtype=np.int64)
            pos_arr = np.array([pos_list], dtype=np.int64)

            # get LM mask and responce length
            res_range_pos = np.array(res_range_pos,dtype=np.int64)
            res_lm_mask, response_length = self._get_lm_mask(res_range_pos)
            res_lm_label = np.multiply(token_id_arr, self._get_lm_mask(res_range_pos + 1)[0])
            res_lm_label = np.array(res_lm_label, dtype=np.int64)


            batch.append((token_id_arr, segment_id_arr, pos_arr, slf_attn_mask_arr, # basic input
                          res_lm_mask, response_length, res_lm_label, # responce
                          (next_goal_pos,next_goal_id)))

            if len(batch) == batch_size:
                # basic input
                token_id_arr = np.concatenate([item [0] for item in batch], axis=0)
                segment_id_arr =  np.concatenate([item [1] for item in batch], axis=0)
                pos_arr =  np.concatenate([item [2] for item in batch], axis=0)
                slf_attn_mask_arr = np.concatenate([item [3] for item in batch], axis=0)

                # reponce input
                lm_mask_arr = np.concatenate([item[4] for item in batch], axis=0)
                response_length_arr =  np.array([item[5] for item in batch], dtype=np.float32)
                res_lm_label_arr = np.concatenate([item[6] for item in batch], axis=0)

                # goal type
                goal_list = [item [-1] for item in batch]
                goal_pos = []
                goal_type_list = []
                for batch_id, (pos, goal_type) in enumerate(goal_list):
                    goal_pos.append([batch_id, pos])
                    goal_type_list.append(goal_type)

                goal_pos = np.array(goal_pos, dtype=np.int64)
                goal_type_list = np.array(goal_type_list, dtype=np.int64)

                yield token_id_arr, segment_id_arr, pos_arr, slf_attn_mask_arr, \
                      lm_mask_arr, response_length_arr, res_lm_label_arr,\
                      goal_pos, goal_type_list
                batch = []

    def get_history(self, raw_item):
        """

        :param raw_item:
        :return:
        """

        token_list = []
        segment_list = []
        bot_range_list = []

        raw_item = Raw_data(*raw_item)
        history, role, dial = raw_item.history.copy(), raw_item.role, raw_item.dialogue
        history.append((role, dial))
        for role, dial in history:
            base = len(token_list)
            tmp = []
            tmp.append('[BOS]')
            tmp.extend(dial.split())
            bot_length = len(tmp)
            tmp.append('[CLS]')
            token_list.extend(tmp)
            segment_list.extend([role.lower()]*len(tmp))
            if role.lower() == 'bot':
                bot_range_list.append((base, base+bot_length))
        assert len(token_list) == len(segment_list)
        return token_list, segment_list, bot_range_list


    def generate_dec_pretrain_batch(self, batch_size=32, pad=True, max_len=SEQ_MAX_LEN, shuffle=True):
        """

        :param batch_size:
        :param pad:
        :param max_len:
        :param shuffle:
        :return:
        """
        dataset = self.whole_dial
        if shuffle:
            id_list = np.random.permutation(len(dataset))
        else:
            id_list = list(range(len(dataset)))

        batch = []
        for item in id_list:
            raw_item = dataset[item]
            try:
                token_list, segment_list, bot_range_list = self.get_history(raw_item)
            except:
                continue

            # convert to id
            token_id_list, segment_id_list, pos_list, seq_len = self.convert_seg_token_into_id(
                token_list, segment_list,pad, max_len)

            # get decoder self attention mask
            slf_attn_mask_arr = self.dec_self_attn_mask(seq_len, seq_len, SEQ_MAX_LEN)

            # convert basic info to array
            token_id_arr = np.array([token_id_list], dtype=np.int64)
            segment_id_arr = np.array([segment_id_list], dtype=np.int64)
            pos_arr = np.array([pos_list], dtype=np.int64)

            # get LM mask and responce length
            bot_range_list = np.array(bot_range_list,dtype=np.int64)
            res_lm_mask, response_length = self._get_lm_mask_stack(bot_range_list)
            # res_lm_label = np.multiply(token_id_arr, self._get_lm_mask_stack(bot_range_list + 1)[0])
            # res_lm_label = np.array(res_lm_label, dtype=np.int64)


            batch.append((token_id_arr, segment_id_arr, pos_arr, slf_attn_mask_arr, # basic input
                          res_lm_mask, response_length))

            if len(batch) == batch_size:
                # basic input
                token_id_arr = np.concatenate([item [0] for item in batch], axis=0)
                segment_id_arr =  np.concatenate([item [1] for item in batch], axis=0)
                pos_arr =  np.concatenate([item [2] for item in batch], axis=0)
                slf_attn_mask_arr = np.concatenate([item [3] for item in batch], axis=0)

                # reponce input
                lm_mask_arr = np.concatenate([item[4] for item in batch], axis=0)
                response_length_arr =  np.array([item[5] for item in batch], dtype=np.float32)

                lm_mask_ids = np.where(lm_mask_arr == 1)
                res_lm_label_arr = np.zeros((token_id_arr.shape))
                res_lm_label_arr[lm_mask_ids[0], lm_mask_ids[1]] = token_id_arr[lm_mask_ids[0], lm_mask_ids[1] + 1]


                yield token_id_arr, segment_id_arr, pos_arr, slf_attn_mask_arr, \
                      lm_mask_arr, response_length_arr, res_lm_label_arr
                batch = []

    def view_dec_batch(self, batch):
        """

        :param batch:
        :return:
        """
        token_id_arr, segment_id_arr, pos_arr, slf_attn_mask_arr, \
        lm_mask_arr, lm_len_arr, lm_lable_arr= batch

        src_ids = np.array(lm_mask_arr * token_id_arr, dtype=np.int64).tolist()
        src_token = [[self.idx2word.get(word) for word in sent if word!=0] for sent in src_ids]

        lm_mask_ids = np.where(lm_mask_arr==1)
        cheak_label_arr = np.zeros((token_id_arr.shape))
        cheak_label_arr[lm_mask_ids[0],lm_mask_ids[1]] = token_id_arr[lm_mask_ids[0],lm_mask_ids[1]+1]

        np.any(cheak_labels!=lm_lable_arr)


        return src_token


    @staticmethod
    def get_position_embed(max_seq_len=SEQ_MAX_LEN, d_model=768):
        """
        PE(pos;2i) = sin(pos/10000^(2i/d_model))
        PE(pos;2i+1) = cos(pos/10000^(2i/_model))

        :return: np.array([ max_len + 1, d_model])
        """
        position_encoding = np.array([
            [pos / np.power(10000, 2.0 * (j // 2) / d_model) for j in range(d_model)]
            for pos in range(max_seq_len)])
        # 偶数列使用sin，奇数列使用cos
        position_encoding[:, 0::2] = np.sin(position_encoding[:, 0::2])
        position_encoding[:, 1::2] = np.cos(position_encoding[:, 1::2])

        # np.save(fliename, position_encoding)
        return position_encoding









if __name__ == '__main__':


    dataset = Dataset(limit=10)
    raw_data = dataset.raw_dataset
    example = raw_data[1][0]
    raw_item = Raw_data(*example)

    token_list, segment_list = dataset.get_enc_item(raw_item)
    mask_token_list, label, mask_pos = dataset.mask_enc(token_list, segment_list)
    mask_token_list, new_segment_list, pos_list, seq_len = dataset.convert_seg_token_into_id(mask_token_list, segment_list)
    label_id = dataset._parse_label([label])
    gen = dataset.generate_enc_batch()
    batch = next(gen)
    len_list = []

    bot_item = dataset.bot_dial[0]
    token_list, segment_list, response, res_pos, next_goal, next_goal_pos = dataset.get_dec_item(bot_item)
    token_id_list, segment_id_list, pos_list, seq_len =  dataset.convert_seg_token_into_id(token_list, segment_list)

    dec_gen = dataset.generate_dec_batch(batch_size=32)
    token_id_arr, segment_id_arr, pos_arr, slf_attn_mask_arr, \
    lm_mask_arr, response_length_arr, res_lm_label_arr, \
    goal_pos, goal_type_list = next(dec_gen)
    dataset.get_position_embed()

    # test
    example = dataset.whole_dial[0]
    token_list, segment_list, range_list = dataset.get_history(example)



    dataset = Dataset(limit=None)
    gen = dataset.generate_dec_pretrain_batch(32)
    batch = next(gen)
    token_id_arr, segment_id_arr, pos_arr, slf_attn_mask_arr, \
    lm_mask_arr, response_length_arr, res_lm_label_arr = batch
    start = time.time()
    for batch in gen:
        input_mask = batch[4]
        if input_mask[input_mask>1].shape != (0,):
            print("error")
    end = time.time()
    print("a epoch cost %d s"%(end-start))
    # dataset = Dataset(limit=None)
    # dec_gen = dataset.generate_dec_batch(batch_size=32)
    # for item in dec_gen:
    #     pass
    ################################################################################################
    # test the input reader
    #
    ################################################################################################
    # sequence input work
    # token_ids = fluid.layers.data(name="token_ids", shape=[None, SEQ_MAX_LEN], dtype='int64')
    # segment_ids = fluid.layers.data(name="segment_ids", shape=[None, SEQ_MAX_LEN], dtype='int64')
    # pos_ids = fluid.layers.data(name="pos_ids", shape=[None, SEQ_MAX_LEN], dtype='int64')
    # input_length = fluid.layers.data(name='input_length', shape=[None, SEQ_MAX_LEN, 1], dtype='int64')
    #
    # # task work
    # goal_type_pos = fluid.layers.data(name="goal_type_pos", shape=[None, 2], dtype='int64')
    # goal_entity_pos = fluid.layers.data(name="goal_entity_pos", shape=[None, 2], dtype='int64')
    # knowledge_s_pos = fluid.layers.data(name="knowledge_s_pos", shape=[None, 2], dtype='int64')
    # knowledge_p_pos = fluid.layers.data(name="knowledge_p_pos", shape=[None, 2], dtype='int64')
    #
    # # task label
    # goal_type_label = fluid.layers.data(name="goal_type_label", shape=[None], dtype='int64')
    # goal_entity_label = fluid.layers.data(name="goal_entity_label", shape=[None], dtype='int64')
    # knowledge_s_label = fluid.layers.data(name="knowledge_s_label", shape=[None], dtype='int64')
    # knowledge_p_label = fluid.layers.data(name="knowledge_p_label", shape=[None], dtype='int64')
    #
    #
    # # data_loader
    # ITERABLE = True
    # BATCH_SIZE =32
    #
    # feed_list = [token_ids, segment_ids, pos_ids, input_length,
    #              goal_type_pos, goal_entity_pos,knowledge_s_pos,knowledge_p_pos,
    #              goal_type_label, goal_entity_label,knowledge_s_label,knowledge_p_label]
    #
    # places = fluid.CPUPlace()
    # data_loader = fluid.io.DataLoader.from_generator(feed_list=feed_list, capacity=10, iterable=ITERABLE)
    # data_loader.set_batch_generator(dataset.generate_enc_batch(batch_size=BATCH_SIZE), places=places)

    # exe = fluid.Executor(fluid.CPUPlace())
    # a = np.array([np.nan])
    # b = np.array([1], dtype=np.int64)
    # res = exe.run(fluid.default_main_program(), feed={"knowledge_p_label":a,"knowledge_s_label":b}, fetch_list=[res])
