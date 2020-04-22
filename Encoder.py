# author:hjt
# contact: hanguantianxia@sina.com
# datetime:2020/4/16 10:44
# software: PyCharm

"""

"""

import os
import json
import six
import numpy as np
import logging
import time

import paddle.fluid as fluid
from paddle.fluid.layers import core
from functools import partial

from transformer_encoder import encoder
from batch_generator import Dataset
from utils import *

#####################################################
# Model Parameter
HIDDEN_SIZE = 768
NUM_HIDDEN_LAYERS = 6
NUM_ATTENTION_HEADS = 6
VOCAB_SIZE = 15416
MAX_POSITION_EMBEDDINGS = 512
TYPE_VOCAB_SIZE = 47
HIDDEN_ACT = 'relu'
HIDDEN_DROPOUT_PROB = 0.
ATTENTION_PROBS_DROPOUT_PROB = 0.
GOAL_TYPE_NUM = 39
GOAL_ENTITY_NUM = 656
KNOWLEDGE_S_NUM = 656
KNOWLEDGE_P_NUM = 33
SEQ_MAX_LEN = 512
###########################################################################
# Train Parameters
BATCH_SIZE = 1
EPOCH_NUM = 10
PRINT_BATCH = 10
SAVE_BATCH = 100
LOAD_PERSISTABLE = False
LOAD_PERSISTABLE_FILE = "dec_model_epoch_5_batch_200.pers"
LOAD_VARS = False
LOAD_VARS_FILE = "dec_model_epoch_4_batch_0.vars"
LOAD_MODEL = True
LOAD_MODEL_FILE = "enc_model_epoch_0_batch_0model_stage_0.npz"
LOAD_OPT = False
LOAD_OPT_FILE = "enc_model_epoch_0_batch_0opt_state_stage_0.npz"

TRAIN_STAT_PATH = "training_msg.json"
MAX_SAVE = 12
LIMIT = 10
USE_CUDA = None

def check_params(dataset:Dataset):
    params = dataset.get_params()
    assert VOCAB_SIZE == params["VOCAB_SIZE"], "Parameter Error, %s shoud be %d , but it is %d. Please cheak!"%\
                                               ("VOCAB_SIZE", params["VOCAB_SIZE"], VOCAB_SIZE)
    assert GOAL_TYPE_NUM == params["GOAL_TYPE_NUM"], "Parameter Error, %s shoud be %d , but it is %d. Please cheak!"%\
                                               ("GOAL_TYPE_NUM", params["GOAL_TYPE_NUM"], GOAL_TYPE_NUM)
    assert GOAL_ENTITY_NUM == params["GOAL_ENTITY_NUM"], "Parameter Error, %s shoud be %d , but it is %d. Please cheak!"%\
                                               ("GOAL_ENTITY_NUM", params["GOAL_ENTITY_NUM"], GOAL_ENTITY_NUM)
    assert KNOWLEDGE_S_NUM == params["KNOWLEDGE_S_NUM"], "Parameter Error, %s shoud be %d , but it is %d. Please cheak!"%\
                                               ("KNOWLEDGE_S_NUM", params["KNOWLEDGE_S_NUM"], KNOWLEDGE_S_NUM)
    assert KNOWLEDGE_P_NUM == params["KNOWLEDGE_P_NUM"], "Parameter Error, %s shoud be %d , but it is %d. Please cheak!"%\
                                               ("KNOWLEDGE_P_NUM", params["KNOWLEDGE_P_NUM"], KNOWLEDGE_P_NUM)
    assert TYPE_VOCAB_SIZE == params["TYPE_VOCAB_SIZE"], "Parameter Error, %s shoud be %d , but it is %d. Please cheak!"%\
                                               ("TYPE_VOCAB_SIZE", params["TYPE_VOCAB_SIZE"], TYPE_VOCAB_SIZE)
    check_info = "Parameter checked."
    print(check_info)
    return check_info

#####################################################
class Model_Config():
    def __init__(self, filename=None):
        if not filename:
            self.config = {
                "HIDDEN_SIZE":HIDDEN_SIZE,
                "NUM_HIDDEN_LAYERS" :NUM_HIDDEN_LAYERS,
                "NUM_ATTENTION_HEADS" :NUM_ATTENTION_HEADS,
                "VOCAB_SIZE" :VOCAB_SIZE,
                "MAX_POSITION_EMBEDDINGS" :MAX_POSITION_EMBEDDINGS,
                "TYPE_VOCAB_SIZE" :TYPE_VOCAB_SIZE,
                "HIDDEN_ACT" :HIDDEN_ACT,
                "HIDDEN_DROPOUT_PROB" :HIDDEN_DROPOUT_PROB,
                "ATTENTION_PROBS_DROPOUT_PROB" :ATTENTION_PROBS_DROPOUT_PROB,
                "GOAL_TYPE_NUM" :GOAL_TYPE_NUM,
                "GOAL_ENTITY_NUM" :GOAL_ENTITY_NUM,
                "KNOWLEDGE_S_NUM" :KNOWLEDGE_S_NUM,
                "KNOWLEDGE_P_NUM" :KNOWLEDGE_P_NUM,
                "SEQ_MAX_LEN" :SEQ_MAX_LEN
            }
        else:
            self.config = self._read(filename)

    def __getitem__(self, key):
        return self.config[key]

    def _read(self, filename):
        return read_json(filename)

    def save(self,filename):
        json_saver(self.config, filename)


class Encoder(object):
    def __init__(self,
                 src_ids,
                 position_ids,
                 sentence_ids,
                 input_mask,
                 config:Model_Config,
                 weight_sharing=True,
                 use_fp16=False):
        self._emb_size = config["HIDDEN_SIZE"]
        self._n_layer =config["NUM_HIDDEN_LAYERS"]
        self._n_head =config["NUM_ATTENTION_HEADS"]
        self._voc_size = config["VOCAB_SIZE"]
        self._max_position_seq_len = config["MAX_POSITION_EMBEDDINGS"]
        self._sent_types = config["TYPE_VOCAB_SIZE"]
        self._hidden_act = config['HIDDEN_ACT']
        self._prepostprocess_dropout = config["HIDDEN_DROPOUT_PROB"]
        self._attention_dropout = config["ATTENTION_PROBS_DROPOUT_PROB"]
        self._weight_sharing = weight_sharing
        # name
        self.encoder_name = "encoder"
        self._word_emb_name = "enc_word_embedding"
        self._pos_emb_name = "enc_pos_embedding"
        self._seg_emb_name = "enc_seg_embedding"

        self._dtype = "float32"
        self._inttype = 'int32'
        # task parameters
        self.goal_type_num = config["GOAL_TYPE_NUM"]
        self.goal_entity_num = config["GOAL_ENTITY_NUM"]
        self.knowledge_s_num = config["KNOWLEDGE_S_NUM"]
        self.knowledge_p_num = config["KNOWLEDGE_P_NUM"]

        # self._param_initializer = fluid.initializer.TruncatedNormal(
        #     scale=config['initializer_range'])

        # parameter
        self.pos_embed = Dataset.get_position_embed(SEQ_MAX_LEN, HIDDEN_SIZE)


        self._build_model(src_ids, position_ids,
                          sentence_ids, input_mask)

    def _build_model(self, src_ids, position_ids, sentence_ids, input_mask):
        """
        padding id in vocabulary must be set to 0

        :param src_ids:
        :param position_ids:
        :param sentence_ids:
        :param pad_matrix:
        :return:
        """
        emb_out = fluid.embedding(
            input=src_ids,
            size=[self._voc_size, self._emb_size],
            dtype=self._dtype,
            param_attr=fluid.ParamAttr(
                name=self._word_emb_name),  # , initializer=self._param_initializer),
            is_sparse=False)

        position_emb_out = fluid.embedding(
            input=position_ids,
            size=[self._max_position_seq_len, self._emb_size],
            dtype=self._dtype,
            param_attr=fluid.ParamAttr(
                name=self._pos_emb_name)  # ,# initializer=self._param_initializer)
        )

        sent_emb_out = fluid.embedding(
            sentence_ids,
            size=[self._sent_types, self._emb_size],
            dtype=self._dtype,
            param_attr=fluid.ParamAttr(
                name=self._seg_emb_name)  # , initializer=self._param_initializer)
        )

        emb_out = emb_out + position_emb_out
        emb_out = emb_out + sent_emb_out

        input_mask = fluid.layers.cast(x=input_mask, dtype=self._dtype)
        self_attn_mask = fluid.layers.matmul(
            x=input_mask, y=input_mask, transpose_y=True)

        self_attn_mask = fluid.layers.scale(
            x=self_attn_mask, scale=1000000.0, bias=-1.0, bias_after_scale=False)
        # print("self_attn_mask",self_attn_mask.shape)

        n_head_self_attn_mask = fluid.layers.stack(
            x=[self_attn_mask] * self._n_head, axis=1)
        # print("n_head_self_attn_mask",n_head_self_attn_mask.shape)
        n_head_self_attn_mask.stop_gradient = True

        self._enc_out = encoder(
            enc_input=emb_out,
            attn_bias=n_head_self_attn_mask,
            n_layer=self._n_layer,
            n_head=self._n_head,
            d_key=self._emb_size // self._n_head,
            d_value=self._emb_size // self._n_head,
            d_model=self._emb_size,
            d_inner_hid=self._emb_size * 4,
            prepostprocess_dropout=self._prepostprocess_dropout,
            attention_dropout=self._attention_dropout,
            relu_dropout=0,
            hidden_act=self._hidden_act,
            preprocess_cmd="",
            postprocess_cmd="dan",
            # param_initializer=self._param_initializer,
            name=self.encoder_name )

    def get_sequence_output(self):
        return self._enc_out

    ##############################################################
    # task function
    # 1. mask goal type get output feature
    # 2. mask goal entity
    # 3. mask knowledge s
    # 4. mask knowledge p
    ##############################################################
    def mask_goal_type(self, goal_type_pos, goal_type_label):
        """"""
        trm_output = self.get_sequence_output()
        selected_enc = fluid.layers.gather_nd(trm_output, goal_type_pos)
        output = fluid.layers.fc(selected_enc, size=self.goal_type_num)

        goal_type_label = fluid.layers.reshape(goal_type_label, shape=[-1, 1])
        loss, goal_type_softmax = fluid.layers.softmax_with_cross_entropy(logits=output,
                                                                          label=goal_type_label, return_softmax=True)
        mean_loss = fluid.layers.mean(loss)
        # print("goal_type_softmax.shape", goal_type_label.shape)

        goal_type_acc = fluid.layers.accuracy(
            input=goal_type_softmax, label=goal_type_label)
        return output, mean_loss, goal_type_acc

    def mask_goal_entity(self, goal_entity_pos, goal_entity_label):
        """

        :param goal_entity_pos:
        :param goal_entity_label:
        :return:
        """
        trm_output = self.get_sequence_output()
        selected_enc = fluid.layers.gather_nd(trm_output, goal_entity_pos)
        output = fluid.layers.fc(selected_enc, size=self.goal_entity_num)

        goal_entity_label = fluid.layers.reshape(goal_entity_label, shape=[-1, 1])
        loss, goal_entity_softmax = fluid.layers.softmax_with_cross_entropy(logits=output,
                                                                            label=goal_entity_label,
                                                                            return_softmax=True)
        mean_loss = fluid.layers.mean(loss)

        goal_entity_acc = fluid.layers.accuracy(
            input=goal_entity_softmax, label=goal_entity_label)
        return output, mean_loss, goal_entity_acc

    def _mask_goal_entity_constant(self):
        output = fluid.layers.fill_constant(shape=[1], dtype=self._dtype, value=0.0)
        mean_loss = fluid.layers.fill_constant(shape=[1], dtype=self._dtype, value=0.0)
        goal_entity_acc = fluid.layers.fill_constant(shape=[1], dtype=self._dtype, value=0.0)

        return output, mean_loss, goal_entity_acc

    def _mask_goal_entity(self, goal_entity_pos, goal_entity_label):
        """"""
        transformer_output = self.get_sequence_output()
        goal_entity_label = fluid.layers.reshape(goal_entity_label, shape=[-1, 1])

        trm_output = transformer_output
        selected_enc = fluid.layers.gather_nd(trm_output, goal_entity_pos)
        output = fluid.layers.fc(selected_enc, size=self.goal_entity_num)

        def _mask_goal_entity():
            """

            :param goal_entity_pos:
            :param goal_entity_label:
            :return:
            """
            loss, goal_entity_softmax = fluid.layers.softmax_with_cross_entropy(logits=output,
                                                                                label=goal_entity_label,
                                                                                return_softmax=True)
            mean_loss = fluid.layers.mean(loss)

            goal_entity_acc = fluid.layers.accuracy(
                input=goal_entity_softmax, label=goal_entity_label)
            return output, mean_loss, goal_entity_acc

        def _mask_goal_entity_constant():
            output = fluid.layers.fill_constant(shape=[1], dtype=self._dtype, value=0.0)
            mean_loss = fluid.layers.fill_constant(shape=[1], dtype=self._dtype, value=0.0)
            goal_entity_acc = fluid.layers.fill_constant(shape=[1], dtype=self._dtype, value=0.0)

            return output, mean_loss, goal_entity_acc

        output, mean_loss, goal_entity_acc = fluid.layers.cond(
            fluid.layers.has_inf(goal_entity_label),
            _mask_goal_entity_constant,
            _mask_goal_entity)
        # output, mean_loss,goal_entity_acc = _mask_goal_entity_constant()
        # fluid.layers.Print(goal_entity_label)

        return output, mean_loss, goal_entity_acc

    def mask_knowledge_s(self, knowledge_s_pos, knowledge_s_label):
        """"""
        trm_output = self.get_sequence_output()
        selected_enc = fluid.layers.gather_nd(trm_output, knowledge_s_pos)
        output = fluid.layers.fc(selected_enc, size=self.knowledge_s_num)
        knowledge_s_label = fluid.layers.reshape(knowledge_s_label, shape=[-1, 1])

        loss, knowledge_s_softmax = fluid.layers.softmax_with_cross_entropy(logits=output,
                                                                            label=knowledge_s_label,
                                                                            return_softmax=True)
        mean_loss = fluid.layers.mean(loss)
        knowledge_s_acc = fluid.layers.accuracy(
            input=knowledge_s_softmax, label=knowledge_s_label)
        return output, mean_loss, knowledge_s_acc

    def mask_knowledge_p(self, knowledge_p_pos, knowledge_p_label):
        """"""
        trm_output = self.get_sequence_output()
        selected_enc = fluid.layers.gather_nd(trm_output, knowledge_p_pos)
        output = fluid.layers.fc(selected_enc, size=self.knowledge_p_num)
        knowledge_p_label = fluid.layers.reshape(knowledge_p_label, shape=[-1, 1])

        loss, knowledge_p_softmax = fluid.layers.softmax_with_cross_entropy(logits=output,
                                                                            label=knowledge_p_label,
                                                                            return_softmax=True)
        mean_loss = fluid.layers.mean(loss)
        knowledge_p_acc = fluid.layers.accuracy(
            input=knowledge_p_softmax, label=knowledge_p_label)
        return output, mean_loss, knowledge_p_acc

    def get_pretrain_output(self, data_dict):
        """

        :param goal_type_pos:
        :param goal_type_label:
        :param goal_entity_pos:
        :param goal_entity_label:
        :param knowledge_s_pos:
        :param knowledge_s_label:
        :param knowledge_p_pos:
        :param knowledge_p_label:
        :return:
        """
        goal_type_pos = data_dict["goal_type_pos"]
        goal_type_label = data_dict['goal_type_label']
        goal_entity_pos = data_dict['goal_entity_pos']
        goal_entity_label = data_dict['goal_entity_label']
        knowledge_s_pos = data_dict['knowledge_s_pos']
        knowledge_s_label = data_dict['knowledge_s_label']
        knowledge_p_pos = data_dict['knowledge_p_pos']
        knowledge_p_label = data_dict['knowledge_p_label']

        goal_type_output, goal_type_loss, goal_type_acc = self.mask_goal_type(
            goal_type_pos,
            goal_type_label
        )

        goal_entity_output, goal_entity_loss, goal_entity_acc = self.mask_goal_entity(
            goal_entity_pos,
            goal_entity_label
        )

        knowledge_s_output, knowledge_s_loss, knowledge_s_acc = self.mask_knowledge_s(
            knowledge_s_pos,
            knowledge_s_label
        )
        knowledge_p_output, knowledge_p_loss, knowledge_p_acc = self.mask_knowledge_p(
            knowledge_p_pos,
            knowledge_p_label
        )

        loss = goal_type_loss + goal_entity_loss + knowledge_s_loss + knowledge_p_loss
        task_mesure = [[goal_type_loss, goal_type_acc],
                       [goal_entity_loss, goal_entity_acc],
                       [knowledge_s_loss, knowledge_s_acc],
                       [knowledge_p_loss, knowledge_p_acc]]

        mean_mesure = fluid.layers.scale(goal_type_acc + goal_entity_acc + knowledge_s_acc + knowledge_p_acc,
                                         0.25)

        return loss, mean_mesure


if __name__ == '__main__':

    # logging tools
    tgt_base_dir = set_base(__file__)
    # LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
    # DATE_FORMAT = "%m/%d/%Y %H:%M:%S %p"
    log_filename = os.path.join(tgt_base_dir, "enc_pre_training.log")
    logger = log(log_filename)
    config = Model_Config()


    ###########################################################################
    # create program
    train_prog = fluid.Program()
    train_startup = fluid.Program()

    # define the network
    with fluid.program_guard(train_prog, train_startup):

        token_ids = fluid.layers.data(name="token_ids", shape=[None, SEQ_MAX_LEN], dtype='int64')
        segment_ids = fluid.layers.data(name="segment_ids", shape=[None, SEQ_MAX_LEN], dtype='int64')
        pos_ids = fluid.layers.data(name="pos_ids", shape=[None, SEQ_MAX_LEN], dtype='int64')
        input_length = fluid.layers.data(name='input_length', shape=[None, SEQ_MAX_LEN, 1], dtype='int64')

        # task work
        goal_type_pos = fluid.layers.data(name="goal_type_pos", shape=[None, 2], dtype='int64')
        goal_entity_pos = fluid.layers.data(name="goal_entity_pos", shape=[None, 2], dtype='int64')
        knowledge_s_pos = fluid.layers.data(name="knowledge_s_pos", shape=[None, 2], dtype='int64')
        knowledge_p_pos = fluid.layers.data(name="knowledge_p_pos", shape=[None, 2], dtype='int64')

        # task label
        goal_type_label = fluid.layers.data(name="goal_type_label", shape=[None], dtype='int64')
        goal_entity_label = fluid.layers.data(name="goal_entity_label", shape=[None], dtype='int64')
        knowledge_s_label = fluid.layers.data(name="knowledge_s_label", shape=[None], dtype='int64')
        knowledge_p_label = fluid.layers.data(name="knowledge_p_label", shape=[None], dtype='int64')

        pretrian_data = {
            "goal_type_pos": goal_type_pos,
            'goal_type_label': goal_type_label,
            'goal_entity_pos': goal_entity_pos,
            'goal_entity_label': goal_entity_label,
            'knowledge_s_pos': knowledge_s_pos,
            'knowledge_s_label': knowledge_s_label,
            'knowledge_p_pos': knowledge_p_pos,
            'knowledge_p_label': knowledge_p_label
        }

        encode = Encoder(token_ids, pos_ids, segment_ids, input_length,config)
        output = encode.get_sequence_output()
        loss, mean_mesure = encode.get_pretrain_output(pretrian_data)
        adam = fluid.optimizer.AdamOptimizer()
        adam.minimize(loss)

    place = fluid.CUDAPlace(0) if USE_CUDA else fluid.CPUPlace()
    exe = fluid.Executor(place)
    data = Dataset(limit=LIMIT)
    check_info = check_params(data)
    logger.info(check_info)
    params_list = train_prog.block(0).all_parameters()
    params_name_list = [p.name for p in params_list]
    write_iterable("encoder_params.param",params_name_list)
    # startup the program
    exe.run(train_startup)

    if LOAD_PERSISTABLE:
        try:
            print("Begin to Load!")
            cpu_exe = fluid.Executor(fluid.CPUPlace())
            fluid.io.load_persistables(cpu_exe, tgt_base_dir, main_program=train_prog,
                                       filename=LOAD_PERSISTABLE_FILE)

            info_msg = "Load %s success!" % (LOAD_PERSISTABLE_FILE)
            logger.info(info_msg)
            print(info_msg)
        except:
            load_error = "the persistable model cannot be loaded."
            logger.error(load_error)
            exe.run(train_startup)

            print(load_error)

    # load the model if we have
    if LOAD_VARS:
        try:
            print("begin to load %s" % (LOAD_VARS_FILE))
            # fluid.io.load_vars(exe, tgt_base_dir, main_program=train_prog, filename=LOAD_VARS_FILE,
            #                    predicate=find_name_enc)

            info_msg = "Load %s success!" % (LOAD_VARS_FILE)
            logger.info(info_msg)
            print(info_msg)
        except:
            load_error = "the vars model cannot be loaded."
            logger.error(load_error)

    # load the model if we have
    if LOAD_MODEL:
        try:
            model_file = os.path.join(tgt_base_dir, LOAD_MODEL_FILE)
            opt_file = os.path.join(tgt_base_dir, LOAD_OPT_FILE)
            print("begin to load %s" % (LOAD_OPT_FILE))

            load_model(model_file, params_name_list, place, opt_state_init_file=opt_file if LOAD_OPT else "")
            info_msg = "Load %s success!" % (LOAD_OPT_FILE)
            logger.info(info_msg)
            print(info_msg)
        except:
            load_error = "the model params cannot be loaded."
            logger.error(load_error)



    logger.info("Begin trainning")
    print("Begin trainning")
    start_time = time.time()
    recoder = time.time()
    for epoch_id in range(EPOCH_NUM):
        data_gen = data.generate_enc_batch(batch_size=BATCH_SIZE)
        for batch_id, item in enumerate(data_gen):
            # load the data and input them into encoder
            token_arr, segment_arr, pos_list_arr, seq_len_arr, \
            goal_type_pos_arr, goal_entity_pos_arr, knowledge_s_pos_arr, knowledge_p_pos_arr, \
            goal_type_label_arr, goal_entity_label_arr, knowledge_s_label_arr, knowledge_p_label_arr = item
            feed_data = {
                "token_ids": token_arr,
                "segment_ids": segment_arr,
                "pos_ids": pos_list_arr,
                "input_length": seq_len_arr,
                "goal_type_pos": goal_type_pos_arr,
                'goal_entity_pos': goal_entity_pos_arr,
                'knowledge_s_pos': knowledge_s_pos_arr,
                'knowledge_p_pos': knowledge_p_pos_arr,
                'goal_type_label': goal_type_label_arr,
                'goal_entity_label': goal_entity_label_arr,
                'knowledge_s_label': knowledge_s_label_arr,
                'knowledge_p_label': knowledge_p_label_arr
            }
            res = exe.run(train_prog, feed=feed_data,
                          fetch_list=[loss, mean_mesure], return_numpy=True)
            # print msg
            if batch_id % PRINT_BATCH == 0:
                now = time.time()
                info_msg = "Now epoch: %d, batch: %d, avg loss: %.3f, task accuracy: %.3f, spend %d s, speed %.2f batch/s, " % \
                           (epoch_id, batch_id, res[0], res[1], now - start_time, PRINT_BATCH / (now - recoder))
                logger.info(info_msg)
                print(info_msg)
                recoder = time.time()

            # save model
            if batch_id % SAVE_BATCH == 0:
                save_msg = "save model at %d epoch, %d batch" % (epoch_id, batch_id)
                model_name = "enc_model" + "_epoch_%d_batch_%d" % (epoch_id, batch_id)
                logger.info(save_msg)
                fluid.io.save_vars(exe, tgt_base_dir, main_program=train_prog, filename=model_name + ".vars",
                                   predicate=lambda var: isinstance(var, fluid.framework.Parameter))
                fluid.io.save_persistables(exe, tgt_base_dir,
                                           main_program=train_prog, filename=model_name + ".pers")

                opt_var_name_list = adam.get_opti_var_name_list()
                save_model_info_msg = save_model(tgt_base_dir, param_name_list=params_name_list,opt_var_name_list=opt_var_name_list, name=model_name)
                logger.info(save_msg+save_model_info_msg)

                if len(os.listdir(tgt_base_dir)) > MAX_SAVE:
                    file_list = [(os.path.join(tgt_base_dir, item), os.path.getmtime(os.path.join(tgt_base_dir, item)))
                                 for item in os.listdir(tgt_base_dir)]
                    delete_file = sorted(file_list, key=lambda x: x[1], reverse=False)
                    os.remove(delete_file[0][0])
                    os.remove(delete_file[1][0])
                    os.remove(delete_file[2][0])

                    del_msg = "delete the model file %s." % (delete_file)
                    logger.info(del_msg)




