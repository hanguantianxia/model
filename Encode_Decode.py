# -*- encoding: utf-8 -*-
'''
@Author          hjt
@File            Encode_Decode.py.py
@Contact         hanguantianxia@sina.com
@License         (C)Copyright 2017-2018, Liugroup-NLPR-CASIA
@Modify Time     2020/4/21 18:41    
@Version         1.0 
@Desciption 

'''

import os
import json
import six
import numpy as np
import logging
import time

import paddle.fluid as fluid
from paddle.fluid.layers import core
from functools import partial

from transformer_encoder import encoder, decoder
from batch_generator import Dataset
from utils import set_base, log, read_json, json_saver, find_name_enc, find_name_dec
from Encoder import Encoder
from Decoder import Decoder
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
KNOWLEDGE_P_NUM = 56
SEQ_MAX_LEN = 512
###########################################################################
# Train Parameters
BATCH_SIZE = 32
EPOCH_NUM = 100
PRINT_BATCH = 10
SAVE_BATCH = 100
LOAD_PERSISTABLE = False
LOAD_PERSISTABLE_FILE = "trm_model_epoch_4_batch_0.pers"
LOAD_VARS = False
LOAD_VARS_FILE = "trm_model_epoch_4_batch_0.vars"
LOAD_SEP = True
LOAD_ENCODER_FILE = "enc_model_epoch_1_batch_0.vars"
LOAD_DECODER_FILE = "dec_model_epoch_4_batch_1300.vars"
TRAIN_STAT_PATH = "training_msg.json"
MAX_SAVE = 12
LIMIT = 10
USE_CUDA = False


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


def network():
    # define encoder input data
    enc_token_ids = fluid.layers.data(name="enc_token_ids", shape=[None, SEQ_MAX_LEN], dtype='int64')
    enc_segment_ids = fluid.layers.data(name="enc_segment_ids", shape=[None, SEQ_MAX_LEN], dtype='int64')
    enc_pos_ids = fluid.layers.data(name="enc_pos_ids", shape=[None, SEQ_MAX_LEN], dtype='int64')
    enc_input_length = fluid.layers.data(name='enc_input_length', shape=[None, SEQ_MAX_LEN, 1], dtype='int64')

    # define decoder input data
    dec_token_ids = fluid.layers.data(name="token_ids", shape=[None, SEQ_MAX_LEN], dtype='int64')
    dec_segment_ids = fluid.layers.data(name="segment_ids", shape=[None, SEQ_MAX_LEN], dtype='int64')
    dec_pos_ids = fluid.layers.data(name="pos_ids", shape=[None, SEQ_MAX_LEN], dtype='int64')
    dec_enc_slf_attn = fluid.layers.data(name='enc_slf_attn', shape=[None, SEQ_MAX_LEN, SEQ_MAX_LEN], dtype='int64')

    # task label
    dec_lm_label_mat = fluid.layers.data(name='lm_label_mat', shape=[None, SEQ_MAX_LEN], dtype='int64')
    dec_lm_pos_mask = fluid.layers.data(name='lm_pos_mask', shape=[None, SEQ_MAX_LEN, SEQ_MAX_LEN], dtype='int64')
    dec_lm_pos_len = fluid.layers.data(name='lm_pos_len', shape=[None, 1], dtype='int64')

    goal_type_pos = fluid.layers.data(name="goal_type_pos", shape=[None, 2], dtype='int64')
    goal_type_label = fluid.layers.data(name="goal_type_label", shape=[None], dtype='int64')

    # enc_dec_mask
    enc_dec_mask = fluid.layers.data(name='enc_dec_mask', shape=[None, SEQ_MAX_LEN, SEQ_MAX_LEN], dtype='int64')

    # network
    encode = Encoder(enc_token_ids, enc_pos_ids, enc_segment_ids,
                     enc_input_length, config)
    enc_output = encode.get_sequence_output()

    decode = Decoder(dec_token_ids, dec_pos_ids, dec_segment_ids,
                     dec_enc_slf_attn, config=config, enc_input=enc_output, enc_input_mask=enc_dec_mask)

    loss, goal_type_acc = decode.pretrain(goal_type_pos, goal_type_label,
                                          dec_lm_label_mat, dec_lm_pos_mask, dec_lm_pos_len)

    input_name_list = [
        enc_token_ids.name,
        enc_segment_ids.name,
        enc_pos_ids.name,
        enc_input_length.name,

        dec_token_ids.name,
        dec_segment_ids.name,
        dec_pos_ids.name,
        dec_enc_slf_attn.name,
        enc_dec_mask.name,

        dec_lm_label_mat.name,
        dec_lm_pos_mask.name,
        dec_lm_pos_len.name,

        goal_type_pos.name,
        goal_type_label.name
    ]
    return loss.name, goal_type_acc.name, input_name_list


if __name__ == '__main__':

    # logging tools
    tgt_base_dir = set_base(__file__)
    # LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
    # DATE_FORMAT = "%m/%d/%Y %H:%M:%S %p"
    log_filename = os.path.join(tgt_base_dir, "enc_pre_training.log")
    logger = log(log_filename)
    config = Model_Config()

    # define program
    train_prog = fluid.Program()
    startup_prog = fluid.Program()

    # define model
    with fluid.program_guard(train_prog, startup_prog):
        # define encoder input data
        enc_token_ids = fluid.layers.data(name="enc_token_ids", shape=[None, SEQ_MAX_LEN], dtype='int64')
        enc_segment_ids = fluid.layers.data(name="enc_segment_ids", shape=[None, SEQ_MAX_LEN], dtype='int64')
        enc_pos_ids = fluid.layers.data(name="enc_pos_ids", shape=[None, SEQ_MAX_LEN], dtype='int64')
        enc_input_length = fluid.layers.data(name='enc_input_length', shape=[None, SEQ_MAX_LEN, 1], dtype='int64')

        # define decoder input data
        dec_token_ids = fluid.layers.data(name="token_ids", shape=[None, SEQ_MAX_LEN], dtype='int64')
        dec_segment_ids = fluid.layers.data(name="segment_ids", shape=[None, SEQ_MAX_LEN], dtype='int64')
        dec_pos_ids = fluid.layers.data(name="pos_ids", shape=[None, SEQ_MAX_LEN], dtype='int64')
        dec_enc_slf_attn = fluid.layers.data(name='enc_slf_attn', shape=[None, SEQ_MAX_LEN, SEQ_MAX_LEN], dtype='int64')

        # task label
        dec_lm_label_mat = fluid.layers.data(name='lm_label_mat', shape=[None, SEQ_MAX_LEN], dtype='int64')
        dec_lm_pos_mask = fluid.layers.data(name='lm_pos_mask', shape=[None, SEQ_MAX_LEN, SEQ_MAX_LEN], dtype='int64')
        dec_lm_pos_len = fluid.layers.data(name='lm_pos_len', shape=[None, 1], dtype='int64')

        goal_type_pos = fluid.layers.data(name="goal_type_pos", shape=[None, 2], dtype='int64')
        goal_type_label = fluid.layers.data(name="goal_type_label", shape=[None], dtype='int64')

        # enc_dec_mask
        enc_dec_mask = fluid.layers.data(name='enc_dec_mask', shape=[None, SEQ_MAX_LEN, SEQ_MAX_LEN], dtype='int64')

        # network
        encode = Encoder(enc_token_ids, enc_pos_ids, enc_segment_ids,
                         enc_input_length, config)
        enc_output = encode.get_sequence_output()

        decode = Decoder(dec_token_ids, dec_pos_ids, dec_segment_ids,
                         dec_enc_slf_attn, config=config, enc_input=enc_output, enc_input_mask=enc_dec_mask)
        dec_output = decode.get_sequence_output()


        loss, goal_type_acc = decode.pretrain(goal_type_pos, goal_type_label,
                                              dec_lm_label_mat, dec_lm_pos_mask, dec_lm_pos_len)

        # loss
        adam = fluid.optimizer.AdamOptimizer()
        adam.minimize(loss)

    # define executor
    place = fluid.CUDAPlace(0) if USE_CUDA else fluid.CPUPlace()
    exe = fluid.Executor(place)

    # start up parameter
    exe.run(startup_prog)
    dataset = Dataset(limit=LIMIT)

    # load the model if we have
    if LOAD_PERSISTABLE:
        try:
            print("begin to load %s"% (LOAD_PERSISTABLE_FILE))
            fluid.io.load_persistables(exe, tgt_base_dir, main_program=train_prog,
                                       filename=LOAD_PERSISTABLE_FILE)

            info_msg = "Load %s success!" % (LOAD_PERSISTABLE_FILE)
            logger.info(info_msg)
            print(info_msg)
        except:
            load_error = "the persistable model cannot be loaded."
            logger.error(load_error)

    # load the model if we have
    if LOAD_VARS:
        try:
            print("begin to load %s"% (LOAD_VARS_FILE))
            fluid.io.load_vars(exe, tgt_base_dir, main_program=train_prog, filename=LOAD_VARS_FILE,
                                   predicate=lambda var: isinstance(var, fluid.framework.Parameter))

            info_msg = "Load %s success!" % (LOAD_VARS_FILE)
            logger.info(info_msg)
            print(info_msg)
        except:
            load_error = "the vars model cannot be loaded."
            logger.error(load_error)

    # load encoder and decoder seperately
    if LOAD_SEP:
        try:
            print("begin to load %s" % (LOAD_VARS_FILE))
            fluid.io.load_vars(exe, tgt_base_dir, main_program=train_prog, filename=LOAD_ENCODER_FILE,
                               predicate=find_name_enc)
            fluid.io.load_vars(exe, tgt_base_dir, main_program=train_prog, filename=LOAD_DECODER_FILE,
                               predicate=find_name_dec)

            info_msg = "Load %s success!" % (LOAD_VARS_FILE)
            logger.info(info_msg)
            print(info_msg)
        except:
            load_error = "the vars model cannot be loaded."
            logger.error(load_error)

    # show the information
    check_params(dataset)

    # clock and message
    start_time = time.time()
    recoder = time.time()
    logger.info("Begin trainning")
    print("Begin trainning")

    for epoch_id in range(EPOCH_NUM):
        data_gen = dataset.generate_fine_tunning_batch(batch_size=BATCH_SIZE)
        for batch_id, item in enumerate(data_gen):
            input_name_list = [
                enc_token_ids.name,
                enc_segment_ids.name,
                enc_pos_ids.name,
                enc_input_length.name,

                dec_token_ids.name,
                dec_segment_ids.name,
                dec_pos_ids.name,
                dec_enc_slf_attn.name,
                enc_dec_mask.name,

                dec_lm_label_mat.name,
                dec_lm_pos_mask.name,
                dec_lm_pos_len.name,

                goal_type_pos.name,
                goal_type_label.name
            ]
            feed_dict = {key:value for key,value in zip(input_name_list, item)}

            res = exe.run(train_prog, feed=feed_dict, fetch_list=[loss, goal_type_acc])
            # print msg
            if batch_id % PRINT_BATCH == 0:
                now = time.time()
                info_msg = "Now epoch: %d, batch: %d, avg loss: %.3f, task accuracy: %.3f, spend %d s, speed %.2f batch/s, " % \
                           (epoch_id, batch_id, res[0], res[1], now - start_time, PRINT_BATCH / (now - recoder))
                logger.info(info_msg)
                print(info_msg)
                recoder = time.time()

            if batch_id % SAVE_BATCH == 0:
                save_msg = "save model at %d epoch, %d batch" % (epoch_id, batch_id)
                model_name = "trm_model" + "_epoch_%d_batch_%d" % (epoch_id, batch_id)
                logger.info(save_msg)
                fluid.io.save_vars(exe, tgt_base_dir, main_program=train_prog, filename=model_name + ".vars",
                                   predicate=lambda var: isinstance(var, fluid.framework.Parameter))
                fluid.io.save_persistables(exe, tgt_base_dir,
                                           main_program=train_prog, filename=model_name + ".pers")

                if len(os.listdir(tgt_base_dir)) > MAX_SAVE:
                    file_list = [
                        (os.path.join(tgt_base_dir, item), os.path.getmtime(os.path.join(tgt_base_dir, item)))
                        for item in os.listdir(tgt_base_dir)]
                    delete_file = sorted(file_list, key=lambda x: x[1], reverse=False)
                    os.remove(delete_file[0][0])
                    os.remove(delete_file[1][0])

                    del_msg = "delete the model file %s." % (delete_file)
                    logger.info(del_msg)