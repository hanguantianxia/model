# author:hjt
# contact: hanguantianxia@sina.com
# datetime:2020/4/19 18:33
# software: PyCharm

"""

"""
import os
import json
import six
import numpy as np
import logging
import time
import sys

import paddle.fluid as fluid
from paddle.fluid.layers import core
from functools import partial

from transformer_encoder import decoder
from batch_generator import Dataset
from utils import set_base, log

#################################################################################
# 参数部分
HIDDEN_SIZE = 768
NUM_HIDDEN_LAYERS = 6
NUM_ATTENTION_HEADS = 6
VOCAB_SIZE = 15416
MAX_POSITION_EMBEDDINGS = 512
TYPE_VOCAB_SIZE = 60
HIDDEN_ACT = 'relu'
HIDDEN_DROPOUT_PROB = 0.
ATTENTION_PROBS_DROPOUT_PROB = 0.
GOAL_TYPE_NUM = 39
GOAL_ENTITY_NUM = 656
KNOWLEDGE_S_NUM = 656
KNOWLEDGE_P_NUM = 57
#################################################################################
# Train Parameters
SEQ_MAX_LEN = 512
BATCH_SIZE = 1
EPOCH_NUM = 5
PRINT_BATCH = 10
SAVE_BATCH = 100
LOAD_PERSISTABLE = False
LOAD_PERSISTABLE_FILE = "dec_model_epoch_4_batch_900.pers"
LOAD_VARS = False
LOAD_VARS_FILE = "dec_model_epoch_4_batch_1300.vars"
TRAIN_STAT_PATH = "training_msg.json"
MAX_SAVE = 12
LIMIT = 10
USE_CUDA = True


###########################################################################

class Decoder(object):
    def __init__(self,
                 src_ids,
                 position_ids,
                 sentence_ids,
                 input_mask,
                 enc_input,
                 enc_input_mask,
                 weight_sharing=True,
                 cache=None,
                 use_fp16=False):
        self._emb_size = HIDDEN_SIZE
        self._n_layer = NUM_HIDDEN_LAYERS
        self._n_head = NUM_ATTENTION_HEADS
        self._voc_size = VOCAB_SIZE
        self._max_position_seq_len = MAX_POSITION_EMBEDDINGS
        self._sent_types = TYPE_VOCAB_SIZE
        self._hidden_act = HIDDEN_ACT
        self._prepostprocess_dropout = HIDDEN_DROPOUT_PROB
        self._attention_dropout = ATTENTION_PROBS_DROPOUT_PROB
        self._weight_sharing = weight_sharing

        self._word_emb_name = "enc_word_embedding"
        self._pos_emb_name = "enc_pos_embedding"
        self._seg_emb_name = "enc_seg_embedding"

        self._dtype = "float32"
        self._inttype = 'int32'

        # task parameters
        self.goal_type_num = GOAL_TYPE_NUM
        self.goal_entity_num = GOAL_ENTITY_NUM
        self.knowledge_s_num = KNOWLEDGE_S_NUM
        self.knowledge_p_num = KNOWLEDGE_P_NUM

        # name
        self.decoder_name = "decoder"

        # parameter
        self.pos_embed = Dataset.get_position_embed(SEQ_MAX_LEN, HIDDEN_SIZE)

        # self._param_initializer = fluid.initializer.TruncatedNormal(
        #     scale=config['initializer_range'])
        self._build_model(src_ids, position_ids, sentence_ids,
                          input_mask, enc_input, enc_input_mask, cache=cache)

    def _build_model(self, src_ids, position_ids, sentence_ids,
                     input_mask, enc_input, enc_input_mask, cache=None):
        """
        padding id in vocabulary must be set to 0

        :param src_ids:
        :param position_ids:
        :param sentence_ids:
        :param pad_matrix:
        :return:
        """
        # embedding
        emb_out = fluid.embedding(
            input=src_ids,
            size=[self._voc_size, self._emb_size],
            dtype=self._dtype,
            param_attr=fluid.ParamAttr(
                name=self._word_emb_name),
            is_sparse=False)

        position_emb_out = fluid.embedding(
            input=position_ids,
            size=[self._max_position_seq_len, self._emb_size],
            dtype=self._dtype,
            param_attr=fluid.ParamAttr(
                name=self._pos_emb_name,
                initializer=fluid.initializer.NumpyArrayInitializer(self.pos_embed)
                , trainable=False)  # ,# initializer=self._param_initializer)
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

        enc_attn_mask = fluid.layers.scale(
            x=enc_input_mask, scale=1000000.0, bias=-1.0, bias_after_scale=False)
        # print(enc_attn_mask.shape)
        enc_attn_mask = fluid.layers.stack(
            x=[enc_attn_mask] * self._n_head, axis=1)
        enc_attn_mask.stop_gradient = True
        # print("self_attn_mask",enc_attn_mask.shape)

        self_attn_mask = fluid.layers.scale(
            x=input_mask, scale=1000000.0, bias=-1.0, bias_after_scale=False)
        n_head_self_attn_mask = fluid.layers.stack(
            x=[self_attn_mask] * self._n_head, axis=1)
        n_head_self_attn_mask.stop_gradient = True

        self._dec_out = decoder(
            dec_input=emb_out,
            enc_output=enc_input,
            dec_slf_attn_bias=n_head_self_attn_mask,
            dec_enc_attn_bias=enc_attn_mask,
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
            caches=cache,
            name=self.decoder_name)

    def get_sequence_output(self):
        return self._dec_out

    ##############################################################
    # task function
    # 1. mask goal type
    # 2. Languague model
    ##############################################################
    def mask_goal_type(self, goal_type_pos, goal_type_label, name='_mask_goal_type'):
        """

        :param goal_type_pos:
        :param goal_type_label:
        :param name:
        :return:
        """
        trm_output = self.get_sequence_output()
        selected_enc = fluid.layers.gather_nd(trm_output, goal_type_pos)
        output = fluid.layers.fc(selected_enc,
                                 size=self.goal_type_num,
                                 param_attr=fluid.ParamAttr(name=self.decoder_name + name + "_fc_w"),
                                 bias_attr=fluid.ParamAttr(name=self.decoder_name + name + "_fc_b"))

        goal_type_label = fluid.layers.reshape(goal_type_label, shape=[-1, 1])
        loss, goal_type_softmax = fluid.layers.softmax_with_cross_entropy(logits=output,
                                                                          label=goal_type_label, return_softmax=True)
        mean_loss = fluid.layers.mean(loss)
        # print("goal_type_softmax.shape", goal_type_label.shape)

        goal_type_acc = fluid.layers.accuracy(
            input=goal_type_softmax, label=goal_type_label)

        return output, mean_loss, goal_type_acc

    def lm_task(self, lm_label_arr, lm_pos_mask, lm_pos_len, name='LM_task'):
        """

        :param LM_pos_mask:
        :return:
        """
        # tgt_output [batch_size, max_len, tgt_vocab_size]
        tgt_output = fluid.layers.fc(self.get_sequence_output(),
                                     size=self._voc_size,
                                     num_flatten_dims=2,
                                     param_attr=fluid.ParamAttr(name=self.decoder_name + name + "_w"),
                                     bias_attr=fluid.ParamAttr(name=self.decoder_name + name + "_b"))
        # loss_nll [batch_size, max_len, 1]
        tgt_output = fluid.layers.softmax(tgt_output, use_cudnn=USE_CUDA)
        loss_nll = fluid.layers.cross_entropy(tgt_output, lm_label_arr)
        # loss_nll [batch_size, max_len]
        loss_nll = fluid.layers.reshape(loss_nll, shape=[BATCH_SIZE, SEQ_MAX_LEN])
        # loss_nll [batch_size, max_len]
        loss_nll *= lm_pos_mask
        # loss_nll [batch_size]
        loss_nll = fluid.layers.reduce_sum(loss_nll, dim=1)
        loss_nll /= lm_pos_len
        loss_nll = fluid.layers.reduce_sum(loss_nll)
        loss_nll /= BATCH_SIZE
        return loss_nll

    def pretrain(self, goal_type_pos, goal_type_label, lm_label_arr,
                 lm_pos_mask, lm_pos_len):
        """

        :param goal_type_pos:
        :param goal_type_label:
        :param token_arr:
        :param LM_pos:
        :return:
        """
        _, loss_goal_type, acc_goal_type = self.mask_goal_type(goal_type_pos, goal_type_label)
        loss_LM = self.lm_task(lm_label_arr, lm_pos_mask, lm_pos_len)
        loss = loss_LM + loss_goal_type

        return loss, acc_goal_type


def fine_tunning():
    ###########################################################################
    # logging tools
    tgt_base_dir = set_base(__file__)
    log_filename = os.path.join(tgt_base_dir, "dec_pre_training.log")
    logger = log(log_filename)

    # define program
    train_prog = fluid.Program()
    startup_prog = fluid.Program()

    # define model
    with fluid.program_guard(train_prog, startup_prog):
        token_ids = fluid.layers.data(name="token_ids", shape=[None, SEQ_MAX_LEN], dtype='int64')
        segment_ids = fluid.layers.data(name="segment_ids", shape=[None, SEQ_MAX_LEN], dtype='int64')
        pos_ids = fluid.layers.data(name="pos_ids", shape=[None, SEQ_MAX_LEN], dtype='int64')
        enc_slf_attn = fluid.layers.data(name='enc_slf_attn', shape=[None, SEQ_MAX_LEN, SEQ_MAX_LEN], dtype='int64')

        lm_label_mat = fluid.layers.data(name='lm_label_mat', shape=[None, SEQ_MAX_LEN], dtype='int64')
        lm_pos_mask = fluid.layers.data(name='lm_pos_mask', shape=[None, SEQ_MAX_LEN, SEQ_MAX_LEN], dtype='int64')
        lm_pos_len = fluid.layers.data(name='lm_pos_len', shape=[None, 1], dtype='int64')

        goal_type_pos = fluid.layers.data(name="goal_type_pos", shape=[None, 2], dtype='int64')
        goal_type_label = fluid.layers.data(name="goal_type_label", shape=[None], dtype='int64')

        enc_input = fluid.layers.fill_constant(shape=[BATCH_SIZE, SEQ_MAX_LEN, HIDDEN_SIZE], dtype='float32', value=0.0)
        enc_mask = fluid.layers.fill_constant(shape=[BATCH_SIZE, SEQ_MAX_LEN, SEQ_MAX_LEN], dtype='float32', value=0.0)

        decode = Decoder(token_ids, pos_ids, segment_ids, enc_slf_attn, enc_input=enc_input, enc_input_mask=enc_mask)
        # output, loss, acc = decode.mask_goal_type(goal_type_pos, goal_type_label)
        # loss, goal_type_acc = decode.pretrain(goal_type_pos, goal_type_label, lm_label_mat, lm_pos_mask, lm_pos_len)
        loss = decode.lm_task(lm_label_mat, lm_pos_mask, lm_pos_len)
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
                                   predicate=lambda var: isinstance(var, fluid.framework.Parameter) and decode.decoder_name in var.name)

            info_msg = "Load %s success!" % (LOAD_VARS_FILE)
            logger.info(info_msg)
            print(info_msg)
        except:
            load_error = "the vars model cannot be loaded."
            logger.error(load_error)

    # show the information
    start_time = time.time()
    recoder = time.time()
    logger.info("Begin trainning")
    print("Begin trainning")
    for epoch_id in range(EPOCH_NUM):

        data_gen = dataset.generate_dec_pretrain_batch(batch_size=BATCH_SIZE)
        for batch_id, item in enumerate(data_gen):

            token_id_arr, segment_id_arr, pos_arr, slf_attn_mask_arr, \
            lm_mask_arr, lm_len_arr, lm_lable_arr = item

            feed_dict = {
                "token_ids": token_id_arr,
                "segment_ids": segment_id_arr,
                "pos_ids": pos_arr,
                "enc_slf_attn": slf_attn_mask_arr,
                # "goal_type_pos": goal_pos,
                # "goal_type_label": goal_type_list,
                "lm_label_mat": lm_lable_arr,
                "lm_pos_mask": lm_mask_arr,
                "lm_pos_len": lm_len_arr
            }

            res = exe.run(train_prog, feed=feed_dict, fetch_list=[loss])
            # print msg
            if batch_id % PRINT_BATCH == 0:
                now = time.time()
                info_msg = "Now epoch: %d, batch: %d, avg loss: %.3f, spend %d s, speed %.2f batch/s, " % \
                           (epoch_id, batch_id, res[0], now - start_time, PRINT_BATCH / (now - recoder))
                logger.info(info_msg)
                print(info_msg)
                recoder = time.time()

            if batch_id % SAVE_BATCH == 0:
                save_msg = "save model at %d epoch, %d batch" % (epoch_id, batch_id)
                model_name = "dec_model" + "_epoch_%d_batch_%d" % (epoch_id, batch_id)
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


if __name__ == '__main__':
    fine_tunning()