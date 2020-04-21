# author:hjt
# contact: hanguantianxia@sina.com
# datetime:2020/4/12 10:35
# software: PyCharm

"""

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
from functools import partial
from typing import List, Dict, Tuple

import paddle.fluid as fluid
import paddle.fluid.layers as layers
import numpy as np

INT_MAX = 100000

def multi_head_attention(queries,
                         keys,
                         values,
                         attn_bias,
                         d_key,
                         d_value,
                         d_model,
                         n_head=1,
                         dropout_rate=0.,
                         cache=None,
                         param_initializer=None,
                         name='multi_head_att'):
    """
    Multi-Head Attention. Note that attn_bias is added to the logit before
    computing softmax activiation to mask certain selected positions so that
    they will not considered in attention weights.
    """
    keys = queries if keys is None else keys
    values = keys if values is None else values

    if not (len(queries.shape) == len(keys.shape) == len(values.shape) == 3):
        raise ValueError(
            "Inputs: quries, keys and values should all be 3-D tensors.")

    def __compute_qkv(queries, keys, values, n_head, d_key, d_value):
        """
        Add linear projection to queries, keys, and values.
        """
        q = layers.fc(input=queries,
                      size=d_key * n_head,
                      num_flatten_dims=2,
                      param_attr=fluid.ParamAttr(
                          name=name + '_query_fc.w_0',
                          initializer=param_initializer),
                      bias_attr=name + '_query_fc.b_0')
        # layers.Print(keys)
        k = layers.fc(input=keys,
                      size=d_key * n_head,
                      num_flatten_dims=2,
                      param_attr=fluid.ParamAttr(
                          name=name + '_key_fc.w_0',
                          initializer=param_initializer),
                      bias_attr=name + '_key_fc.b_0')
        # layers.Print(values)
        v = layers.fc(input=values,
                      size=d_value * n_head,
                      num_flatten_dims=2,
                      param_attr=fluid.ParamAttr(
                          name=name + '_value_fc.w_0',
                          initializer=param_initializer),
                      bias_attr=name + '_value_fc.b_0')
        return q, k, v

    def __split_heads(x, n_head):
        """
        Reshape the last dimension of inpunt tensor x so that it becomes two
        dimensions and then transpose. Specifically, input a tensor with shape
        [bs, max_sequence_length, n_head * hidden_dim] then output a tensor
        with shape [bs, n_head, max_sequence_length, hidden_dim].
        """
        hidden_size = x.shape[-1]
        # The value 0 in shape attr means copying the corresponding dimension
        # size of the input as the output dimension size.
        reshaped = layers.reshape(
            x=x, shape=[0, 0, n_head, hidden_size // n_head], inplace=True)

        # permuate the dimensions into:
        # [batch_size, n_head, max_sequence_len, hidden_size_per_head]
        return layers.transpose(x=reshaped, perm=[0, 2, 1, 3])

    def __combine_heads(x):
        """
        Transpose and then reshape the last two dimensions of inpunt tensor x
        so that it becomes one dimension, which is reverse to __split_heads.
        """
        if len(x.shape) == 3: return x
        if len(x.shape) != 4:
            raise ValueError("Input(x) should be a 4-D Tensor.")

        trans_x = layers.transpose(x, perm=[0, 2, 1, 3])
        # The value 0 in shape attr means copying the corresponding dimension
        # size of the input as the output dimension size.
        return layers.reshape(
            x=trans_x,
            shape=[0, 0, trans_x.shape[2] * trans_x.shape[3]],
            inplace=True)

    def scaled_dot_product_attention(q, k, v, attn_bias, d_key, dropout_rate):
        """
        Scaled Dot-Product Attention
        [[
          0 L*L -inf
          -inf -inf
        ]]maxLen*maxLen
        """
        scaled_q = layers.scale(x=q, scale=d_key**-0.5)
        # print("q",q.shape)
        # print("k",k.shape)
        product = layers.matmul(x=scaled_q, y=k, transpose_y=True)
        # print("product",product.shape)
        if attn_bias:
            # print('attn_bias',attn_bias.shape)
            # print(product.shape)
            # print(product.shape)
            # print(attn_bias.shape)
            product += attn_bias
        weights = layers.softmax(product)
        # layers.Print(weights)

        if dropout_rate:
            weights = layers.dropout(
                weights,
                dropout_prob=dropout_rate,
                dropout_implementation="upscale_in_train",
                is_test=False)
        out = layers.matmul(weights, v)
        return out
    # layers.Print(queries)
    # layers.Print(keys)
    q, k, v = __compute_qkv(queries, keys, values, n_head, d_key, d_value)

    if cache is not None:  # use cache and concat time steps
        # Since the inplace reshape in __split_heads changes the shape of k and
        # v, which is the cache input for next time step, reshape the cache
        # input from the previous time step first.
        k = cache["k"] = layers.concat(
            [layers.reshape(
                cache["k"], shape=[0, 0, d_model]), k], axis=1)
        v = cache["v"] = layers.concat(
            [layers.reshape(
                cache["v"], shape=[0, 0, d_model]), v], axis=1)

    q = __split_heads(q, n_head)
    k = __split_heads(k, n_head)
    v = __split_heads(v, n_head)

    ctx_multiheads = scaled_dot_product_attention(q, k, v, attn_bias, d_key,
                                                  dropout_rate)

    out = __combine_heads(ctx_multiheads)

    # Project back to the model size.
    proj_out = layers.fc(input=out,
                         size=d_model,
                         num_flatten_dims=2,
                         param_attr=fluid.ParamAttr(
                             name=name + '_output_fc.w_0',
                             initializer=param_initializer),
                         bias_attr=name + '_output_fc.b_0')
    return proj_out


def positionwise_feed_forward(x,
                              d_inner_hid,
                              d_hid,
                              dropout_rate,
                              hidden_act,
                              param_initializer=None,
                              name='ffn'):
    """
    Position-wise Feed-Forward Networks.
    This module consists of two linear transformations with addd ReLU activation
    in between, which is applied to each position separately and identically.
    """
    hidden = layers.fc(input=x,
                       size=d_inner_hid,
                       num_flatten_dims=2,
                       act=hidden_act,
                       param_attr=fluid.ParamAttr(
                           name=name + '_fc_0.w_0',
                           initializer=param_initializer),
                       bias_attr=name + '_fc_0.b_0')
    if dropout_rate:
        hidden = layers.dropout(
            hidden,
            dropout_prob=dropout_rate,
            dropout_implementation="upscale_in_train",
            is_test=False)
    out = layers.fc(input=hidden,
                    size=d_hid,
                    num_flatten_dims=2,
                    param_attr=fluid.ParamAttr(
                        name=name + '_fc_1.w_0', initializer=param_initializer),
                    bias_attr=name + '_fc_1.b_0')
    return out


def pre_post_process_layer(prev_out, out, process_cmd, dropout_rate=0.,
                           name=''):
    """
    Add residual connection, layer normalization and droput to the out tensor
    optionally according to the value of process_cmd.
    This will be used before or after multi-head attention and position-wise
    feed-forward networks.
    """
    for cmd in process_cmd:
        if cmd == "a":  # add residual connection
            out = out + prev_out if prev_out else out
        elif cmd == "n":  # add layer normalization
            out_dtype = out.dtype
            if out_dtype == fluid.core.VarDesc.VarType.FP16:
                out = layers.cast(x=out, dtype="float32")
            out = layers.layer_norm(
                out,
                begin_norm_axis=len(out.shape) - 1,
                param_attr=fluid.ParamAttr(
                    name=name + '_layer_norm_scale',
                    initializer=fluid.initializer.Constant(1.)),
                bias_attr=fluid.ParamAttr(
                    name=name + '_layer_norm_bias',
                    initializer=fluid.initializer.Constant(0.)))
            if out_dtype == fluid.core.VarDesc.VarType.FP16:
                out = layers.cast(x=out, dtype="float16")
        elif cmd == "d":  # add dropout
            if dropout_rate:
                out = layers.dropout(
                    out,
                    dropout_prob=dropout_rate,
                    dropout_implementation="upscale_in_train",
                    is_test=False)
    return out


pre_process_layer = partial(pre_post_process_layer, None)
post_process_layer = pre_post_process_layer


def encoder_layer(enc_input,
                  attn_bias,
                  n_head,
                  d_key,
                  d_value,
                  d_model,
                  d_inner_hid,
                  prepostprocess_dropout,
                  attention_dropout,
                  relu_dropout,
                  hidden_act,
                  preprocess_cmd="n",
                  postprocess_cmd="da",
                  param_initializer=None,
                  name=''):
    """The encoder layers that can be stacked to form a deep encoder.
    This module consits of a multi-head (self) attention followed by
    position-wise feed-forward networks and both the two components companied
    with the post_process_layer to add residual connection, layer normalization
    and droput.
    """
    attn_output = multi_head_attention(
        pre_process_layer(
            enc_input,
            preprocess_cmd,
            prepostprocess_dropout,
            name=name + '_pre_att'),
        None,
        None,
        attn_bias,
        d_key,
        d_value,
        d_model,
        n_head,
        attention_dropout,
        param_initializer=param_initializer,
        name=name + '_multi_head_att')
    attn_output = post_process_layer(
        enc_input,
        attn_output,
        postprocess_cmd,
        prepostprocess_dropout,
        name=name + '_post_att')
    ffd_output = positionwise_feed_forward(
        pre_process_layer(
            attn_output,
            preprocess_cmd,
            prepostprocess_dropout,
            name=name + '_pre_ffn'),
        d_inner_hid,
        d_model,
        relu_dropout,
        hidden_act,
        param_initializer=param_initializer,
        name=name + '_ffn')
    return post_process_layer(
        attn_output,
        ffd_output,
        postprocess_cmd,
        prepostprocess_dropout,
        name=name + '_post_ffn')


def encoder(enc_input,
            attn_bias,
            n_layer,
            n_head,
            d_key,
            d_value,
            d_model,
            d_inner_hid,
            prepostprocess_dropout,
            attention_dropout,
            relu_dropout,
            hidden_act,
            preprocess_cmd="n",
            postprocess_cmd="da",
            param_initializer=None,
            name=''):
    """
    The encoder is composed of a stack of identical layers returned by calling
    encoder_layer.
    """
    for i in range(n_layer):
        enc_output = encoder_layer(
            enc_input,
            attn_bias,
            n_head,
            d_key,
            d_value,
            d_model,
            d_inner_hid,
            prepostprocess_dropout,
            attention_dropout,
            relu_dropout,
            hidden_act,
            preprocess_cmd,
            postprocess_cmd,
            param_initializer=param_initializer,
            name=name + '_layer_' + str(i))
        enc_input = enc_output
    enc_output = pre_process_layer(
        enc_output, preprocess_cmd, prepostprocess_dropout, name="post_encoder")

    return enc_output


def knowledge_task(enc_output:layers.data, mask_pos_list:List[List[int]],type_list:List[List[str]], entities_size:int,
                   property_size:int,name='knowledge'):
    """
    the knowledge task for pre-train stage:
    There are 2 types for knowledge, one for S, one for P
    1. mask entity to predict
    2. mask property to predict

    :param enc_output:
    :param mask_pos:
    :param type_list:
    :return:
    """
    assert len(mask_pos_list) == len(type_list), "InputError: Type list must have the same length as mask_pos_list"
    s_fc = lambda x: layers.fc(x, entities_size,
                               param_attr=fluid.ParamAttr(name=name+"_S_w"),
                               bias_attr=fluid.ParamAttr(name=name+"_S_b"),
                               name=name+"_S")
    p_fc = lambda x: layers.fc(x, property_size,
                               param_attr=fluid.ParamAttr(name=name+"_P_w"),
                               bias_attr=fluid.ParamAttr(name=name+"_P_b"),
                               name=name+"_P")

    S_res_list = []
    P_res_list = []
    for batch_id, (mask_pos_sub,type_sub) in enumerate(zip(mask_pos_list, type_list)):
        for mask_pos, mask_type in zip(mask_pos_sub, type_sub):
            tmp = layers.slice(enc_output, axes=[0,1,2], starts=[batch_id, mask_pos, 0], ends=[batch_id+1, mask_pos+1, INT_MAX])
            if mask_type.lower() == 'p':
                P_res_list.append(p_fc(tmp))
            elif mask_type.lower() == 's':
                S_res_list.append(s_fc(tmp))
    S_output = layers.concat(S_res_list, axis=0)
    P_res_list = layers.concat(P_res_list, axis=0)
    return S_output, P_res_list


def decoder_layer(dec_input,
                  enc_output,
                  slf_attn_bias,
                  dec_enc_attn_bias,
                  n_head,
                  d_key,
                  d_value,
                  d_model,
                  d_inner_hid,
                  prepostprocess_dropout,
                  attention_dropout,
                  relu_dropout,
                  hidden_act,
                  preprocess_cmd="n",
                  postprocess_cmd="da",
                  cache=None,
                  name=''):
    """ The layer to be stacked in decoder part.
    The structure of this module is similar to that in the encoder part except
    a multi-head attention is added to implement encoder-decoder attention.
    """
    slf_attn_output = multi_head_attention(
        pre_process_layer(dec_input,
                          preprocess_cmd,
                          prepostprocess_dropout,
                          name=name+"_pre_slf_att"),
        None,
        None,
        slf_attn_bias,
        d_key,
        d_value,
        d_model,
        n_head,
        attention_dropout,
        cache=cache,
        name=name + '_slf_multi_head_att')
    slf_attn_output = post_process_layer(
        dec_input,
        slf_attn_output,
        postprocess_cmd,
        prepostprocess_dropout, name=name + '_post_slf_att')

    enc_attn_output = multi_head_attention(
        pre_process_layer(slf_attn_output, preprocess_cmd,
                          prepostprocess_dropout,
                          name=name + "_pre_enc_att"),
        enc_output,
        enc_output,
        dec_enc_attn_bias,
        d_key,
        d_value,
        d_model,
        n_head,
        attention_dropout,
        cache=cache,
        name=name + "_enc_multi_head_attn")
    enc_attn_output = post_process_layer(
        slf_attn_output,
        enc_attn_output,
        postprocess_cmd,
        prepostprocess_dropout,
        name="_post_enc_att")
    ffd_output = positionwise_feed_forward(
        pre_process_layer(enc_attn_output, preprocess_cmd,
                          prepostprocess_dropout,
                          name=name + '_pre_ffn'),
        d_inner_hid,
        d_model,
        relu_dropout,
        hidden_act=hidden_act,
        name=name + '_ffn')
    dec_output = post_process_layer(
        enc_attn_output,
        ffd_output,
        postprocess_cmd,
        prepostprocess_dropout, name=name + '_post_ffn')
    return dec_output


def decoder(dec_input,
            enc_output,
            dec_slf_attn_bias,
            dec_enc_attn_bias,
            n_layer,
            n_head,
            d_key,
            d_value,
            d_model,
            d_inner_hid,
            prepostprocess_dropout,
            attention_dropout,
            relu_dropout,
            hidden_act,
            preprocess_cmd,
            postprocess_cmd,
            caches=None,
            name=''):
    """
    The decoder is composed of a stack of identical decoder_layer layers.
    """
    for i in range(n_layer):
        dec_output = decoder_layer(
            dec_input,
            enc_output,
            dec_slf_attn_bias,
            dec_enc_attn_bias,
            n_head,
            d_key,
            d_value,
            d_model,
            d_inner_hid,
            prepostprocess_dropout,
            attention_dropout,
            relu_dropout,
            hidden_act,
            preprocess_cmd,
            postprocess_cmd,
            cache=None if caches is None else (caches[i], i),
            name=name + "_layer_" + str(i))
        dec_input = dec_output
    dec_output = pre_process_layer(dec_output, preprocess_cmd,
                                   prepostprocess_dropout,
                                   name=name+'post_decoder')
    return dec_output

if __name__ == '__main__':
    main_program = fluid.Program()
    startup_program = fluid.Program()

    # parameters
    max_len = 128
    d_model = 128
    batch_size = 8
    d_inner_hid = d_model * 4
    n_head = 8
    n_layer = 6
    mask_pos_list = [[1,2]]
    type_list = [['S', 'P']]
    entities_size = 32
    property_size = 16



    with fluid.program_guard(main_program, startup_program):
        q = fluid.layers.data(name='query', shape=[None, max_len, d_model], dtype="float64")
        k = fluid.layers.data(name='key', shape=[None, max_len, d_model], dtype="float64")
        v = fluid.layers.data(name='value', shape=[None, max_len, d_model], dtype="float64")
        attn_bias = fluid.layers.data(name='attn_bias',
                                      shape=[None, n_head, max_len, max_len], dtype="float64")
        #     layers.Print(attn_bias)
        attn_bias1 = fluid.layers.data(name='attn_bias1',
                                       shape=[None, n_head, max_len, max_len], dtype="float64")
        attn_output = multi_head_attention(q, k, v, attn_bias1, d_model,
                                           d_model, d_model, n_head)

        ffd_output = positionwise_feed_forward(
            attn_output, d_inner_hid, d_model, 0.,hidden_act='relu'
        )
        output1 = post_process_layer(attn_output, ffd_output, '',
                                     0.0)
        #     output = encoder_layer(q, attn_bias1, 1, d_model, d_model,
        #                      d_model, d_inner_hid, prepostprocess_dropout=0.,
        #                      attention_dropout=0., relu_dropout=0.)
        output = encoder(q, attn_bias1, n_layer, n_head, d_model, d_model,
                               d_model, d_inner_hid,hidden_act='relu', prepostprocess_dropout=0,
                               attention_dropout=0, relu_dropout=0, name='encoder')

        knowledge_S_output,knowledge_P_output  = knowledge_task(output, mask_pos_list, type_list,entities_size, property_size)
    # work
    INF = 2 ^ 32 + 1
    input_data = np.random.rand(batch_size, max_len, d_model)
    attn_data = np.zeros((batch_size, n_head, max_len))
    attn_data[:, :, 4:] = -INF
    attn_data = np.zeros((batch_size, n_head, max_len, max_len)) + np.expand_dims(attn_data, axis=2)

    attn_data1 = np.ones((batch_size, n_head, max_len))
    attn_data1[:, :, 4:] = 0
    a = np.expand_dims(attn_data1, axis=2)
    b = np.expand_dims(attn_data1, axis=3)
    attn_data1 = np.matmul(b, a)
    attn_data1[attn_data1 == 0] = -INF
    attn_data1[attn_data1 > -INF] = 0

    # attn_data1 = mask_generator(4,4,max_len)
    # print(attn_data1.shape)

    # Executor
    place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(startup_program)
    start = time.time()
    res = exe.run(main_program,
                     feed={"query": input_data, "key": input_data,
                           "value": input_data, "attn_bias": attn_data,
                           "attn_bias1": attn_data1},
                     fetch_list=[output,knowledge_P_output, knowledge_S_output])
    end = time.time()
    print(res[2])
    print(res[1])
    print("Time Cost: {}s.".format(end-start))
