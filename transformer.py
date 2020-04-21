# author:hjt
# contact: hanguantianxia@sina.com
# datetime:2020/4/11 21:32
# software: PyCharm

"""

"""

from functools import partial
import numpy as np

import paddle.fluid as fluid
import paddle.fluid.layers as layers
from paddle.fluid.layers.utils import map_structure

dropout_seed = 0


def wrap_layer_with_block(layer, block_idx):
    """
    Make layer define support indicating block, by which we can add layers
    to other blocks within current block. This will make it easy to define
    cache among while loop.
    """

    class BlockGuard(object):
        """
        BlockGuard class.

        BlockGuard class is used to switch to the given block in a program by
        using the Python `with` keyword.
        """

        def __init__(self, block_idx=None, main_program=None):
            self.main_program = fluid.default_main_program(
            ) if main_program is None else main_program
            self.old_block_idx = self.main_program.current_block().idx
            self.new_block_idx = block_idx

        def __enter__(self):
            self.main_program.current_block_idx = self.new_block_idx

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.main_program.current_block_idx = self.old_block_idx
            if exc_type is not None:
                return False  # re-raise exception
            return True

    def layer_wrapper(*args, **kwargs):
        with BlockGuard(block_idx):
            return layer(*args, **kwargs)

    return layer_wrapper


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
                         static_kv=False):
    """
    Multi-Head Attention. Note that attn_bias is added to the logit before
    computing softmax activiation to mask certain selected positions so that
    they will not considered in attention weights.

    queries: [batch_size, max_len, d_mdoel]


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
                      bias_attr=False,
                      num_flatten_dims=2)
        # For encoder-decoder attention in inference, insert the ops and vars
        # into global block to use as cache among beam search.
        fc_layer = wrap_layer_with_block(
            layers.fc, fluid.default_main_program().current_block(
            ).parent_idx) if cache is not None and static_kv else layers.fc
        k = fc_layer(
            input=keys,
            size=d_key * n_head,
            bias_attr=False,
            num_flatten_dims=2)
        v = fc_layer(
            input=values,
            size=d_value * n_head,
            bias_attr=False,
            num_flatten_dims=2)
        return q, k, v

    def __split_heads_qkv(queries, keys, values, n_head, d_key, d_value):
        """
        Reshape input tensors at the last dimension to split multi-heads
        and then transpose. Specifically, transform the input tensor with shape
        [bs, max_sequence_length, n_head * hidden_dim] to the output tensor
        with shape [bs, n_head, max_sequence_length, hidden_dim].
        """
        # The value 0 in shape attr means copying the corresponding dimension
        # size of the input as the output dimension size.
        reshaped_q = layers.reshape(
            x=queries, shape=[0, 0, n_head, d_key], inplace=True)
        # permuate the dimensions into:
        # [batch_size, n_head, max_sequence_len, hidden_size_per_head]
        q = layers.transpose(x=reshaped_q, perm=[0, 2, 1, 3])
        # For encoder-decoder attention in inference, insert the ops and vars
        # into global block to use as cache among beam search.
        reshape_layer = wrap_layer_with_block(
            layers.reshape,
            fluid.default_main_program().current_block(
            ).parent_idx) if cache is not None and static_kv else layers.reshape
        transpose_layer = wrap_layer_with_block(
            layers.transpose,
            fluid.default_main_program().current_block().
            parent_idx) if cache is not None and static_kv else layers.transpose
        reshaped_k = reshape_layer(
            x=keys, shape=[0, 0, n_head, d_key], inplace=True)
        k = transpose_layer(x=reshaped_k, perm=[0, 2, 1, 3])
        reshaped_v = reshape_layer(
            x=values, shape=[0, 0, n_head, d_value], inplace=True)
        v = transpose_layer(x=reshaped_v, perm=[0, 2, 1, 3])

        if cache is not None:  # only for faster inference
            cache_, i = cache
            if static_kv:  # For encoder-decoder attention in inference
                cache_k, cache_v = cache_["static_k"], cache_["static_v"]
                # To init the static_k and static_v in global block.
                static_cache_init = wrap_layer_with_block(
                    layers.assign,
                    fluid.default_main_program().current_block().parent_idx)
                static_cache_init(
                    k,
                    fluid.default_main_program().global_block().var(
                        "static_k_%d" % i))
                static_cache_init(
                    v,
                    fluid.default_main_program().global_block().var(
                        "static_v_%d" % i))
                k, v = cache_k, cache_v
            else:  # For decoder self-attention in inference
                # use cache and concat time steps.
                cache_k, cache_v = cache_["k"], cache_["v"]
                k = layers.concat([cache_k, k], axis=2)
                v = layers.concat([cache_v, v], axis=2)
                cache_["k"], cache_["v"] = (k, v)
        return q, k, v

    def __combine_heads(x):
        """
        Transpose and then reshape the last two dimensions of inpunt tensor x
        so that it becomes one dimension, which is reverse to __split_heads.
        """
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
        """
        product = layers.matmul(x=q, y=k, transpose_y=True, alpha=d_key**-0.5)
        if attn_bias:
            product += attn_bias
        weights = layers.softmax(product)
        layers.Print(weights)

        layers.Print(weights)
        if dropout_rate:
            weights = layers.dropout(
                weights,
                dropout_prob=dropout_rate,
                seed=dropout_seed,
                is_test=False)
        out = layers.matmul(weights, v)
        return out

    q, k, v = __compute_qkv(queries, keys, values, n_head, d_key, d_value)
    q, k, v = __split_heads_qkv(q, k, v, n_head, d_key, d_value)

    ctx_multiheads = scaled_dot_product_attention(q, k, v, attn_bias, d_model,
                                                  dropout_rate)

    out = __combine_heads(ctx_multiheads)

    # Project back to the model size.
    proj_out = layers.fc(input=out,
                         size=d_model,
                         bias_attr=False,
                         num_flatten_dims=2)
    return proj_out


def positionwise_feed_forward(x, d_inner_hid, d_hid, dropout_rate):
    """
    Position-wise Feed-Forward Networks.
    This module consists of two linear transformations with a ReLU activation
    in between, which is applied to each position separately and identically.
    """
    hidden = layers.fc(input=x,
                       size=d_inner_hid,
                       num_flatten_dims=2,
                       act="relu")
    if dropout_rate:
        hidden = layers.dropout(
            hidden, dropout_prob=dropout_rate, seed=dropout_seed, is_test=False)
    out = layers.fc(input=hidden, size=d_hid, num_flatten_dims=2)
    return out


def pre_post_process_layer(prev_out, out, process_cmd, dropout_rate=0.):
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
            out = layers.layer_norm(
                out,
                begin_norm_axis=len(out.shape) - 1,
                param_attr=fluid.initializer.Constant(1.),
                bias_attr=fluid.initializer.Constant(0.))
        elif cmd == "d":  # add dropout
            if dropout_rate:
                out = layers.dropout(
                    out,
                    dropout_prob=dropout_rate,
                    seed=dropout_seed,
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
                  preprocess_cmd="",
                  postprocess_cmd="dan"):
    """The encoder layers that can be stacked to form a deep encoder.
    This module consits of a multi-head (self) attention followed by
    position-wise feed-forward networks and both the two components companied
    with the post_process_layer to add residual connection, layer normalization
    and droput.
    """
    attn_output = multi_head_attention(
        enc_input, # query
        None,
        None,
        attn_bias,
        d_key,
        d_value,
        d_model,
        n_head,
    )
    attn_output = post_process_layer(enc_input, attn_output, postprocess_cmd,
                                     prepostprocess_dropout)
    # print(attn_output.shape)
    ffd_output = positionwise_feed_forward(
        attn_output, d_inner_hid, d_model, relu_dropout
    )
    # print(ffd_output.shape)
    # layers.Print(ffd_output)


    output = post_process_layer(attn_output, ffd_output, postprocess_cmd,
                              prepostprocess_dropout)
    # layers.Print(output)

    return output


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
            preprocess_cmd="",
            postprocess_cmd="dan"):
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
            preprocess_cmd,
            postprocess_cmd)
        enc_input = enc_output
    enc_output = enc_input
    return enc_output


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
                  preprocess_cmd,
                  postprocess_cmd,
                  cache=None):
    """ The layer to be stacked in decoder part.
    The structure of this module is similar to that in the encoder part except
    a multi-head attention is added to implement encoder-decoder attention.
    """
    slf_attn_output = multi_head_attention(
        pre_process_layer(dec_input, preprocess_cmd, prepostprocess_dropout),
        None,
        None,
        slf_attn_bias,
        d_key,
        d_value,
        d_model,
        n_head,
        attention_dropout,
        cache=cache)
    slf_attn_output = post_process_layer(
        dec_input,
        slf_attn_output,
        postprocess_cmd,
        prepostprocess_dropout, )
    enc_attn_output = multi_head_attention(
        pre_process_layer(slf_attn_output, preprocess_cmd,
                          prepostprocess_dropout),
        enc_output,
        enc_output,
        dec_enc_attn_bias,
        d_key,
        d_value,
        d_model,
        n_head,
        attention_dropout,
        cache=cache,
        static_kv=True)
    enc_attn_output = post_process_layer(
        slf_attn_output,
        enc_attn_output,
        postprocess_cmd,
        prepostprocess_dropout, )
    ffd_output = positionwise_feed_forward(
        pre_process_layer(enc_attn_output, preprocess_cmd,
                          prepostprocess_dropout),
        d_inner_hid,
        d_model,
        relu_dropout, )
    dec_output = post_process_layer(
        enc_attn_output,
        ffd_output,
        postprocess_cmd,
        prepostprocess_dropout, )
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
            preprocess_cmd,
            postprocess_cmd,
            caches=None):
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
            preprocess_cmd,
            postprocess_cmd,
            cache=None if caches is None else (caches[i], i))
        dec_input = dec_output
    dec_output = pre_process_layer(dec_output, preprocess_cmd,
                                   prepostprocess_dropout)
    return dec_output

if __name__ == '__main__':
    main_program = fluid.Program()
    startup_program = fluid.Program()

    # parameters
    max_len = 4
    d_model = 8
    batch_size = 2
    d_inner_hid = d_model * 4
    n_head = 3
    n_layer = 6

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
            attn_output, d_inner_hid, d_model, 0.
        )
        output1 = post_process_layer(attn_output, ffd_output, '',
                                     0.0)
        #     output = encoder_layer(q, attn_bias1, 1, d_model, d_model,
        #                      d_model, d_inner_hid, prepostprocess_dropout=0.,
        #                      attention_dropout=0., relu_dropout=0.)
        # enc_output = encoder(q, attn_bias1, n_layer, n_head, d_model, d_model,
        #                        d_model, d_inner_hid, prepostprocess_dropout=0,
        #                        attention_dropout=0, relu_dropout=0)
        # dec_output = decoder(q, enc_output)

    # work
    INF = 2**32 -1
    input_data = np.random.rand(batch_size, max_len, d_model)
    attn_data = np.zeros((batch_size, n_head, max_len))
    attn_data[:, :, 2:] = -INF
    attn_data = np.zeros((batch_size, n_head, max_len, max_len)) + np.expand_dims(attn_data, axis=2)

    attn_data1 = np.ones((batch_size, n_head, max_len))
    attn_data1[:, :, 2:] = 0
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
    output = exe.run(main_program,
                     feed={"query": input_data, "key": input_data,
                           "value": input_data, "attn_bias": attn_data,
                           "attn_bias1": attn_data1},
                     fetch_list=[attn_output])
    # print(output[0])


