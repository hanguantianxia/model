# author:hjt
# contact: hanguantianxia@sina.com
# datetime:2020/4/11 17:21
# software: PyCharm

"""

"""
import numpy as np

import paddle.fluid as fluid
import paddle.fluid.layers as layers

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
        k = layers.fc(input=keys,
                      size=d_key * n_head,
                      num_flatten_dims=2,
                      param_attr=fluid.ParamAttr(
                          name=name + '_key_fc.w_0',
                          initializer=param_initializer),
                      bias_attr=name + '_key_fc.b_0')
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
        product = layers.matmul(x=scaled_q, y=k, transpose_y=True)
        if attn_bias:
            product += attn_bias
        weights = layers.softmax(product)
        ############################
        # add code
        layers.Print(attn_bias,message="The content of input layer:")

        attn_mask = attn_bias == 0
        attn_mask = layers.cast(attn_mask, 'float64')
        layers.Print(weights)
        weights = layers.elementwise_mul(attn_mask, weights)
        layers.Print(weights)

#         weights = layers.elementwise_mul(weights, attn_mask)
        ############################
        if dropout_rate:
            weights = layers.dropout(
                weights,
                dropout_prob=dropout_rate,
                dropout_implementation="upscale_in_train",
                is_test=False)
        out = layers.matmul(weights, v)
        return out

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

if __name__ == '__main__':

    main_program = fluid.Program()
    startup_program = fluid.Program()

    # parameters
    max_len = 16
    d_model = 8
    batch_size = 1
    n_head = 2

    with fluid.program_guard(main_program, startup_program):
        q = fluid.layers.data(name='query', shape=[None, max_len, d_model], dtype="float64")
        k = fluid.layers.data(name='key', shape=[None, max_len, d_model], dtype="float64")
        v = fluid.layers.data(name='value', shape=[None, max_len, d_model], dtype="float64")
        attn_bias = fluid.layers.data(name='attn_bias',
                                      shape=[None, n_head, max_len, max_len], dtype="float64")
        #     layers.Print(attn_bias)
        attn_bias1 = fluid.layers.data(name='attn_bias1',
                                       shape=[None, n_head, max_len, max_len], dtype="float64")

        output = multi_head_attention(q, k, v, attn_bias, d_model,
                                      d_model, d_model, n_head)
        output1 = multi_head_attention(q, k, v, attn_bias1, d_model,
                                       d_model, d_model, n_head)

        soft_max = layers.softmax(attn_bias1)
        layers.Print(soft_max)
    #     layers.Print(output)

    # work
    INF = -2^32 + 1
    input_data = np.random.rand(batch_size, max_len, d_model)
    attn_data = np.zeros((batch_size, n_head, max_len))
    attn_data[:, :, 4:] = -INF
    attn_data = np.zeros((batch_size, n_head, max_len, max_len)) + np.expand_dims(attn_data, axis=2)

    attn_data1 = np.zeros((batch_size, n_head, max_len))
    attn_data1[:, :, 4:] = -INF
    a = np.expand_dims(attn_data1, axis=2)
    b = np.expand_dims(attn_data1, axis=3)
    attn_data1 = np.matmul(b, a)
    attn_data1[np.isnan(attn_data1)] = -INF

    print(attn_data1.shape)

    # Executor
    place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(startup_program)
    output, output1, soft_max = exe.run(main_program,
                                        feed={"query": input_data, "key": input_data, "value": input_data,
                                              "attn_bias": attn_data, "attn_bias1": attn_data1},
                                        fetch_list=[output, output1, soft_max])
    # print(output)
    # print(output1)

