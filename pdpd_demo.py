import paddle.fluid as fluid
import numpy as np


SEQ_MAX_LEN = 512
HIDDEN_SIZE = 768
########################################################################################
train_prog = fluid.Program()
startup_prog = fluid.Program()
pos_embed = np.load("pos_emb.npy")
pos_embed = np.array(pos_embed, dtype=np.float64)

with fluid.program_guard(train_prog, startup_prog):
    pos_ids = fluid.layers.data(name="pos_ids", shape=[None, SEQ_MAX_LEN], dtype='int64')
    pos_embed_layer = fluid.embedding(pos_ids, size=[SEQ_MAX_LEN,HIDDEN_SIZE],
                                             dtype='float64',
                                             padding_idx=0,
                                             param_attr=fluid.ParamAttr(name="position_embedding",
                                             initializer=fluid.initializer.NumpyArrayInitializer(pos_embed),
                                             trainable=False))

place = fluid.CPUPlace()
exe = fluid.Executor(place)
exe.run(startup_prog)

data_input = np.array([[i for i in range(512)]], dtype=np.int64)

res= exe.run(train_prog, feed={"pos_ids":data_input},
        fetch_list=[pos_embed_layer])

print(res)