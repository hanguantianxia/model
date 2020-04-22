import json
import re
import os
import time
import logging
from collections import namedtuple

from paddle.fluid import core
import paddle.fluid as fluid
import numpy as np

Doc = namedtuple("Document", ("id", "title", "content"))
SegDoc = namedtuple("Seg_Document",("id", "title", "content", "seg_content"))

def read_origin_data(filename, *, limit=None):
    """
    read the origin work from multi-lines json
    return the list of json with length of limit

    :param filename: the json filename
    :param limit: only read the limit number of lines of work(if it's None type return )
    :return:
    """
    papers = []
    with open(filename, 'r', encoding='utf-8') as f:
        for id ,line in enumerate(f.readlines()):
            dic = json.loads(line)
            papers.append(dic)
            if limit is not None and id > limit-2:
                break
    return papers

def dict_doc(documents)->dict:
    """
    transfer the origin document list into document dictionary

    :param documents: the list of document
    :return:
    """
    doc_dict = {}
    for doc in documents:
        doc_id = doc.get('id')
        sentences = doc.get('sentences')
        title = sentences[0]
        content = "".join(sentences[1:])
        doc_dict[doc_id] = Doc(doc_id, title, content)
    return doc_dict

def pre_text(text: str) -> str:
    """
    do some dirty work for the string
    1. delete the punctuation
    2. sub the number into <NUM>

    :param text:
    :return:
    """
    punc = re.compile(
        r',|/|：|;|:|\'|`|\[|\]|<|>|\?|:|"|\{|\}|\~|!|@|#|\$|%|\^|&|\(|\)|-|=|\_|\+|，|、|‘|’|【|】|·|！|”|“| |…|（|）|」|「|《|》|。|，|\.|。|;|；|\+',
        re.S)
    res = punc.sub(" ", text)

    # num = re.compile("\d+")
    # res = num.sub("<NUM>", res)

    return res

def time_clock(func):
    def compute_time_clock(*args, **kwargs):
        start = time.time()
        res = func(*args, **kwargs)
        end = time.time()
        print("Implement function %s, using %.2f s"%(func.__name__ ,end-start))
        return res
    return compute_time_clock

@time_clock
def hello_word():
    print("Hello world")
    time.sleep(1)
    return 0


def get_programname(file):
    prgram_filename = os.path.split(file)[1]
    return os.path.splitext(prgram_filename)[0]

def read_json(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def normal_leven(str1, str2):
    len_str1 = len(str1) + 1
    len_str2 = len(str2) + 1
    # create matrix
    matrix = [0 for n in range(len_str1 * len_str2)]
    # init x axis
    for i in range(len_str1):
        matrix[i] = i
    # init y axis
    for j in range(0, len(matrix), len_str1):
        if j % len_str1 == 0:
            matrix[j] = j // len_str1

    for i in range(1, len_str1):
        for j in range(1, len_str2):
            if str1[i - 1] == str2[j - 1]:
                cost = 0
            else:
                cost = 1
            matrix[j * len_str1 + i] = min(matrix[(j - 1) * len_str1 + i] + 1,
                                           matrix[j * len_str1 + (i - 1)] + 1,
                                           matrix[(j - 1) * len_str1 + (i - 1)] + cost)

    return matrix[-1]

def time_clock(func):
    def compute_time_clock(*args, **kwargs):
        start = time.time()
        res = func(*args, **kwargs)
        end = time.time()
        print("Implement function %s, using %.2f s"%(func.__name__ ,end-start))
        return res
    return compute_time_clock

@time_clock
def hello_word():
    print("Hello world")
    time.sleep(1)
    return 0


def set_base(file,keyword='result'):
    """

    :param file:
    :param keyword:
    :return:
    """
    base_flle = get_programname(file)
    base_dir = os.path.join(".", keyword, base_flle)
    if not os.path.exists(base_dir):
        os.mkdir(base_dir)
    return base_dir


def json_saver(obj, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(obj, f)




@time_clock
def read_train_data(filename) :
    """
    read the train.txt and get the list of all work
    :param filename:
    :return:
    """
    corpus = []  # 存储所有对话的列表
    with open(filename, 'r', encoding='utf-8') as f:
        data = list()  # 存储一个对话的列表
        for line in f:
            if line == '\n':
                corpus.append(data)
                data = list()
                continue
            raw_data = line.strip().split('\t')

            processed_data = []
            for idx, item in enumerate(raw_data):
                try:
                    temp = eval(item)
                except:
                    temp = item
                finally:
                    if idx == 2 and type(temp)==int:
                        temp = str(temp)
                    processed_data.append(temp)
            data.append(processed_data)
    return corpus

def log(filename= "test.log",level=logging.DEBUG):
    #创建logger，如果参数为空则返回root logger
    logger = logging.getLogger("encocder pretrain")
    logger.setLevel(level)  #设置logger日志等级

    #这里进行判断，如果logger.handlers列表为空，则添加，否则，直接去写日志
    if not logger.handlers:
        #创建handler
        fh = logging.FileHandler(filename,encoding="utf-8")
        ch = logging.StreamHandler()
        #设置输出日志格式
        LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
        DATE_FORMAT = "%m/%d/%Y %H:%M:%S %p"
        logger.propagate = False
        formatter = logging.Formatter(
            fmt=LOG_FORMAT,
            datefmt=DATE_FORMAT
            )

        #为handler指定输出格式
        fh.setFormatter(formatter)

        #为logger添加的日志处理器
        logger.addHandler(fh)

    return logger #直接返回logger

class Clock():

    def __init__(self):
        time.time()


def save_model(filename, param_name_list, opt_var_name_list, name=''):
    save_model_file = os.path.join(filename, name + "model_stage_0")
    save_opt_state_file = os.path.join(filename, name + "opt_state_stage_0")


    model_stage_0 = {}
    for name in param_name_list:
        t = np.asarray(fluid.global_scope().find_var(name).get_tensor())
        model_stage_0[name] = t
    np.savez(save_model_file, **model_stage_0)

    opt_state_stage_0 = {}
    for name in opt_var_name_list:
        t_data = np.asarray(fluid.global_scope().find_var(name).get_tensor())
        opt_state_stage_0[name] = t_data
    np.savez(save_opt_state_file, **opt_state_stage_0)
    info_msg = "Finish saving the parameter. "
    return info_msg

def load_model(model_init_file, param_name_list, place, opt_state_init_file='',datatype='float32'):
    """ init model """

    try:
        model_init = np.load(model_init_file)
    except:
        print("load init model failed", model_init_file)
        raise Exception("load init model failed")

    print("load init model")
    loading_msg = []
    for name in param_name_list:
        try:
            t = fluid.global_scope().find_var(name).get_tensor()
            load_param = model_init[str(name)]
            if load_param.shape == np.asarray(t).shape:
                t.set(load_param.astype(datatype), place)
        except AttributeError as e:
            loading_msg.append(str(e) + "%s exist not in this model and cannot be load!"%name)
        except KeyError as e:
            loading_msg.append(str(e) + "%s exist not in this model and cannot be load!"%name)

    return loading_msg

    # load opt state
    if opt_state_init_file != "":
        print("begin to load opt state")
        opt_state_data = np.load(opt_state_init_file)
        for k, v in opt_state_data.items():
            t = fluid.global_scope().find_var(str(k)).get_tensor()
            t.set(v, place)
        print("set opt state finished")

    print("init model parameters finshed")

def write_iterable(filename, iterable_obj):
    with open(filename, 'w', encoding='utf8') as f:
        for i in iterable_obj:
            f.write(str(i)+'\n')



if __name__ == '__main__':
    res = hello_word()
    print(res)