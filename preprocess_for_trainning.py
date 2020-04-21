# author:hjt
# contact: hanguantianxia@sina.com
# datetime:2020/4/6 17:11
# software: PyCharm

"""

"""
import json
import os
import re
from collections import defaultdict, namedtuple
from typing import Dict

role_list = ["Bot", "User"]
Situation = namedtuple("Situation", ["date", "workday", "time", "location", "theme"])


def sub_space(string):
    return re.sub(" ", "", string)


def remove_punctuation(line):
    """remove_punctuation"""
    # print(line)
    # return re.sub('[^\u4e00-\u9fa5^a-z^A-Z^0-9]', '', line)
    return re.sub("[\s+\.\!\/_,$%^*(+\"\')]+|[+——()?【】“”！，。？、~@#￥%……&*（）]+", "", line)


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
        for id, line in enumerate(f.readlines()):
            dic = json.loads(line)
            papers.append(dic)
            if limit is not None and id > limit - 2:
                break
    return papers


def replace_word(raw_str: str):
    raw_str = raw_str.replace("poi", "兴趣点")
    return raw_str


def process_situation(raw_situation: str):
    """
     获取时间,地点,主题,日期,周几
    :param raw_situation:
    :return: Tuple[Date, Workday, time, location, theme]
    """
    space_sub = re.compile(r' ', re.S)
    time_pat = re.compile(r"..\d+:\d+", re.S)
    sub_time = re.compile(r"聊天时间:", re.S)
    date_pat = re.compile(r"\d{4}-\d{1,2}-\d{1,2}")
    sub_date = re.compile(r"聊天日期:", re.S)
    theme_pat = re.compile(r"聊天主题:(.*)", re.S)
    workday_pat = re.compile(r'星期.', re.S)
    local_sub = re.compile(r'，', re.S)

    raw_situation = space_sub.sub("", raw_situation)  # 去空格
    item_time = re.findall(time_pat, raw_situation)  # 找时间
    raw_situation = time_pat.sub("", raw_situation)  # 去时间
    raw_situation = sub_time.sub("", raw_situation)  # 去时间

    item_date = date_pat.findall(raw_situation)
    raw_situation = date_pat.sub("", raw_situation)  # 去date
    raw_situation = sub_date.sub("", raw_situation)  # 去date

    item_theme = theme_pat.findall(raw_situation)
    raw_situation = theme_pat.sub("", raw_situation)  # 去date

    item_work = workday_pat.findall(raw_situation)
    location = [local_sub.sub('', raw_situation)]

    situation = Situation(item_date, item_work, item_time, location, item_theme)
    return situation


def judge_role(first_goal: str):
    """

    :param first_goal:
    :return:
    """
    sub_space = re.compile(r" ", re.S)
    pattern = re.compile(r"User主动", re.S)
    pattern2 = re.compile(r"用户主动", re.S)

    content = sub_space.sub("", first_goal)

    if re.findall(pattern, content) != [] or re.findall(pattern2, content) != []:
        return 1
    else:
        return 0


def preprocess_response(responce: str):
    """

    :param responce: string of responce
    :return:  Tuple[responce_sub, goal_or_not]
    """
    tag = re.compile(r'\[(\d*)\]', re.S)
    res = re.findall(tag, responce)
    if res != []:
        return tag.sub('', responce).strip(), int(res[0])
    else:
        return responce.strip(), 0


def preprocess_usr_profile(usr_prof: Dict):
    """
    merge the usr profile
    :param usr_prof:
    :return: processed usr_profile
    """
    p_p_key = ["喜欢的电影", "喜欢的明星", "喜欢的poi", "喜欢的音乐", "喜欢的新闻", "接受的电影", '喜欢的兴趣点']
    p_a_key = ["同意的新闻", "同意的音乐", "同意的美食", "同意的poi", "同意的电影", "接受的音乐", "同意的兴趣点"]
    n_key = ["没有接受的电影", "没有接受的音乐"]

    new_prof = defaultdict(list)
    for key, value in usr_prof.items():
        sub_key = sub_space(key)
        split_key = key.split(' ')
        if sub_key in p_p_key or sub_key in p_a_key:
            new_key = "active" + " " + split_key[-1]
        elif sub_key in n_key:
            new_key = "negative" + " " + split_key[-1]
        else:
            new_key = key
        new_key = replace_word(new_key)
        if type(value) == list:
            new_prof[new_key].extend(value)
        elif type(value) == str:
            new_prof[new_key].append(value)
        else:
            print(value)
    return new_prof


def str_tuple(*items, split_char='\t', split_string='\n'):
    """

    :param items:
    :return:
    """
    str_list = [str(item).lower() for item in items]
    res = split_char.join(str_list)
    res += split_string
    return res


def repalce(raw_data: str):
    pro_data = raw_data.replace("User", "用户")
    pro_data = pro_data.replace(",", "，")
    return pro_data


def sub_info(string, token="<info>", remain_space=False):
    string = repalce(string)

    pattern0 = re.compile(r'\[\d*\]')
    pattern1 = re.compile(r' ')
    pattern2 = re.compile(r"『(.*?)』")
    pattern3 = re.compile(r'『.*?(?=,)')

    res = pattern0.sub("", string)
    if not remain_space:
        res = pattern1.sub("", res)
    res = pattern2.sub(token, res)
    res = pattern3.sub(token, res)

    pattern2 = re.compile(r"『(.*?)』")
    pattern3 = re.compile(r'『(.*?(?=,))')
    entities1 = pattern2.findall(string)
    entities2 = pattern3.findall(string)
    entities = entities1 + entities2
    entities = [pattern1.sub("", item) for item in entities]
    return res, entities


def split_goal(goal):
    pattern = re.compile(r"\((.*?)\)", re.S)
    sub_content = re.compile(r"\(.*?\)", re.S)
    goal_type = re.sub(sub_content, '', goal)
    goal_content = re.findall(pattern, goal)
    if goal_content == []:
        goal_content = ''
    else:
        goal_content = goal_content[0]
    return goal_type, remove_punctuation(repalce(goal_content))


def process_goal(goal_item, situation):
    """
    ## Goal处理方案
    - 问答：分为一类就是 “用户主动按<info>问<info>的信息Bot回答用户满意并好评”
    - 关于明星的聊天，按照<info>的个数来分类
    - 电影推荐：分为转到音乐，转到电影，拒绝，注意：**评论不作为实体,<info>**
    - 美食推荐：一个
    - 寒暄：一个
    - 提问：问新闻，问主演，问歌曲名，问电影名
    - 音乐推荐：统一为一个实体，那就是音乐，第一个<info>实体
    - 音乐播放：一个

    :param goal_item:
    :return:
    """
    pattern2 = re.compile(r"『.*?』")
    pattern3 = re.compile(r'『.*?(?=,)')


    goal, entities = sub_info(goal_item)
    goal_type, goal_content = split_goal(goal)
    entities = [item.strip() for item in entities]
    goal_content = pattern2.sub('',goal_content)
    goal_content = pattern3.sub('',goal_content)

    # process 提问
    if goal_type == "提问":
        if goal_content.find("新闻") != -1:
            goal_type += "新闻"
        elif goal_content.find("主演") != -1:
            goal_type += "主演"
            entities = [entities[0], entities[-1]]
        elif goal_content.find("歌曲名") != -1:
            goal_type += "歌曲名"
        elif goal_content.find("电影") != -1:
            goal_type += "电影名"

    # 处理音乐推荐
    # 按照<info>的数目进行分类
    if goal_type == "音乐推荐":
        if len(entities) > 3:
            entities = [entities[0], entities[-1]]
            if goal_content.find("播放") != -1:
                goal_type += "主动播放"
            else:
                goal_type += "二次"
        else:
            entities = [entities[0]]
            if goal_content.find("天气") != -1:
                goal_type += "天气"
            elif goal_content.find("暂时") != -1:
                goal_type += "拒绝"
            else:
                goal_type += "接受"



        entities = [re.sub("  ", ' ', item) for item in entities]

    # 处理寒暄
    if goal_type == "寒暄":
        entities = [situation[-1], situation[-3]]
        tmp = lambda x: x[0] if x != [] else ''
        entities = [tmp(item) for item in entities]

    # 兴趣点推荐
    if goal_type == "兴趣点推荐":
        entities = [entities[0], entities[1]]

    # 处理电影推荐，有喜欢这个关键字就说明会引出下一个id
    if goal_type == "电影推荐":
        if len(entities) > 3:
            entities = [entities[0], entities[-2]]
            goal_type += "拒绝"
        elif goal_content.find("喜欢") != -1:
            entities = [entities[0], entities[-1]]
            if goal_content.find("音乐") != -1:
                goal_type += "转音乐"
            elif goal_content.find("电影") != -1:
                goal_type += "转电影"
        else:
            entities = [entities[0]]
            if goal_content.find("暂时") != -1:
                goal_type += "不聊电影"
            else:
                goal_type += "接受"


    # 问答，分两类，一个是有两个实体，一个是有一个实体的
    if goal_type == "问答":
        if goal_content.find("按") != -1:
            entities = [entities[-1]]
            goal_type += "其他"
        elif goal_content.find("电影") != -1:
            goal_type += "电影"
        elif goal_content.find("音乐") != -1:
            goal_type += "音乐"

    # 新闻点播，实体为第一个
    if goal_type == "新闻点播":
        entities = [entities[0]]

    # 问日期，实体为空
    if goal_type == "问日期":
        entities = ['']

    # 新闻推荐，实体为第一个
    if goal_type == "新闻推荐":
        entities = [entities[0]]
        if goal_content.find("暂时") != -1:
            goal_type += "拒绝"
        else:
            goal_type += "接受"


    # 关于明星的聊天，实体为第一个
    if goal_type == "关于明星的聊天":
        if goal_content.find("主演") != -1:
            entities = [entities[0], entities[1]]
            #提取电影和明星
            goal_type += "电影"
        elif goal_content.find("主唱")!= -1:
            entities = [entities[0], entities[1]]
            goal_type += "音乐"
        elif goal_content.find("生日") != -1:
            entities = [entities[0]]
            goal_type += "生日"
        else:
            goal_type += "其他"
    return goal_type, entities


def goaltype2goal_tmp():
    goal2tmp = {
        '问日期':'用户 主动 问 日期 Bot 根据 <info> 的 <info> 回答 然后 用户 满足 并 好评',
        '再见':'',
        '问用户姓名':'Bot 主动 问 用户 姓名 一句 只能 问 <info>',
        '问用户性别':'Bot 主动 问 用户 性别 一句 只能 问 <info>',
        '问用户年龄':'Bot 主动 问 用户 年龄 一句 只能 问 <info>',
        '问用户爱好':'Bot 主动 问 用户 有 什么 爱好 根据 <info> 回答',
        '问天气':'用户 主动 用户 问 天气 根据 给定 知识 Bot 回复 完整 的 天气 信息 用户 满足 并 好评',
        '兴趣点推荐':'Bot 主动 Bot 推荐 在 <info> 订 <info> 用户 问 <info> 的 <info> <info> <info> Bot 逐一 回答 后 最终 用户 接受 并 提供 预订 信息 <info> 和 <info>',
        '音乐点播':'用户 主动 用户 点播 <info> Bot 播放 音乐 用户 听歌 并 做 简单 好评',
        '新闻点播':'用户 主动 用户 问 <info> 的 新闻 Bot 回答 <info> 用户 满足 并 好评',
        '问时间':'用户 主动 问 时间 Bot 回答 用户 满足 并 好评',
        '天气信息推送':'Bot 主动 根据 给定 知识 Bot 推送 完整 的 天气 信息 用户 满足 并 好评',
        #  有两种的个数
        '美食推荐':'Bot 主动 推荐 这种 天气 适合 吃 <info> 用户 接受 需要 聊 2 轮',
        '寒暄':'Bot 主动 根据 给定 的 <info> 寒暄 第一句 问候 要 带 用户 名字 聊天 内容 不要 与 <info> 矛盾 聊天 要 自然 不要 太 生硬',
        '播放音乐': 'Bot 主动 询问 是否 播放 用户 同意 后 Bot 播放 音乐 <info>',
        # 三种
        '问答音乐': '用户 主动 问 <info> 音乐 主唱 Bot 回答 <info> 用户 满足 并 好评',
        '问答电影': '用户 主动 问 电影 <info> 主演 是 谁 Bot 回答 <info> 用户 满足 并 好评',
        '问答其他': '用户 主动 按 <info> 问 <info> 的 信息 Bot 回答 用户 满意 并 好评',
        # 四种
        '新闻推荐接受': 'Bot 主动 推荐 <info> 的 新闻 <info> 用户 接受 需要 聊 2 轮',
        '新闻推荐拒绝': 'Bot 主动 聊 <info> 的 新闻 <info> 用户 回答 暂时 不想 聊 新闻 聊 1 轮 即可',
        # 五种
        '关于明星的聊天电影': 'Bot 主动 Bot 主动 从 <info> 聊到 他 的 主演 <info> 根据 给定 的 明星 信息 聊 <info> 相关 内容 至少 要 聊 2 轮 避免 话题 切换 太 僵硬 不够 自然',
        '关于明星的聊天其他': 'Bot 主动 根据 给定 的 明星 信息 聊 <info> 相关 内容 至少 要 聊 2 轮 避免 话题 切换 太 僵硬 不够 自然',
        '关于明星的聊天生日': 'Bot 主动 从 <info> 的 生日 开始 聊 根据 给定 的 明星 信息 聊 <info> 相关 内容 至少 要 聊 2 轮 避免 话题 切换 太 僵硬 不够 自然',
        '关于明星的聊天音乐': 'Bot 主动 Bot 主动 从 <info> 聊到 他 的 主唱 <info> 根据 给定 的 明星 信息 聊 <info> 相关 内容 至少 要 聊 2 轮 避免 话题 切换 太 僵硬 不够 自然',

        '提问电影名': 'Bot 主动 问 用户 最 喜欢 的 电影 名 用户 回答 最 喜欢 <info>',
        '提问主演': 'Bot 主动 问 用户 最 喜欢 <info> 的 哪个 主演 不 可以 问 用户 <info> 的 主演 是 谁 用户 回答 最 喜欢 <info>',
        '提问歌曲名': 'Bot 主动 问 用户 最 喜欢 的 歌曲名 用户 回答 最 喜欢 <info>',
        '提问新闻': 'Bot 主动 最 喜欢 谁 的 新闻 用户 回答 最 喜欢 <info> 的 新闻',


        # 七种
        '电影推荐接受':'Bot 主动 使用 <info> 的 某个 评论 当做 推荐 理由 来 推荐 <info> 用户 先问 电影 <info> 中 的 一个 或 多个 Bot 回答 最终 用户 接受',
        '电影推荐转电影':'Bot 主动 Bot 使用 <info> 的 某个 评论 当做 推荐 理由 来 推荐 <info> 用户 切换 话题 用户 现在 更 喜欢 <info> 的 电影',
        '电影推荐转音乐':'Bot 主动 Bot 使用 <info> 的 某个 评论 当做 推荐 理由 来 推荐 <info> 用户 切换 话题 用户 现在 喜欢 <info> 的 音乐',
        '电影推荐拒绝':'Bot主动Bot使用<info>的某个评论当做推荐理由来推荐<info>用户拒绝拒绝原因可以是<info>；Bot使用<info>的某个评论当做推荐理由来推荐<info>用户先问电影<info>中的一个或多个Bot回答最终用户接受注意不要在一句话推荐两个电影',
        '电影推荐不聊电影':'Bot 主动 Bot 使用 <info> 的 某个 评论 当做 推荐 理由 来 推荐 <info> 用户 回答 暂时 不想 聊 电影 聊 1 轮 即可',


        '音乐推荐接受': 'Bot 主动 Bot 使用 <info> 的 某个 评论 当做 推荐 理由 来 推荐 <info> 用户 接受 需要 聊 2 轮',
        '音乐推荐主动播放': 'Bot 主动 播放 完毕 后 Bot 推荐 同 主唱 的 其他 歌曲 先 使用 <info> 的 某个 评论 当做 推荐 理由 来 推荐 <info> 用户 拒绝 拒绝 原因 可以 是 <info> 聊 1 轮 即可 ； 再 使用 <info> 的 某个 评论 当做 推荐 理由 来 推荐 <info> 用户 接受 需要 聊 2 轮 注意 不要 在 一句 话 推荐 两首歌',
        '音乐推荐天气': 'Bot 主动 推荐 这种 天气 适合 听 <info> 用户 接受 需要 聊 2 轮',
        '音乐推荐拒绝': 'Bot 主动 使用 <info> 的 某个 评论 当 推荐 理由 来 推荐 <info> 用户 回答 暂时 不想 听 音乐 聊 1 轮 即可',
        '音乐推荐二次': 'Bot 主动 Bot 使用 <info> 的 某个 评论 当做 推荐 理由 来 推荐 <info> 用户 拒绝 拒绝 原因 可以 是 <info> 聊 1 轮 即可 ； Bot 使用 <info> 的 某个 评论 当做 推荐 理由 来 推荐 <info> 用户 接受 需要 聊 2 轮 注意 不要 在 一句 话 推荐 两首歌'
    }
    return goal2tmp


def preprocess_knowledge(knowledge):
    new_knowledge = []
    for kg in knowledge:
        if kg[1].find("评论") != -1:
            tmp_split = kg[2].split('。')
            for tmp_str_item in tmp_split:
                if tmp_str_item.strip() != '':
                    new_kg = [kg[0], kg[1], tmp_str_item + '。']
                    new_knowledge.append(new_kg)

        elif len(kg[1].split()) == 1 or kg[1] == '喜欢 的 新闻':
            if kg[1].startswith('2018'):
                kg[1] = '天气'
            new_knowledge.append(kg)
        else:
            new_kg = []
            p = kg[1].split()
            p_pre = p[:-1]
            p_next = p[-1]
            new_kg.append(kg[0] + ('').join(p_pre))
            new_kg.append(p_next)
            new_kg.append(kg[2])
            new_knowledge.append(new_kg)



    return new_knowledge



if __name__ == '__main__':
    train_file = os.path.join(".", "work", "resource", "train.txt")
    tgt_file = os.path.join(".", "work", "train.txt")

    log_file = os.path.join(".", "work", "log.txt")
    entities_file = os.path.join(".", "work", "entities_set")

    origin_data = read_origin_data(train_file)

    usr_profile_keys_set = set()
    entities_set = set()
    goal_type_list = list()

    f_log = open(log_file, 'w', encoding='utf-8')

    with open(tgt_file, 'w', encoding='utf8') as fout:
        for dialog in origin_data:
            situation = dialog['situation']
            goal = dialog['goal']
            usr_prof = dialog['user_profile']
            knowledge = dialog['knowledge']
            conversations = dialog['conversation']
            # process user profile
            usr_prof = preprocess_usr_profile(usr_prof)
            # process knowledge
            knowledge = preprocess_knowledge(knowledge)


            # compute all user profile keys
            if usr_prof:
                for item in usr_prof.keys():
                    usr_profile_keys_set.add(item)

            situation = process_situation(situation)

            # TODO:
            # 1. process situation -> Tuple[]
            # 2. process user profile
            # 3. judge the role of each conversation
            # 4. merge the user profile into several types
            # 5. write the conversation format
            # (conversation, goal_tag, current_goal, first_goal, final_goal,knowledge, situation, user_profile, role)
            #
            goals = goal.split("-->")
            # kg
            kg_entities_set = {item[0] for item in knowledge}
            for goal_item in goals:
                goal_type, goal_entities = process_goal(goal_item, situation)
                goal_type_list.append(goal_type)
                ##################################################################################################

                for item in goal_entities:
                    entities_set.add(item)
                    if re.findall('2018-8-26', item):
                        # print(goal_type, ":\t" ,item, "\t", goal_item)
                        split_str = "\n======================================================\n"
                        f_log.write(str_tuple(goal_type, item, goal_item,split_str))
                #     if len(item) > 30:
                #         split_str = "\n======================================================\n"
                #
                #         log_str = str_tuple(goal_item, goal_type, goal_entities, kg_entities_set, split_char='\n',
                #                             split_string=split_str)
                #         f_log.write(log_str)

                # if len(union_goal_entities) == 0 and goal_type.find("问") == -1 and goal_type.find("再见") == -1 \
                #         and goal_type.find("寒暄") == -1  and goal_type.find("天气") == -1:
                # if  goal_type.find("寒暄") != -1:
                #
                ##################################################################################################
                # if  goal_type.find("音乐推荐") != -1:
                #     split_str = "\n======================================================\n"
                #
                #     log_str = str_tuple(goal_item, goal_type,goal_entities,kg_entities_set,split_char='\n',
                #                         split_string=split_str)
                #     f_log.write(log_str)

            role_id = judge_role(goals[0])  # find the first role
            first_goal, final_goal = goals[0], goals[-2]
            first_goal, final_goal = process_goal(first_goal, situation), process_goal(final_goal, situation)
            processed_goals = [process_goal(item, situation) for item in goals]  # 所有已处理的goals

            conv_item = []
            history = []
            raw_cur_goals = []
            for dial_id, conversation in enumerate(conversations):
                # get goal
                response, goal_id = preprocess_response(conversation)
                # get current goal
                try:
                    raw_current_goal = goals[goal_id - 1] if goal_id != 0 else raw_current_goal
                    current_goal = sub_info(raw_current_goal, remain_space=True)[0]
                except IndexError:
                    if goal_id > len(goals):
                        raw_current_goal = raw_current_goal
                        current_goal = sub_info(raw_current_goal, remain_space=True)[0]


                goal_tag = 1 if goal_id != 0 else 0
                role = role_list[role_id]  # get conversation role
                role_id = (role_id + 1) % len(role_list)


                conv_item.append([role, response, goal_tag, current_goal.strip(), first_goal,
                                final_goal, str(processed_goals), knowledge, tuple(situation), dict(usr_prof), history.copy()])
                history.append((role, response))
                raw_cur_goals.append(raw_current_goal)

            # 写入数据
            # 中颗粒度的goal
            # find next goal
            for item_id, item in enumerate(conv_item):
                try:
                    next_goal = raw_cur_goals[item_id+1]
                    next_goal_type = [process_goal(next_goal, situation)[0]]
                except IndexError:
                    next_goal_type = []
                item.append(next_goal_type)
                tgt_str = str_tuple(*item)
                fout.write(tgt_str)

            fout.write('\n')
    f_log.close()

    with open(entities_file, 'w', encoding='utf-8') as f_entities:
        f_entities.write(str(entities_set))

    goals_file = os.path.join(".", "work", "goals_dict.json")
    with open(goals_file, 'w', encoding='utf-8') as f:
        tmp_str = json.dumps(goaltype2goal_tmp(), ensure_ascii=False)
        f.write(tmp_str)


