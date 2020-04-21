# author:hjt
# contact: hanguantianxia@sina.com
# datetime:2020/4/13 14:10
# software: PyCharm

"""

"""
from build_dicts import file_saver

Segment_tpye = [
"S", "P", "O",                                      # knowledge seg
"date", "workday", "time", "location", "theme",     # situation                  # goal_seq
"current_goal",                                     # current goal
"user", "bot",                                      # conversation role
 # 'active poi',                                      # usr profile
 # 'active news',
 # 'active star',
 # 'active movie',
 # 'active food',
 # 'active music',
 # 'negative movie',
 # 'negative music',
 # 'name',
 # 'addr',
 # 'age',
 # 'sex',
 # 'reject',
 # 'profession',
'active 兴趣点',
 'active 新闻',
 'active 明星',
 'active 电影',
 'active 美食',
 'active 音乐',
 'negative 电影',
 'negative 音乐',
 '姓名',
 '居住地',
 '年龄区间',
 '性别',
 '拒绝',
 '职业状态'
]

if __name__ == '__main__':
    Segment_dict = {"[PAD]":0,"[UNK]":1}

    for idx, item in enumerate(Segment_tpye):
        Segment_dict[item] = len(Segment_dict)

    for i in range(10):
        Segment_dict["goal_type"+str(i)] = len(Segment_dict)
        Segment_dict["goal_entity"+str(i)] = len(Segment_dict)
    file_saver("./work/Segment_dict.json", Segment_dict)