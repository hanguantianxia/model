# author:hjt
# contact: hanguantianxia@sina.com
# datetime:2020/4/13 14:10
# software: PyCharm

"""

"""


Segment_tpye = [
"S", "P", "O",                                      # knowledge seg
"date", "workday", "time", "location", "theme",     # situation
"goal_type", "goal_entity",                         # goal_seq
"current_goal",                                     # current goal
"user", "bot",                                      # conversation role
 'active poi',                                      # usr profile
 'active news',
 'active star',
 'active movie',
 'active food',
 'active music',
 'negative movie',
 'negative music',
 'name',
 'addr',
 'age',
 'sex',
 'reject',
 'profession'
]

if __name__ == '__main__':
    Segment_dict = {"[PAD]":0,"[UNK]":1}
    base = len(Segment_dict)
    for i in range(len(Segment_tpye)):
        Segment_dict[str(i)] = i + base
    base = len(Segment_dict)
    for idx, item in enumerate(Segment_tpye):
        Segment_dict[item] = idx + base
