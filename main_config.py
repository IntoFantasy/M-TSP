import numpy as np


# 进行全局参数的设置
def _init():
    global GA_Information
    GA_Information = {}


def initiation(name, value):
    GA_Information[name] = value


def getvalue(name):
    return GA_Information[name]


def setvalue(city_id):
    GA_Information["city_first"] = city_id


_init()
initiation("city_number", 21)          # 城市个数
initiation("city_first", 5)            # 出发城市编号
initiation("individual_number", 200)   # 每轮的个体数
initiation("iteration_number", 300)    # 迭代次数
initiation("variation_rate", 0.65)     # 设置变异率
