import numpy as np
import GA
import main_config as mc


def run(city_id):
    mc.setvalue(city_id)
    answers = []
    times = []
    for k in range(100):
        ga = GA.GAModel()
        answer_list, time_list = ga.train()
        answers.append(answer_list[-1].gene)
        times.append(time_list[-1])
    time_least = times[0]
    pos = 0
    for j in range(100):
        if times[j] < time_least:
            time_least = times[j]
            pos = j
    print(city_id)
    print("答案路径是:")
    print(answers[pos])
    print("时间是：")
    print(time_least)
    print(times)


for i in range(1, 21):
    run(i)
