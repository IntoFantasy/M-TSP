import numpy as np
import GA
import main_config as mc

ga = GA.GAModel()
answer_list, time_list = ga.train()
print("答案路径是:")
print(answer_list[-1].gene)
print("时间是:")
print(time_list[-1])
for ind in time_list:
    print(ind)
