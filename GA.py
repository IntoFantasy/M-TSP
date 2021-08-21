import numpy as np
import random
import main_config as mc

city_distance = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0,	20.28,	40.91,	37.42,	24.37,	26.31,	19.19,	40.9,	29.73,	25.92,	38.89,	79.06,	77.04,	51.23,	55.49,	31.74,	49.06,	53.15,	28.97,	39.51,	40.94],
    [0,  20.28,	0,	32.23,	28.74,	23.44,	17.63,	39.47,	61.18,	50.01,	46.2,	40.92,	81.09,	79.29,	63.09,	71.12,	52.02,	69.34,	73.43,	49.25,	59.79,	61.22],
    [0,  40.91,	32.23,	0,	11.75,	32.55,	14.6,	53.53,	81.41,	68.38,	60.26,	29.45,	69.62,	67.82,	55.18,	59.65,	72.65,	89.97,	94.06,	69.88,	80.42,	75.28],
    [0, 37.42,	28.74,	11.75,	0,	29.06,	11.11,	50.04,	77.92,	64.89,	56.77,	25.96,	66.13,	64.33,	51.69,	56.16,	69.16,	86.48,	90.57,	66.39,	76.93,	71.79],
    [0, 24.37,	23.44,	32.55,	29.06,	0,	17.95,	20.98,	48.86,	35.83,	27.71,	41.24,	81.41,	79.61,	63.41,	71.44,	52.69,	70.01,	74.1,	42.47,	53.01,	42.73],
    [0, 26.31,	17.63,	14.6,	11.11,	17.95,	0,	38.93,	66.81,	53.78,	45.66,	23.29,	63.46,	61.66,	45.46,	53.49,	58.05,	75.37,	79.46,	55.28,	65.82,	60.68],
    [0, 19.19,	39.47,	53.53,	50.04,	20.98,	38.93, 	0,	 38.8,	25.77,	17.65,	54.66,	94.83,	92.81,	   67,	71.26,	47.51,	64.83,	68.92,	32.41,	42.95,	21.75],
    [0,  40.9,  61.18,	81.41,	77.92,	48.86,	66.81,	38.8,	0,	18.57,	22.89,	63.31,	103.48,	101.46,	 75.65,	79.91,	34.48,	51.8,	55.89,	17.81,	10.25,	31.77],
    [0, 29.73,	50.01,	68.38,	64.89,	35.83,	53.78,	25.77,	18.57,	0,	8.12,	52.14,	92.31,	90.29,	64.48,	68.74,	22.55,	39.87,	43.96,	6.64,	17.18,	38.7],
    [0, 25.92,	46.2,	60.26,	56.77,	27.71,	45.66,	17.65,	22.89,	8.12,	0,	60.26,	100.43,	98.41,	72.6,	76.86,	30.67,	47.99,	52.08,	14.76,	25.3,	39.4],
    [0, 38.89,	40.92,	29.45,	25.96,	41.24,	23.29,	54.66,	63.31,	52.14,	60.26,	0,	43.73,	41.92,	25.73,	33.76,	54.15,	71.47,	75.56,	51.38,	61.92,	76.41],
    [0, 79.06,	81.09,	69.62,	66.13,	81.41,	63.46,	94.83,	103.48,	92.31,	100.43,	43.73,	0,	10.82,	34.94,	59.08,	94.32,	111.64,	115.73,	91.55,	102.09,	116.58],
    [0, 77.04,	79.29,	67.82,	64.33,	79.61,	61.66,	92.81,	101.46,	90.29,	98.41,	41.92,	10.82,	0,	25.81,	49.95,	92.3,	109.62,	113.71,	89.53,	100.07,	114.56],
    [0, 51.23,	63.09,	55.18,	51.69,	63.41,	45.46,	   67,	 75.65,	64.48,	72.6,	25.73,	34.94,	25.81,	0,	24.14,	66.49,	83.81,	87.9,	63.72,	74.26,	88.75],
    [0, 55.49,	71.12,	59.65,	56.16,	71.44,	53.49,	71.26,	79.91,	68.74,	76.86,	33.76,	59.08,	49.95,	24.14,	0,	70.75,	88.07,	92.16,	67.98,	78.52,	93.01],
    [0, 31.74,	52.02,	72.65,	69.16,	52.69,	58.05,	47.51,	34.48,	22.55,	30.67,	54.15,	94.32,	92.3,	66.49,	70.75,	0,	17.32,	21.41,	22.55,	33.09,	54.61],
    [0, 49.06,	69.34,	89.97,	86.48,	70.01,	75.37,	64.83,	51.8,	39.87,	47.99,	71.47,	111.64,	109.62,	83.81,	88.07,	17.32,	0,	4.09,	39.87,	50.41,	71.93],
    [0, 53.15,	73.43,	94.06,	90.57,	74.1,	79.46,	68.92,	55.89,	43.96,	52.08,	75.56,	115.73,	113.71,	87.9,	92.16,	21.41,	4.09,	0,	43.96,	54.5,	76.02],
    [0, 28.97,	49.25,	69.88,	66.39,	42.47,	55.28,	32.41,	17.81,	6.64,	14.76,	51.38,	91.55,	89.53,	63.72,	67.98,	22.55,	39.87,	43.96,	0,	16.42,	37.94],
    [0, 39.51,	59.79,	80.42,	76.93,	53.01,	65.82,	42.95,	10.25,	17.18,	25.3,	61.92,	102.09,	100.07,	74.26,	78.52,	33.09,	50.41,	54.5,	16.42,	0,	30.38],
    [0, 40.94,	61.22,	75.28,	71.79,	42.73,	60.68,	21.75,	31.77,	38.7,	39.4,	76.41,	116.58,	114.56,	88.75,	93.01,	54.61,	71.93,	76.02,	37.94,	30.38,	0]
                          ])


def rt(a):
    b = sorted(a)
    for i in range(21):
        if b[i] != i+1:
            return False
    return True


# 定义个体类
class Individual:
    # 个体初始化
    def __init__(self, gene=None):
        self.gene_len = mc.getvalue("city_number")
        if gene is None:
            gene = [i for i in range(1, self.gene_len+1)]
            random.shuffle(gene)
        self.gene = gene
        self.fitness, self.time = self.fitness_evaluate()

    # 对个体的生存能力进行评估（此处为最多的用时）
    def fitness_evaluate(self):
        city_first_pos = self.gene.index(5)
        time1 = time2 = 0
        if city_first_pos != 0:
            time1 = city_distance[5][self.gene[0]]
            for i in range(city_first_pos):
                time1 += city_distance[self.gene[i]][self.gene[i+1]]
        if city_first_pos != self.gene_len-1:
            for j in range(city_first_pos, self.gene_len-1):
                time2 += city_distance[self.gene[j]][self.gene[j+1]]
            time2 += city_distance[5][self.gene[-1]]
        time = max(time1, time2)
        fitness = 10/time                                  # fitness函数: time越大, fitness越小
        return fitness, time


# 训练模型
class GAModel:
    def __init__(self):
        self.individual_number = mc.getvalue("individual_number")     # 基因库的个体数
        self.gene_len = mc.getvalue("city_number")         # 单个基因长度
        self.best = None                                   # 最佳个体
        self.gene_pool = []                                # 基因库
        self.answer_list = []                              # 每一代的最优解
        self.time_list = []                                # 每一代最优解所花时间

    # 杂交
    def cross(self):
        new_gene = []
        random.shuffle(self.gene_pool)
        for i in range(0, self.individual_number-2, 2):
            # 父亲基因
            gene_father = self.gene_pool[i]
            # 母亲基因
            gene_mother = self.gene_pool[i+1]
            # 进行浅拷贝
            gene_son1 = gene_father.gene[:]
            gene_son2 = gene_mother.gene[:]
            # 生成杂交片段目录
            index1 = random.randint(0, self.gene_len-2)
            index2 = random.randint(index1, self.gene_len-1)
            pos_recorder1 = {value: index for index, value in enumerate(gene_son1)}
            pos_recorder2 = {value: index for index, value in enumerate(gene_son2)}
            for j in range(index1, index2):
                value1, value2 = gene_son1[j], gene_son2[j]
                pos1, pos2 = pos_recorder1[value2], pos_recorder2[value1]
                gene_son1[j], gene_son1[pos1] = gene_son1[pos1], gene_son1[j]
                gene_son2[j], gene_son2[pos2] = gene_son2[pos2], gene_son2[j]
                pos_recorder1[value1], pos_recorder1[value2] = pos1, j
                pos_recorder2[value2], pos_recorder2[value1] = pos2, j
            new_gene.append(Individual(gene_son1))
            new_gene.append(Individual(gene_son2))
        return new_gene

    # 突变
    def variation(self, variation_rate, new_gene):
        for individual in new_gene:
            if random.random() < variation_rate:
                # 反转变异
                old_gene = individual.gene[:]
                # 生成变异区间
                index1 = random.randint(0, self.gene_len-2)
                index2 = random.randint(index1, self.gene_len-1)
                mutate = old_gene[index2:index1:-1]
                individual.gene = old_gene[:index1+1] + mutate + old_gene[index2+1:]
        # 新旧基因进行合并
        self.gene_pool += new_gene

    # 选择
    def select(self):
        # 锦标赛算法(5进2)
        game_time = self.individual_number/2
        winners = []
        for i in range(int(game_time)):
            group = []
            # 组成小组
            for j in range(5):
                player = random.choice(self.gene_pool)
                player = Individual(player.gene)
                group.append(player)
            group_sorted = sorted(group, key=lambda individual: individual.fitness, reverse=True)
            winners += group_sorted[:2]
        self.gene_pool = winners

    # 下一代基因生成
    def gene_next(self):
        # 杂交
        new_gene = self.cross()
        # 突变
        self.variation(mc.getvalue("variation_rate"), new_gene)
        # 选择
        self.select()
        # 遍历找出最优的基因
        for i in self.gene_pool:
            if i.fitness > self.best.fitness:
                self.best = i

    # 训练
    def train(self):
        # 生成初代种群
        self.gene_pool = [Individual() for _ in range(mc.getvalue("individual_number"))]
        self.best = self.gene_pool[0]
        for i in self.gene_pool:
            if i.fitness > self.best.fitness:
                self.best = i
        # 开始迭代
        for i in range(mc.getvalue("iteration_number")):
            # 生成下一代基因
            self.gene_next()
            self.answer_list.append(self.best)
            self.time_list.append(10/self.best.fitness)
        return self.answer_list, self.time_list
