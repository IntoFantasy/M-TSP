import numpy as np
import random
import main_config as mc

city_distance = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0,	15.32,	42.32,	39.92,	20.45,	24.74,	28.6,	35.16,	27.47,	32.65,	27.93,	61.18,	55.7,	61.31,	64.39,	33.16,	56.88,	85.8,	34.31,	51.16,	46.21],
    [0, 22.82,	0,	36.53,	34.13,	19.83,	18.95,	49.62,	57.98,	50.3,	53.67,	36.48,	69.73,	64.26,	76.41,	82.01,	55.98,	79.7,	108.63,	57.14,	73.99,	69.04],
    [0,  43.78,	30.49,	0,	15.61,	33.1,	16.93,	62.9,	67.36,	59.67,	66.94,	21.64,	54.88,	49.41,	64.08,	67.17,	65.36,	89.08,	118,	66.51,	83.36,	78.41],
    [0, 41.45,	28.16,	15.68,	0,	30.77,	14.61,	60.57,	65.03,	57.34,	64.62,	19.31,	52.56,	47.09,	61.75,	64.84,	63.03,	86.75,	115.68,	64.19,	81.03,	76.09],
    [0, 25.55,	17.43,	36.74,	34.35,	0,	19.17,	29.8,	40.46,	41.44,	33.84,	36.7,	69.94,	64.47,	75.27,	78.36,	47.13,	70.84,	99.77,	48.28,	61.88,	49.82],
    [0, 26.84,	13.55,	17.58,	15.18,	16.17,	0,	45.96,	56.63,	54.32,	50.01,	17.53,	50.78,	45.31,	57.46,	63.06,	60,	83.72,	112.65,	61.16,	78.01,	65.99],
    [0, 22.1,	35.61,	54.93,	52.53,	18.19,	37.35,	0,	33.76,	37.98,	27.14,	38.44,	71.69,	66.21,	71.82,	74.91,	43.67,	67.39,	96.32,	44.83,	55.17,	20.03],
    [0, 36.57,	51.89,	67.31,	64.91,	36.77,	55.94,	41.68,	0,	20.03,	30.63,	44.21,	77.46,	71.97,	77.59,	80.67,	34.99,	58.7,	87.63,	26.87,	31.65,	26.71],
    [0, 29.12,	44.44,	59.86,	57.47,	37.99,	53.86,	46.14,	20.27,	0,	20.78,	36.76,	70.01,	64.53,	70.14,	73.23,	27.03,	50.75,	79.68,	19.43,	36.27,	31.33],
    [0, 26.58,	40.1,	59.42,	57.02,	22.67,	41.84,	27.58,	23.15,	13.06,	0,	42.93,	76.18,	70.69,	76.31,	79.39,	40.1,	63.82,	92.74,	32.49,	44.57,	39.62],
    [0, 35.23,	36.28,	27.48,	25.08,	38.89,	22.73,	52.25,	50.1,	42.41,	56.3,	0,	37.62,	32.15,	44.31,	49.91,	48.1,	71.82,	100.74,	49.25,	66.1,	61.15],
    [0, 62.01,	63.06,	54.26,	51.86,	65.67,	49.51,	79.03,	76.88,	69.19,	83.08,	31.16,	0,	11.41,	50.45,	76.69,	74.88,	98.6,	127.52,	76.03,	92.88,	87.93],
    [0, 60.8,	61.86,	53.06,	50.66,	64.47,	48.31,	77.82,	75.66,	67.98,	81.86,	29.95,	15.68,	0,	44.36,	75.48,	73.67,	97.39,	126.32,	74.83,	91.67,	86.73],
    [0, 43.46,	51.06,	44.77,	42.37,	52.32,	37.51,	60.48,	58.32,	50.64,	64.52,	19.16,	31.76,	21.41,	0,	43.49,	56.33,	80.04,	108.97,	57.48,	74.33,	69.38],
    [0, 46.3,	56.41,	47.61,	45.21,	55.16,	42.86,	63.32,	61.16,	53.48,	67.36,	24.51,	57.76,	52.28,	43.25,	0,	59.17,	82.88,	111.81,	60.32,	77.17,	72.22],
    [0, 30.46,	45.78,	61.2,	58.81,	39.33,	55.2,	47.48,	30.88,	22.68,	43.47,	38.1,	71.35,	65.87,	71.48,	74.57,	0,	23.72,	52.64,	30.03,	46.88,	41.93],
    [0, 54.01,	69.33,	84.75,	82.35,	62.87,	78.75,	71.03,	54.42,	46.23,	67.01,	61.65,	94.9,	89.42,	95.03,	98.11,	23.55,	0,	28.92,	53.58,	70.43,	65.48],
    [0, 68.91,	84.23,	99.65,	97.25,	77.77,	93.65,	85.93,	69.32,	61.13,	81.91,	76.55,	109.79,	104.32,	109.93,	113.01,	38.44,	14.9,	0,	68.48,	85.33,	80.38],
    [0, 28.62,	43.94,	59.36,	56.96,	37.48,	53.36,	45.64,	19.76,	12.08,	32.86,	36.26,	69.5,	64.03,	69.63,	72.72,	27.03,	50.75,	79.68,	0,	35.77,	30.82],
    [0, 35.64,	50.96,	66.38,	63.99,	41.26,	60.38,	46.16,	14.72,	19.1,	35.12,	43.28,	76.53,	71.05,	76.66,	79.75,	34.06,	57.78,	86.7,	25.95,	0,	25.78],
    [0, 49.99,	65.31,	80.73,	78.33,	48.5,	67.66,	30.31,	29.07,	33.45,	49.46,	57.63,	90.88,	85.4,	91.01,	94.09,	48.41,	72.12,	101.05,	40.29,	45.07,	0]
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
        city_first = mc.getvalue("city_first")
        city_first_pos = self.gene.index(city_first)
        time1 = time2 = 0
        if city_first_pos != 0:
            time1 = city_distance[city_first][self.gene[0]]
            for i in range(city_first_pos):
                time1 += city_distance[self.gene[i]][self.gene[i+1]]
        if city_first_pos != self.gene_len-1:
            for j in range(city_first_pos, self.gene_len-1):
                time2 += city_distance[self.gene[j]][self.gene[j+1]]
            time2 += city_distance[self.gene[-1]][city_first]
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
