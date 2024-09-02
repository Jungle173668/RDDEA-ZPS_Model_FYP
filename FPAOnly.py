import numpy as np
from scipy.special import gamma
from math import sin
from math import pi
from math import e
import time
import math
import random
import torch

import NonlinearFunctions
import DataGeneration

def Levy(d, beta):  # Levy飞行
    sigma = pow((gamma(1 + beta) * sin(pi * beta / 2) / (gamma((1 + beta) / 2) * beta * pow(2, (beta - 1) / 2))),
                (1 / beta))
    u = np.random.randn(1, d) * sigma
    v = np.random.randn(1, d)
    step = [(u[0][i] / pow(abs(v[0][i]), (1 / beta))) for i in range(d)]
    L = step
    return L


def simple_bound(s, Lb, Ub):
    # print(s)
    ns_tmp = s
    # print(s)
    # print(Lb)
    # print(Ub)
    for i in range(len(s)):
        if s[i] > Ub[i]:
            ns_tmp[i] = Ub[i]
        if s[i] < Lb[i]:
            ns_tmp[i] = Lb[i]
    s = ns_tmp
    return s


# 设置参数
D = 10  # 基础参数
N_data = 20 * D  # 拉丁采样的数据量
N_model = 20  # RBFNN的数量
N_train = 300  # 模型训练次数（优化算法优化次数）
N_centers = 8 * D  # 聚类中心的数量
B = int(0.8 * N_data)  # 训练集的数量

dbc_criterion = 1e-3  # dbc变化初始阈值
w = int(0.1 * N_centers)  # 需要进行数据生成的比例
alpha = 0.20  # 零点采样的数据比例

batch_size = 32  # 一次传入模型的数据数量

is_origin = 1  # 代表是否调转x和y. 0为改进后模型，1为原模型，2为改进后模型+FPA

# FPA基础参数设置，分别代表种群数量，转移概率，迭代次数，参数下、上边界，Lvey飞行参数，logistics混沌映射系数, 高斯生成的sigma
para = [100, 0.7, 300, -2, 2, 1.5, 3.99, 1e-3] 

# 实例化函数
# func_list = [NonlinearFunctions.Func_N1()]#,NonlinearFunctions.Func_N2()]
func_list = [NonlinearFunctions.Func_N1(),NonlinearFunctions.Func_N2(),
             NonlinearFunctions.Func_N3(),NonlinearFunctions.Func_N4(),
             NonlinearFunctions.Func_N5(),NonlinearFunctions.Func_N6()]

# 需要寻优的参数
# N_centers_choices = [i for i in range(80,101)]

# 记录寻优结果
Euler_distances = []

func_result = []
func_var = []

for func in func_list:
    results = []
    for exp_times in range(20):
        # ========== 生成数据集（DBC, ZCSampling）和参数 ==========
        
        x_train, y_train, x_predict, y_predict, whole_x, whole_y, \
            centers, transformed_sigma, mean_CP = DataGeneration.Generate_fulldataset(func, N_data, B, N_centers, dbc_criterion, w, alpha, is_origin)        
        
        # 参数
        n = len(x_train)
        p = 0.7
        N_iter = 200
        d = func.d
        Lb = func.Lb
        Ub = func.Ub
        Levy_param = 1.5

        goal_func = func.func_N

        # 储存结果
        plot_fmin = []
        S = []
        Sol = []
        Fitness = []

        for i in x_train:
            Sol.append(i)
            Fitness.append(np.linalg.norm(torch.tensor(goal_func(i))-torch.tensor(func.root)))
        
        for i in range(n):
            S.append(Sol[i])
        
        fmin = min(Fitness)
        I = Fitness.index(fmin)
        best = Sol[I]

        # 传粉
        for t in range(N_iter):
            for i in range(n):
                L = Levy(d, Levy_param)
                delta = Sol[i]-best
                dS = torch.tensor(L) * delta
                S[i] = Sol[i] + dS
                S[i] = simple_bound(S[i], Lb, Ub)

            else:
                epsilon = random.random()
                JK = list(set(random.sample(list(range(n)), n)))
                S[i] = (1 - epsilon) * (Sol[JK[1]]-Sol[i]) + epsilon * (best-Sol[JK[1]]) + Sol[i]
                S[i] = simple_bound(S[i], Lb, Ub)

            Fnew = np.linalg.norm(torch.tensor(goal_func(S[i]))-torch.tensor(func.root))
            
            if Fnew < Fitness[i]:
                Sol[i] = S[i]
                Fitness[i] = Fnew
            if Fnew < fmin:
                best = S[i]
                fmin = Fnew
            
        # 自然变异
        worst_n = int(0.2*(t/N_iter)* n + 0.05*n)
        sort_fit = np.argsort(Fitness)
        for j in range(worst_n):
            i = sort_fit[j + n - worst_n]
            epsilon = random.random()
            JK = list(set(random.sample(list(range(n)), n)))
            # 对si个体位置更新
            S[i] = (epsilon) * Sol[i] * np.tan((random.random() - 0.5) * np.pi) \
                    + epsilon * (Sol[JK[1]] - Sol[JK[2]] + Sol[JK[3]] - Sol[JK[4]]) + (1-epsilon)*best
            S[i] = simple_bound(S[i], Lb, Ub)
            
            Fnew = np.linalg.norm(torch.tensor(goal_func(S[i]))-torch.tensor(func.root))
            
            # 如果新位置更优，则替换掉
            if Fnew < Fitness[i]:
                Sol[i] = S[i]
                Fitness[i] = Fnew
            if Fnew < fmin:
                best = Sol[i]
                fmin = Fnew
        
        plot_fmin.append(fmin)

        print(best, fmin)
        results.append(fmin)
    
    func_result.append(np.mean(results))
    func_var.append(np.std(results))

print(func_result)
print(func_var)
