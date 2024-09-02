import numpy as np
from scipy.special import gamma
from math import sin
from math import pi
from math import e
import time
import math
import random
import torch

def map_func_logistic(x, alpha):#logistic混沌映射函数
    return (1 - x) * x * alpha

def generate_nearby_vectors(base_vector, scale):
    return np.random.normal(loc=base_vector,scale=scale)

def Levy(d, beta):  # Levy飞行
    sigma = pow((gamma(1 + beta) * sin(pi * beta / 2) / (gamma((1 + beta) / 2) * beta * pow(2, (beta - 1) / 2))),
                (1 / beta))
    u = np.random.randn(1, d) * sigma
    v = np.random.randn(1, d)
    step = [(u[0][i] / pow(abs(v[0][i]), (1 / beta))) for i in range(d)]
    L = step
    return L

def simple_bound(s, Lb, Ub):
    ns_tmp = s
    # print(s)
    # print(Lb)
    # print(Ub)
    for i in range(len(s[0])):
        if s[0][i] > Ub[i]:
            ns_tmp[0][i] = Ub[i]
        if s[0][i] < Lb[i]:
            ns_tmp[0][i] = Lb[i]
    s = ns_tmp
    return s




def fpa_origin(para, goal_func):
    n = para[0]  # 种群数量
    p = para[1]  # 转移概率
    N_iter = para[2]  # 迭代次数
    d = para[3]  # 搜寻空间的维数
    Lb = para[4]  # 下边界
    Ub = para[5]  # 上边界
    Levy_param = para[6]

    plot_fmin = []
    S = []
    Sol = []
    Fitness = []

    # 随机生成搜索空间
    l = [np.random.rand(1,d)]
    
    for i in range(1, n):
        l.append(map_func_logistic(l[i - 1], 3.99))
    for i in range(n):
        Sol.append((Ub - Lb) * l[i] + Lb)
        Fitness.append(goal_func(Sol[i]))
    for i in range(n):
        S.append(Sol[i])

    fmin = min(Fitness)
    I = Fitness.index(fmin)
    best = Sol[I]

    for t in range(N_iter):
        for i in range(n):
            L = Levy(d, Levy_param)
            dS = L * (Sol[i] - best)
            S[i] = Sol[i] + dS
            S[i] = simple_bound(S[i], Lb, Ub)
        else:
            epsilon = random.random()
            JK = list(set(random.sample(list(range(n)), n)))
            S[i] = (1 - epsilon) * (Sol[JK[1]]-Sol[i]) + epsilon * (best-Sol[JK[1]]) + Sol[i]
            S[i] = simple_bound(S[i], Lb, Ub)
        
        Fnew = goal_func(S[i])
        if Fnew < Fitness[i]:
            Sol[i] = S[i]
            Fitness[i] = Fnew
        if Fnew < fmin:
            best = S[i]
            fmin = Fnew
            

    # 自然变异
    worst_n = int(0.2*(t/N_iter)* n + 0.05*n)
    # 将种群按照适应度从大到小排序
    sort_fit = np.argsort(Fitness)
    for j in range(worst_n):
        i = sort_fit[j + n - worst_n]
        epsilon = random.random()
        JK = list(set(random.sample(list(range(n)), n)))
        # 对si个体位置更新
        S[i] = (epsilon) * Sol[i] * np.tan((random.random() - 0.5) * np.pi) \
                + epsilon * (Sol[JK[1]] - Sol[JK[2]] + Sol[JK[3]] - Sol[JK[4]]) + (1-epsilon)*best
        S[i] = simple_bound(S[i], Lb, Ub)
        Fnew = goal_func(S[i])
        # 如果新位置更优，则替换掉
        if Fnew < Fitness[i]:
            Sol[i] = S[i]
            Fitness[i] = Fnew
        if Fnew < fmin:
            best = Sol[i]
            fmin = Fnew
    
    plot_fmin.append(fmin)
    if np.var(plot_fmin[-10:]) < 1e-6 and t > 20:
        
        return best, fmin
    
    return best, fmin