import numpy as np
from scipy.special import gamma
from math import sin
from math import pi
import time
import random
import torch
from collections import OrderedDict

'''
=========================
     合并成[1,d]的向量
=========================
'''
def vector_concat(params_dict):
    flattened_params = []
    sizes = {}  # 记录size

    for param_name, param_value in params_dict.items():
        flattened_params.append(torch.flatten(param_value))
        sizes[param_name] = param_value.shape
    
    final_vector = torch.cat(flattened_params)
    # print(sizes)
    # print(final_vector)
    # print(final_vector.shape)
    
    return final_vector, sizes

'''
=========================
         还原向量
=========================
'''
def vector_split(sizes, final_vector):
    start = 0
    parameters = {}
    for param_name, param_size in sizes.items():
        end = start + param_size.numel()
        parameters[param_name] = final_vector[start: end].view(param_size)
        start = end
    
    # for param_name, param_value in parameters.items():
    #     print(param_name)
    #     print(param_value)
    
    parameters = OrderedDict(parameters)

    return parameters

'''
=========================
    logistic混沌映射函数
=========================
用于生成数据
'''
def map_func_logistic(x, alpha):
    return (1 - x) * x * alpha

'''
=========================
    高斯分布生成随机向量
=========================
'''
def generate_nearby_vectors(base_vector, scale):
    return np.random.normal(loc=base_vector,scale=scale)
    

'''
=========================
        Levy飞行
=========================
生成遵循Levy分布的参数L
'''
def Levy(d, beta):
    sigma = pow((gamma(1 + beta) * sin(pi * beta / 2) / (gamma((1 + beta) / 2) * beta * pow(2, (beta - 1) / 2))),
                (1 / beta))
    u = np.random.randn(1, d) * sigma
    v = np.random.randn(1, d)
    step = [(u[0][i] / pow(abs(v[0][i]), (1 / beta))) for i in range(d)]
    L = step

    return L

'''
=========================
        
=========================
生成遵循Levy分布的参数L
'''
def simple_bound(s, Lb, Ub):
    ns_tmp = s
    for i in range(len(s[0])):
        if s[0][i] > Ub:
            ns_tmp[0][i] = Ub
        if s[0][i] < Lb:
            ns_tmp[0][i] = Lb
    s = ns_tmp
    return s

'''
=========================
     FPA优化RBFNN参数
=========================
INPUT：
para：若干与FPA迭代相关的参数
goal_func：适应值计算方法，即MSE Loss
rbf：需优化的模型
（使用rbf.params_dict得到需要优化的参数: RBFNN的参数字典，包括
centers torch.Size([50, 2])
beta torch.Size([50])
linear.weight torch.Size([2, 50])
linear.bias torch.Size([2])
）

OUTPUT：


'''

def fpa(para, goal_func, rbf, x_train, y_train):
    n = para[0]  # 产生种群数量
    p = para[1]  # 转移概率
    N_iter = para[2]  # 迭代次数
    Lb = para[3]  # 参数下边界
    Ub = para[4]  # 参数上边界
    Levy_param = para[5]  # Levy飞行参数
    alpha = para[6]  # logistics混沌映射生成数据
    scale = para[7]  # 高斯函数生成数据的sigma

    params_dict = rbf.state_dict()  # 需优化的参数字典
    # for key, value in params_dict.items():
    #     print(key, value.shape)

    # 合并向量，并记录size用于后续还原
    final_vector, sizes = vector_concat(params_dict)

    d = len(final_vector)  # 总参数个数

    plot_fmin = []
    S = []  # 储存解
    Sol = []  # 中间数组
    Fitness = []  # 不同解的适应值

    # =========== FPA搜索 ============
    # l = [np.random.rand(1,d)]  # 随机产生一个向量
    
    l = [final_vector.unsqueeze(0).detach().numpy()]
    
    # # 修正范围
    # if Lb > np.min(l[0], axis=1):
    #     Lb = np.min(l[0], axis=1)
    #     print(Lb)

    # if Ub < np.max(l[0], axis=1):
    #     Ub = np.max(l[0], axis=1)
    #     print(Ub)

    # 依次生成随机参数空间
    for i in range(1,n):
        # l.append(map_func_logistic(l[i-1],alpha))  # logistics混沌映射生成数据
        l.append(generate_nearby_vectors(l[i-1], scale))

    # 计算原始参数解、Fitness值
    for i in range(n):
        Sol.append((Ub-Lb)* l[i] + Lb)

        transformed_vector = torch.from_numpy(Sol[i]).squeeze()  # 转换参数格式
        rbf.load_state_dict(vector_split(sizes,transformed_vector))  # 更新网络参数
        Fitness.append(goal_func(x_train,rbf(y_train)))  # 计算Fitness值

        S.append(Sol[i])  # 记录参数初始解

    fmin = min(Fitness)  # 最小Fitness
    I = Fitness.index(fmin)  # 最小Fitness下标
    best = Sol[I]  # 最佳参数向量

    # 开始FPA搜索
    for t in range(N_iter):

        # 传粉
        for i in range(n):
            if random.random() < p:
                # 全局搜索
                L = Levy(d, Levy_param)
                dS = L * (Sol[i] - best)
                S[i] = Sol[i] + dS
                S[i] = simple_bound(S[i], Lb, Ub)  # 限制S[i]的数值范围
            else:
                #局部搜索
                epsilon = random.random()
                JK = list(set(random.sample(list(range(n)), n)))  # 随机选择向量
                S[i] = (1 - epsilon) * (Sol[JK[1]]-Sol[i]) + epsilon * (best-Sol[JK[1]]) + Sol[i]
                S[i] = simple_bound(S[i], Lb, Ub)
            
            # 计算新的Fitness值
            rbf.load_state_dict(vector_split(sizes,torch.from_numpy(Sol[i]).squeeze()))  # 更新网络参数
            Fnew = goal_func(x_train,rbf(y_train))  # 计算Fitness值
            
            if Fnew < Fitness[i]:  # 若更好则替换原解
                Sol[i] = S[i]
                Fitness[i] = Fnew
            if Fnew < fmin:  # 记录最佳解
                best = S[i]
                fmin = Fnew

        # 自然变异（一次完整搜索后）
        worst_n = int(0.2*(t/N_iter)* n + 0.05*n)

        # 将种群按照适应度从大到小排序
        # 将 PyTorch 张量转换为 NumPy 数组
        numpy_array = np.array([tensor.item() for tensor in Fitness])
        sort_fit = np.argsort(numpy_array)
        
        for j in range(worst_n):  # 对最差的几个适应度样本进行搜索更新
            i = sort_fit[j + n - worst_n]
            epsilon = random.random()
            JK = list(set(random.sample(list(range(n)), n)))
            
            # 对si个体位置更新
            S[i] = (epsilon) * Sol[i] * np.tan((random.random() - 0.5) * np.pi) \
                   + epsilon * (Sol[JK[1]] - Sol[JK[2]] + Sol[JK[3]] - Sol[JK[4]]) + (1-epsilon)*best
            S[i] = simple_bound(S[i],Lb,Ub)  # 纠正范围
            
            rbf.load_state_dict(vector_split(sizes,torch.from_numpy(S[i]).squeeze()))  # 更新网络参数
            Fnew = goal_func(x_train,rbf(y_train))  # 计算Fitness值

            # 如果新位置更优，则替换掉
            if Fnew < Fitness[i]:
                Sol[i] = S[i]
                Fitness[i] = Fnew
            if Fnew < fmin:
                best = Sol[i]
                fmin = Fnew
        
        plot_fmin.append(float(fmin))
        # 判断停止条件
        if np.var(plot_fmin[-10:]) < 1e-6 and t > 20:
            rbf.load_state_dict(vector_split(sizes,torch.from_numpy(best).squeeze()))
            
            return rbf, goal_func(x_train,rbf(y_train))

    return rbf, goal_func(x_train,rbf(y_train))

    # # 还原向量，parameters为更新后的向量
    # parameters = vector_split(sizes, final_vector)
