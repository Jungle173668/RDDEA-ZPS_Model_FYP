import numpy as np
from scipy.stats.qmc import LatinHypercube
from scipy.stats.qmc import scale
import random
import math

import NonlinearFunctions
import torch

import Plot

from sklearn.cluster import KMeans
'''
========================
LatinHypercube Sampling
========================
INPUT: 
Lb, Ub：各个维度的上下界，为列表
d：采样维度
n：采样数
goal_func：被采样的函数

OUTPUT:
data：采样后的数据集，列表，里面的元素为[x,y]，x和y分别为列表(相当于列表三层嵌套)。
'''

def generate_Latin(Lb,Ub,d,n,goal_func):

    sampler = LatinHypercube(d, seed=0)
    sample = sampler.random(n)
    sample = scale(sample,Lb,Ub)
    
    data = []
    for i in range(n):
        temp_y = goal_func(sample[i])
        data.append([list(sample[i]), list(temp_y)])

    return data


# 用例
# func = NonlinearFunctions.Func_N1()
# data = generate_Latin(func.Lb,func.Ub,func.d,50,func.func_N1)

'''
================
KMeans Clustring
================
仅对train_data进行处理
INPUT:
x_train: 训练集
n_centers: 聚类中心的数量（以及RBFNN中间神经元的数量）

OUTPUT:


'''

def KMeans_Clustering(x_train, n_centers):

    len_x = x_train.size(0)
    # if torch.cuda.is_available():
    #     cluster_model = KMeans(n_clusters = n_centers, max_iter=1000).fit(x_train.cuda())
    # else:
    #     cluster_model = KMeans(n_clusters = n_centers, max_iter=1000).fit(x_train.cpu())

    cluster_model = KMeans(n_clusters = n_centers, max_iter=1000).fit(x_train.cpu())

    centers = cluster_model.cluster_centers_  # 聚类中心
    indices = cluster_model.predict(x_train)  # 点所属类别
    
    dist = {}  # 每个类别点到该类中心的距离累计
    points_count = {}  # 每个类别的点数

    # 建立字典
    for k in set(cluster_model.labels_):
        dist['{}'.format(k)] = 0  # 字符串化的数字为key，所有key的value为0.
        points_count['{}'.format(k)] = 0  # 同上
    
    # 填充字典: 遍历每个点
    for i in range(len_x):
        k = indices[i]
        points_count['{}'.format(k)] += 1
        dist['{}'.format(k)] += np.linalg.norm(x_train[i].numpy()-centers[k])  # 计算到k类中心的欧式距离
    
    # 所有点到各自中心距离的平均, 一个数字
    set_mean_dist = np.array([value for key, value in dist.items() if value!=0]).mean()
    
    # 计算各个类别CP值, 求平均, 一个数字。用于判断CP值是否收敛（变化量是否超过阈值）。
    mean_CP = np.array([value/points_count['{}'.format(k)] for key, value in dist.items() if (value!=0 and points_count['{}'.format(k)]!=0)]).mean()
    
    # 计算RBFNN中sigma参数
    # sigma使用该组CP值
    sigmas = []
    for k in set(cluster_model.labels_):
        # 只有一个点或没有点的类，设置为平均距离
        if (dist['{}'.format(k)] == 0) or (points_count['{}'.format(k)]==0):
            dist['{}'.format(k)] = set_mean_dist
        
        # 计算sigma并加入列表
        sigmas.append(1 / (2 * pow(dist['{}'.format(k)]/ points_count['{}'.format(k)], 2)))

    transformed_sigma = torch.tensor(sigmas, dtype=torch.float)  # 转换sigma，在网络中直接相乘即可
    centers = torch.tensor(centers, dtype=torch.float)  # centers转换为tensor

    # 有gpu转移到gpu上
    if torch.cuda.is_available():
        transformed_sigma = transformed_sigma.cuda()
        centers = centers.cuda()

    return centers, transformed_sigma, mean_CP, dist, indices, points_count



'''
==============
DBC genetation
==============
对y聚类，得到需要扩充的y的类别
找到对应的x，对x进行扩充得到x'
计算出x'对应的y', (x',y')即为新的数据集

INPUT:
indices: 数据点所属类别
dist: 数据点到中心的距离和，共N_centers个数
x_train, y_train: 训练集
l: dleta x 的波动范围

OUTPUT:
x_train, y_train: 生成新数据后的数据集
indices: 输出新数据的所属类别
'''

def DBC(x_train, y_train, dist, point_count, indices, l, w, func, Lb, Ub):

    # 计算每组CP值并倒序排列
    CPs = np.array([value/point_count['{}'.format(key)] for key, value in dist.items()])
    CPs_dict = {index: value for index, value in enumerate(CPs)}

    sorted_CPs = sorted(CPs_dict.items(), key=lambda x: x[1], reverse=True)
    
    # 数据的维度
    d = x_train.size()[1]

    # 对前w个类进行数据生成
    for i in range(w):
        k = int(sorted_CPs[i][0])  # 取出类别编号
        k_index = [i for i, j in enumerate(indices) if j==k]  # 类别下标, y的下标
        
        # 对每个类别中的数据做生成
        # 由于不均匀的是y，但由y无法获取x，因此对y对应的x做生成，计算出相应的y
        for j in k_index:
            # 计算delta_x
            # if torch.cuda.is_available():
            #     delta_x = torch.randn(d).cuda() * l
            # else:
            #     delta_x = torch.randn(d) * l
            
            delta_x = torch.randn(d) * l

            new_x = delta_x + x_train[j]
            new_x = new_x.reshape([1,d])

            # 统一化x的范围
            Lb_tensor = torch.tensor(Lb, dtype=new_x.dtype, device=new_x.device)
            Ub_tensor = torch.tensor(Ub, dtype=new_x.dtype, device=new_x.device)
            # print(Lb_tensor[0])

            for i in range(new_x.shape[1]):
                if new_x[0][i] < Lb_tensor[i]:
                    new_x[0][i] = Lb_tensor[i]
                if new_x[0][i] > Ub_tensor[i]:
                    new_x[0][i] = Ub_tensor[i]    
            
            # 计算delta_y
            new_y = torch.tensor(func(np.array(new_x.cpu())[0]))
            new_y = new_y.reshape([1,d])
            # if torch.cuda.is_available():
            #     new_y = new_y.cuda()
            
            # 拼接数据集
            x_train = torch.cat((x_train, new_x),0)
            y_train = torch.cat((y_train, new_y),0)
            indices = np.append(indices,k)
            
            # print(x_train)
            # print(y_train)

    return x_train, y_train, indices        

'''
======================
Zero-crossing Sampling
======================
INPUT:
x_train, y_train: 训练集

OUTPUT:
x_train, y_train: 生成新数据后的数据集
'''

def ZCSampling(x_train, y_train, alpha, indices, l, func, Lb, Ub):

    # 计算出每个y的范数
    distances = []
    for y in y_train:
        distances.append(np.linalg.norm(y))
    
    # 选出前alpha比例的点
    N_points = int(len(y_train)*alpha)
    sorted_indices = np.argsort(distances)  # 排序后下标
    top_index = sorted_indices[:N_points]  # 前alpha比例下标
    
    # top_x = x_train[top_index]  # 需要进行数据生成的x
    # top_indices = indices[top_index]  # 对应x所属的类别

    # 数据生成
    d = x_train.size()[1]

    for j in top_index:
        x = x_train[j]
        x_index = indices[j]
        
        # 生成delta_x
        # if torch.cuda.is_available():
        #     delta_x = torch.randn(d).cuda() * l
        # else:
        #     delta_x = torch.randn(d) * l
        delta_x = torch.randn(d) * l
        # 生成新的x
        new_x = delta_x + x
        new_x = new_x.reshape([1,d])

        # 统一化x的范围
        Lb_tensor = torch.tensor(Lb, dtype=new_x.dtype, device=new_x.device)
        Ub_tensor = torch.tensor(Ub, dtype=new_x.dtype, device=new_x.device)
        # print(Lb_tensor[0])

        for i in range(new_x.shape[1]):
            if new_x[0][i] < Lb_tensor[i]:
                new_x[0][i] = Lb_tensor[i]
            if new_x[0][i] > Ub_tensor[i]:
                new_x[0][i] = Ub_tensor[i]  

        # 生成新的y
        new_y = torch.tensor(func(np.array(new_x.cpu())[0]))
        new_y = new_y.reshape([1,d])
        # if torch.cuda.is_available():
        #     new_y = new_y.cuda()
        
        # 拼接数据集
        x_train = torch.cat((x_train, new_x),0)
        y_train = torch.cat((y_train, new_y),0)
        indices = np.append(indices,x_index)

    return x_train, y_train, indices       

'''
=======================
    生成数据 - 总流程
=======================
INPUT:
func: 目标函数
N_data: 拉丁采样的数据量
B: 训练集的数量
N_centers: 聚类中心数量
dbc_criterion: dbc变化初始阈值
w: 需要进行数据生成的比例
alpha: 零点采样的数据比例

OUTPUT:

'''

def Generate_fulldataset(func, N_data, B, N_centers, dbc_criterion, w, alpha, is_origin):

    # print('Latin Begin')
    # 拉丁超立方采样
    data = generate_Latin(func.Lb,func.Ub,func.d,N_data,func.func_N)
    # print('Latin End')
    
    # 训练集
    train_data = []
    t_choice = random.sample(range(len(data)), B)  # 随机选取下标
    for i in range(len(t_choice)):
        train_data.append(data[t_choice[i]])

    # 测试集
    predict_data = []
    p_choice = list(set(range(len(data))).difference(set(t_choice)))
    for i in range(len(p_choice)):
        predict_data.append(data[p_choice[i]])

    # 转换为array 
    train_data = np.array(train_data)  # train_data.shape = (88,2,2)
    predict_data = np.array(predict_data)
    whole_data = np.array(data)  # 后续用于评估聚合模型的整体MSE

    # 分离x和y，转换为tensor
    x_train = torch.tensor(train_data[:, 0], dtype=torch.float)
    y_train = torch.tensor(train_data[:, 1], dtype=torch.float)  # 同为二维，因此不用reshape了

    x_predict = torch.tensor(predict_data[:, 0], dtype=torch.float)
    y_predict = torch.tensor(predict_data[:, 1], dtype=torch.float)

    whole_x = torch.tensor(whole_data[:, 0], dtype=torch.float)
    whole_y = torch.tensor(whole_data[:, 1], dtype=torch.float)

    # 判断有无gpu，有则移动变量
    if torch.cuda.is_available():
        print(torch.cuda.is_available)
        x_train = x_train.cuda()
        y_train = y_train.cuda()

        x_predict = x_predict.cuda()
        y_predict = y_predict.cuda()

        whole_x = whole_x.cuda()
        whole_y = whole_y.cuda()

    '''
    centers, transformed_sigma, mean_CP是RBFNN中需要用到的参数
    dist: 各组内距离之和。用于dbc后新的transformed_sigma计算
    indices：各点所属类，dbc要用
    points_count：用于算mean_CP，DBC和RBFNN中要
    '''

    # print('KMeans Begin')
    if is_origin != 1:
        centers, transformed_sigma, mean_CP, dist, indices, points_count \
                = KMeans_Clustering(y_train.cpu(), N_centers)
    else:
        centers, transformed_sigma, mean_CP, dist, indices, points_count \
                = KMeans_Clustering(x_train.cpu(), N_centers)
    # print('KMeans End')
    
    # =========  DBC，用于生成均匀数据 ==========
    count = 0  # dbc进行的次数
    l =  math.sqrt(np.sum((func.Ub-func.Lb)**2) / func.d) * 5 * 10e-3 # x的波动变化范围
    
    # print('DBC前：')
    # Plot.Plot_Scatter(x_train,torch.norm(y_train,dim=1),0)

    # 符合条件，则继续进行dbc生成
    while abs(dbc_criterion-mean_CP)/dbc_criterion > 5e-5 * func.d and count<3:
        dbc_criterion = mean_CP  # 计算CP平均值（变化前）
        last_n = x_train.size(0)  # 记录最后一个下标
        
        # DBC生成数据
        x_train, y_train, indices = DBC(x_train.cpu(), y_train.cpu(), dist, points_count, indices, l, w, func.func_N, func.Lb, func.Ub)
        
        if is_origin != 1:
            # 零点采样生成数据
            x_train, y_train, indices = ZCSampling(x_train.cpu(), y_train.cpu(), alpha, indices, l, func.func_N, func.Lb, func.Ub)

        # 处理新的数据点
        for i in range(last_n, x_train.size(0)):
            k = indices[i]
            points_count['{}'.format(k)] += 1
            dist['{}'.format(k)] += np.linalg.norm(y_train[i].cpu().numpy() - centers[k].cpu().numpy())
        
        # 重新计算mean_CP
        mean_CP = np.array([value / points_count[key] for key, value in dist.items() if value!=0]).mean()

        # 重新计算sigmas
        sigmas = []
        for k in sorted(set(indices)):
            # sigmas.append(1 / (2 * pow(dist['{}'.format(k)]/ points_count['{}'.format(k)], 2)))
            sigmas.append(1 / (2 * pow(dist['{}'.format(k)], 2)))
        
        # transformed_sigma = torch.tensor(sigmas, dtype=torch.float)
        transformed_sigma = torch.tensor(sigmas, dtype=torch.float)
        transformed_sigma = transformed_sigma * 1/torch.min(transformed_sigma)
        # print(transformed_sigma)

        if torch.cuda.is_available():
            transformed_sigma = transformed_sigma.cuda()

        count += 1

    # # 绘制数据点分布
    # print('DBC后：')
    # Plot.Plot_Scatter(x_train,torch.norm(y_train,dim=1),1)

    # 转换数据类型为float
    x_train = x_train.float()
    y_train = y_train.float()

    x_predict = x_predict.float()
    y_predict = y_predict.float()

    whole_x = whole_x.float()
    whole_y = whole_y.float()

    if is_origin == 1:
        y_train = torch.norm(y_train, p=1, dim=1, keepdim=True)
        y_predict = torch.norm(y_predict, p=1, dim=1, keepdim=True)
        whole_y = torch.norm(whole_y, p=1, dim=1, keepdim=True)

    if torch.cuda.is_available():
        x_train = x_train.cuda()
        y_train = y_train.cuda()

        x_predict = x_predict.cuda()
        y_predict = y_predict.cuda()

        whole_x = whole_x.cuda()
        whole_y = whole_y.cuda()


    return x_train, y_train, x_predict, y_predict, whole_x, whole_y, centers, transformed_sigma, mean_CP
