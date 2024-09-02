import numpy as np
import pandas as pd
import NonlinearFunctions
import DataGeneration
import random
from sklearn.cluster import KMeans
import torch
import math
import time

import Plot
from RBFNN import RBFN
import ModelTraining
from FPA import fpa
from FPAOrigin import fpa_origin

from torch.optim.lr_scheduler import ReduceLROnPlateau

# 设置参数
D = 10  # 基础参数
N_data = 20 * D  # 拉丁采样的数据量
N_model = 10  # RBFNN的数量
N_train = 100  # 模型训练次数（优化算法优化次数）
N_centers = 8 * D  # 聚类中心的数量
B = int(0.8 * N_data)  # 训练集的数量

w = int(0.1 * N_centers)  # 需要进行数据生成的比例

dbc_criterion = 1e-3  # dbc变化初始阈值

alpha = 0.20  # 零点采样的数据比例

batch_size = 32  # 一次传入模型的数据数量

is_origin = 0  # 代表是否调转x和y. 0为改进后模型，1为原模型，2为改进后模型+FPA

# FPA基础参数设置，分别代表种群数量，转移概率，迭代次数，参数下、上边界，Lvey飞行参数，logistics混沌映射系数, 高斯生成的sigma
para = [100, 0.7, 300, -2, 2, 1.5, 3.99, 1e-3] 

# ============= 损失函数 =============
if torch.cuda.is_available():
    loss_fn = torch.nn.MSELoss().cuda()  # 定义损失函数
else:
    loss_fn = torch.nn.MSELoss()  # 定义损失函数

# 实例化函数
func_list = [NonlinearFunctions.Func_N1(),NonlinearFunctions.Func_N2(),NonlinearFunctions.Func_N3(),
             NonlinearFunctions.Func_N4(),NonlinearFunctions.Func_N5(),NonlinearFunctions.Func_N6()]
# func = func_list[0]

# 需要寻优的参数
# N_centers_choices = [10*i for i in range(1,16)]
result_matrix = np.zeros((20,15))  # 20行次实验，占据20行；一个center占一列
time_matix = np.zeros((20,15))

# 记录寻优结果
Best_losses = []

N_count = -1
# 依次搜索N_centers
for func in func_list:
    N_count = N_count + 1
    print('===================== func',N_count+1,' ======================')
    
    Euler_distances = []
    Time_expenditure = []

    for j in range(20):  # 训练20次
        starttime = time.time()
        print('================= 第',j,'次训练 ====================')
        Funcs = []
        MSEs = []

        # for model_num in range(N_model):
        for model_num in range(20):
            # ========== 生成数据集（DBC, ZCSampling）和参数 ==========
            # print('生成数据中')
            x_train, y_train, x_predict, y_predict, whole_x, whole_y, \
                centers, transformed_sigma, mean_CP = DataGeneration.Generate_fulldataset(func, N_data, B, N_centers, dbc_criterion, w, alpha, is_origin)        
            # print('生成数据结束')
            
            # ========== 建立神经网络 ==========
            '''
            # Reverse RBFNN with Adam Optimizer
            '''
            if is_origin == 0:
                # # print('建立RBFNN')
                rbf = RBFN(centers, transformed_sigma, mean_CP, func.d)
                # # print('建立RBFNN End')
                # # print('Training Begin')
                best_model, best_val_loss = ModelTraining.Reverse_RBFNN_with_Adam(rbf, N_train, x_train, y_train, batch_size)
                # # print('Training End')
            
            '''
            # Original model: RBFNN with FPA Optimizer     
            '''
            if is_origin == 1:
                rbf = RBFN(centers, transformed_sigma, mean_CP, 1)
                best_model, best_val_loss = ModelTraining.Origin_RBFNN_with_Adam(rbf, N_train, x_train, y_train, batch_size)

            '''
            # Reverse RBFNN with FPA Optimizer     
            '''   
            if is_origin == 2:
                rbf = RBFN(centers, transformed_sigma, mean_CP, func.d)
                best_model, best_val_loss = ModelTraining.Reverse_RBFNN_with_Adam(rbf, N_train, x_train, y_train, batch_size)
                best_model, best_val_loss = fpa(para, loss_fn, best_model, x_train[-30:],y_train[-30:])

            # 计算每个rbf的mse
            if torch.isnan(best_val_loss):
                model_num = model_num-1
            else:
                print(best_val_loss)
                mse = 1 / best_val_loss.item()
                MSEs.append(mse)
                Func = best_model.predict
                Funcs.append(Func)
            
            
        # =========== MSE加权, 聚合RBFNN, 求解方程 ===========
        d = func.d
        
        # N_model个模型的加权预测结果
        predict_result = ModelTraining.final_fitting_MSE(Funcs, MSEs, d)
        print(predict_result)

        endtime = time.time()
        
        # 与真解的欧式距离
        Eular_dist = math.sqrt(d * loss_fn(predict_result.cpu(), func.root))
        print(Eular_dist)
        Euler_distances.append(Eular_dist)
        Time_expenditure.append(endtime-starttime)
    
    print(Euler_distances)
    result_matrix[:,N_count] = Euler_distances

    print(Time_expenditure)
    time_matix[:,N_count] = Time_expenditure


print(result_matrix)
print(Time_expenditure)

df = pd.DataFrame(result_matrix)
df.to_excel('final_results.xlsx',index=False)

df = pd.DataFrame(time_matix)
df.to_excel('final_time.xlsx',index=False)