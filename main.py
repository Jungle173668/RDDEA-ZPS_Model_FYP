import numpy as np
import pandas as pd
import NonlinearFunctions
import DataGeneration
import random
from sklearn.cluster import KMeans
import torch
import math

import Plot
from RBFNN import RBFN
import ModelTraining
from FPA import fpa
from FPAOrigin import fpa_origin

from torch.optim.lr_scheduler import ReduceLROnPlateau


# torch.autograd.set_detect_anomaly(True)  # 开启异常检测

if __name__ == '__main__':
    # 设置参数
    D = 10  # 基础参数
    N_data = 20 * D  # 拉丁采样的数据量
    N_model = 20  # RBFNN的数量
    N_train = 300  # 模型训练次数（优化算法优化次数）
    N_centers = 5 * D  # 聚类中心的数量
    B = int(0.8 * N_data)  # 训练集的数量

    dbc_criterion = 1e-3  # dbc变化初始阈值
    w = int(0.1 * N_centers)  # 需要进行数据生成的比例
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
    func_list = [NonlinearFunctions.Func_N1()]  #,NonlinearFunctions.Func_N2()]
    # func_list = [NonlinearFunctions.Func_N1(),NonlinearFunctions.Func_N2(),
    #              NonlinearFunctions.Func_N3(),NonlinearFunctions.Func_N4(),
    #              NonlinearFunctions.Func_N5(),NonlinearFunctions.Func_N6()]

    # 记录寻优结果
    Euler_distances = []
    Best_losses = []
    Time_expenditure = []

    func_results = []
    func_var = []

    results_matrix = np.zeros((20,len(func_list)))  # 20行6列用于存储最终结果

    count = -1
    for func in func_list:
        count += 1
        results = []
        for exp_times in range(1):
            Funcs = []
            MSEs = []

            # for model_num in range(N_model):
            for model_num in range(10):
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
                # Reverse RBFNN with FPA Optimizer     
                '''   
                if is_origin == 2:
                    rbf = RBFN(centers, transformed_sigma, mean_CP, func.d)
                    best_model, best_val_loss = ModelTraining.Reverse_RBFNN_with_Adam(rbf, N_train, x_train, y_train, batch_size)
                    best_model, best_val_loss = fpa(para, loss_fn, best_model, x_train[-30:],y_train[-30:])

                    # parameters = best_model.state_dict()
                    # print(parameters)

                '''
                # Original model: RBFNN with FPA Optimizer     
                '''
                if is_origin == 1:
                    rbf = RBFN(centers, transformed_sigma, mean_CP, 1)
                    best_model, best_val_loss = ModelTraining.Origin_RBFNN_with_Adam(rbf, N_train, x_train, y_train, batch_size)


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
            if is_origin != 1:  # 改进后的模型(0和2)
                d = func.d
                
                # N_model个模型的加权预测结果
                predict_result = ModelTraining.final_fitting_MSE(Funcs, MSEs, d)
                print(predict_result)

                # 与真解的欧式距离
                Eular_dist = math.sqrt(d * loss_fn(predict_result, func.root))
                print(Eular_dist)
                results.append(Eular_dist)
            
            else:
                d = 1
                # N_model个模型的加权函数（输入x(torch)，输出y的范数(float)）
                x = torch.tensor([[0.5, -0.5]])
                final_func = ModelTraining.origin_final_fitting_MSE(Funcs, MSEs)
                # print(final_func.func(x))
                
                # FPA对final_func优化
                para_origin = [200, 0.5, 100, func.d, func.Lb, func.Ub, 1.5]
                best_sol, fmin = fpa_origin(para_origin, final_func.func)
                # 与真解的欧式距离
                print(best_sol)
                # print(np.linalg.norm(torch.tensor(best_sol) - func.root))
                results.append(np.linalg.norm(torch.tensor(best_sol) - func.root))
        
        # results_matrix[:,count] = results
        func_results.append(np.mean(results))
        func_var.append(np.std(results))
        
    # print(func_results)
    # print(func_var)
    # df = pd.DataFrame(results_matrix)
    # df.to_excel('data_ablation_result.xlsx',index=False)

    