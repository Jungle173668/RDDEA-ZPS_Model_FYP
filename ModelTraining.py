import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

from RBFNN import RBFN

"""
=================
Final fitting MSE
=================
INPUT:
Funcs: 模型predict列表
MSEs: 模型对应的mse，即权重

OUTPUT：
result: 加权模型输出的预测结果
"""
def final_fitting_MSE(Funcs, MSEs, d):
    # 创建0向量
    zeros_tensor = torch.zeros(d)
    zeros_tensor = zeros_tensor.reshape([1,d])

    predict_result = torch.zeros(d)
    predict_result = predict_result.reshape([1,d])

    if torch.cuda.is_available():
        zeros_tensor = zeros_tensor.cuda()
        predict_result = predict_result.cuda()

    for i in range(len(MSEs)):
        predict_result += MSEs[i] * Funcs[i](zeros_tensor)
    
    return predict_result / sum(MSEs)

"""
=================
Final fitting MSE
=================
INPUT:
Funcs: 模型predict列表
MSEs: 模型对应的mse，即权重
x: 格式为[1, d]的tensor

OUTPUT：
result: 加权模型输出的预测结果
"""
# def origin_final_fitting_MSE(Funcs, MSEs, d, x):
#     predict_result = torch.zeros(d)
#     predict_result = predict_result.reshape([1,d])

#     for i in range(len(MSEs)):
#         predict_result += MSEs[i] * Funcs[i](x)
    
#     return predict_result / sum(MSEs)

class origin_final_fitting_MSE():
    def __init__(self, funcs, MSEs):
        self.funcs = funcs
        self.MSEs = MSEs
        self.sum_MSE = sum(MSEs)

    def func(self, x):
        y = 0
        for i in range(len(self.MSEs)):
            y += self.MSEs[i] * (self.funcs[i](x)).item()
        
        return y / self.sum_MSE


        

"""
=================================
Reverse RBFNN with Adam Optimizer
=================================
INPUT:
centers: 聚类中心 & 径向基函数中心
transformed_sigma: 径向基函数参数
mean_CP: 初始化RBFN用到的参数

N_train: 每个模型的最大训练次数
x_train: 训练集x
y_train: 训练集y

batch_size: 每次投入的数据集个数


OUTPUT:
best_val_loss: 模型的最终loss
best_model: 训练好的模型
"""

def Reverse_RBFNN_with_Adam(rbf, N_train, x_train, y_train, batch_size):
    # # ========== RBFN网络结构 ==========
    # rbf = RBFN(centers, transformed_sigma, mean_CP, d)

    # ============ 损失函数 =============
    if torch.cuda.is_available():
        loss_fn = torch.nn.MSELoss().cuda()  # 定义损失函数
    else:
        loss_fn = torch.nn.MSELoss()  # 定义损失函数

    # 参数类型转换
    params = rbf.parameters()  # 需要优化的参数
    for name, param in rbf.named_parameters():  # 转换为float
        param.data = param.data.float()

    # =========== Adam优化法优化参数 ============
    decay = 0.01  # Adam优化法中参数
    optimizer = torch.optim.Adam(params, lr=1e10-5,weight_decay=decay)

    # 动态调整学习率
    scheduler = ReduceLROnPlateau(optimizer, mode='min',factor=0.1,patience=5,verbose=True)

    # =========== 训练 ===========
    counter = 0  # loss连续上升次数
    patience = 10  # loss上升阈值
    losses = []

    for i in range(N_train):
        # print('第',i+1,'次训练')

        # 每次更新参数使用batch_size个数的数据
        num = int(len(x_train)/batch_size)

        for j in range(num):  # 分割数据集
            optimizer.zero_grad()  # 梯度清零

            if torch.cuda.is_available():
                x = rbf.forward(y_train[j*batch_size: (j+1)*batch_size].cuda())
                loss = loss_fn(x, x_train[j*batch_size: (j+1)*batch_size].cuda())

            else:
                x = rbf.forward(y_train[j*batch_size: (j+1)*batch_size])
                loss = loss_fn(x, x_train[j*batch_size: (j+1)*batch_size])
            
            # print('反向传播Begin')
            loss.backward()  # 反向传播
            # print('反向传播End')

            # print('更新参数Begin')
            optimizer.step()  # 更新参数
            # print('更新参数End')
    
        # =========== 评价loss，判断是否停止 ===========
        # 停止条件1:

        # 在测试集上计算loss                
        # test_x_bar = rbf.forward(y_predict)
        # test_loss = loss_fn(test_x_bar, x_predict)
        # scheduler.step(test_loss)

        # 在训练集上计算loss(后几个，是零点生成的，更接近解集)
        # print('计算预测值')
        test_x_bar = rbf.forward(y_train[-40:])
        # print('计算Loss值')
        test_loss = loss_fn(test_x_bar, x_train[-40:])
        
        # # 在全集上计算loss
        # index = torch.randperm(whole_y.size(0))[:40]
        # text_x_bar = rbf.forward(whole_y[index])
        # test_loss = loss_fn(text_x_bar, whole_x[index])
        
        # test_x_bar = rbf.forward(whole_y[0:40])
        # test_loss = loss_fn(test_x_bar, whole_x[0:40])

        scheduler.step(test_loss)

        # print(test_loss)
        losses.append(float(test_loss))

        if i == 0:
            best_val_loss = test_loss
            best_model = rbf
      
        # # 更换loss
        # test_loss = loss

        if test_loss >= best_val_loss:
            counter += 1
        else:
            counter = 0
            # best_val_loss = test_loss
            best_val_loss = loss
            best_model = rbf

        if counter > patience:
        # if counter > 100:
            print('Loss continues to increase. Break.')
            print(best_val_loss)
            break

        # # 停止条件2:
        # if i == 0:
        #     pre_loss = torch.tensor(float('inf'))
        #     best_model = rbf
        #     best_val_loss = pre_loss

        # test_x_bar = rbf.forward(y_predict)
        # test_loss = loss_fn(test_x_bar, x_predict)

        # if test_loss < pre_loss:
        #     best_val_loss = test_loss
        #     best_model = rbf
        #     counter = 0
        # else:
        #     counter = counter + 1

        # pre_loss = test_loss

        # if counter > 5:
        #     break

        # # 查看模型参数
        # for name, param in rbf.named_parameters():
        #     if param.grad is not None:
        #         print(name, param.grad.abs().mean())

    print('该轮模型训练完成.')
    print(best_val_loss)
    # Plot.Plot_loss(losses)

    # for param in rbf.parameters():
    #     print(param)

    return best_model, best_val_loss

def Origin_RBFNN_with_Adam(rbf, N_train, x_train, y_train, batch_size):
    # ============ 损失函数 =============
    if torch.cuda.is_available():
        loss_fn = torch.nn.MSELoss().cuda()  # 定义损失函数
    else:
        loss_fn = torch.nn.MSELoss()  # 定义损失函数

    # 参数类型转换
    params = rbf.parameters()  # 需要优化的参数
    for name, param in rbf.named_parameters():  # 转换为float
        param.data = param.data.float()

    # =========== Adam优化法优化参数 ============
    decay = 0.01  # Adam优化法中参数
    optimizer = torch.optim.Adam(params, lr=1e10-10,weight_decay=decay)

    # 动态调整学习率
    scheduler = ReduceLROnPlateau(optimizer, mode='min',factor=0.1,patience=5,verbose=True)

    # =========== 训练 ===========
    counter = 0  # loss连续上升次数
    patience = 10  # loss上升阈值
    losses = []

    for i in range(N_train):
        print('第',i+1,'次训练')

        # 每次更新参数使用batch_size个数的数据
        num = int(len(x_train)/batch_size)

        for j in range(num):  # 分割数据集
            optimizer.zero_grad()  # 梯度清零

            if torch.cuda.is_available():
                y = rbf.forward(x_train[j*batch_size: (j+1)*batch_size].cuda())
                loss = loss_fn(y, y_train[j*batch_size: (j+1)*batch_size].cuda())

            else:
                y = rbf.forward(x_train[j*batch_size: (j+1)*batch_size])
                loss = loss_fn(y, y_train[j*batch_size: (j+1)*batch_size])
            
            # print('反向传播Begin')
            loss.backward()  # 反向传播
            # print('反向传播End')

            # print('更新参数Begin')
            optimizer.step()  # 更新参数
            # print('更新参数End')


        # 停止条件1
        test_y_bar = rbf.forward(x_train[-30:])
        test_loss = loss_fn(test_y_bar, y_train[-30:])

        scheduler.step(test_loss)

        losses.append(float(test_loss))
        print(test_loss)

        if i == 0:
            best_val_loss = test_loss
            best_model = rbf

        if test_loss >= best_val_loss:
            counter += 1
        else:
            counter = 0
            best_val_loss = test_loss
            best_model = rbf

        if counter > patience:
            print('Loss continues to increase. Break.')
            print(best_val_loss)
            break
        
    print('该轮模型训练完成.')
    print(best_val_loss)

    return best_model, best_val_loss