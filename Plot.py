import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
import matplotlib
import pyperclip
import math

# 设置字体为Times New Roman
matplotlib.rcParams['font.family'] = 'Times New Roman'
# 设置全局字体大小
matplotlib.rcParams.update({'font.size': 15})


def Plot_Scatter(x,y,flag):
    # 提取x的坐标
    x1 = x[:, 0]
    x2 = x[:, 1]

    # 创建一个三维坐标系
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    if flag==0:
    # 绘制三维坐标图
        ax.scatter(x1.numpy(), x2.numpy(), y.numpy(), color='red')
    else:
        ax.scatter(x1.numpy(), x2.numpy(), y.numpy(), color='blue')
    # 设置坐标轴标签
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_zlabel('Y')

    # 显示图形
    plt.show()

    # 绘制截面图
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    axs[0].scatter(x1.numpy(), x2.numpy(), color='red' if flag == 0 else 'blue', alpha=0.5)
    axs[0].set_xlabel('X1')
    axs[0].set_ylabel('X2')

    axs[1].scatter(x1.numpy(), y.numpy(), color='red' if flag == 0 else 'blue', alpha=0.5)
    axs[1].set_xlabel('X1')
    axs[1].set_ylabel('Y')

    axs[2].scatter(x2.numpy(), y.numpy(), color='red' if flag == 0 else 'blue', alpha=0.5)
    axs[2].set_xlabel('X2')
    axs[2].set_ylabel('Y')

    plt.show()

def Plot_loss(losses):
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Test Loss')
    plt.show()


def Line_graph(centers, result_data):
    result = result_data  # 20次实验，center数量从8到100，共93个值
    # 计算每个center数量下的均值和置信区间
    mean_mse = np.mean(result, axis=0)  # 计算均值

    # 计算置信区间
    conf_intervals = []
    
    for mse_values in result.T:  # 对每个center数量的MSE值进行计算
        conf_interval = stats.norm.interval(0.95, loc=np.mean(mse_values), scale=stats.sem(mse_values))        
        conf_intervals.append(conf_interval)

    conf_intervals = np.array(conf_intervals).T
    
    smooth_window = 10

    # 设置字体为Times New Roman
    matplotlib.rcParams['font.family'] = 'Times New Roman'
    # 设置全局字体大小
    matplotlib.rcParams.update({'font.size': 17})

    plt.plot(centers, mean_mse, color='#3131B7', label='Mean MSE')

    # 绘制置信区间
    plt.fill_between(centers, conf_intervals[0], conf_intervals[1], color='#3131B7', alpha=0.15, label='95% Confidence Interval')
    # 添加标题和标签
   
    plt.title('Mean ED with 95% Confidence Interval')
    plt.xlabel('Number of Centers')
    plt.ylabel('Mean ED')

    # 添加图例
    plt.legend()
    plt.grid(True)

    # 显示图形
    plt.show()


def Line_graph_time(centers, result_data, time_data):
    result_mse = result_data  # MSE 数据
    result_time = time_data  # 时间数据

    # 计算每个center数量下的均值和置信区间（MSE）
    mean_mse = np.mean(result_mse, axis=0)  # 计算均值

    # 计算置信区间（MSE）
    conf_intervals_mse = []
    
    for mse_values in result_mse.T:  # 对每个center数量的MSE值进行计算
        conf_interval_mse = stats.norm.interval(0.95, loc=np.mean(mse_values), scale=stats.sem(mse_values))        
        conf_intervals_mse.append(conf_interval_mse)

    conf_intervals_mse = np.array(conf_intervals_mse).T
    
    # 平均时间数据
    mean_time = np.mean(result_time, axis=0)
    # 计算置信区间（Time）
    conf_interval_time = []
    
    for time_values in result_time.T:  # 对每个center数量的时间值进行计算
        conf_interval_time.append(stats.norm.interval(0.95, loc=np.mean(time_values), scale=stats.sem(time_values)))
        
    conf_intervals_time = np.array(conf_interval_time).T

    # 创建图形和轴对象
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # 绘制 MSE 曲线和置信区间
    color = 'tab:blue'
    ax1.set_xlabel('ZPS proportion w')
    ax1.set_ylabel('Mean MSE', color=color)
    ax1.plot(centers, mean_mse, color=color, label='Mean MSE')
    ax1.fill_between(centers, conf_intervals_mse[0], conf_intervals_mse[1], color=color, alpha=0.15, label='95% Confidence Interval (MSE)')
    ax1.tick_params(axis='y', labelcolor=color)

    # 创建第二个纵轴对象
    ax2 = ax1.twinx()  

    # 绘制 Time 曲线和置信区间
    color = 'tab:red'
    ax2.set_ylabel('Mean Time Expenditure (s)', color=color)
    ax2.plot(centers, mean_time, color=color, linestyle='--', label='Mean Time')
    ax2.fill_between(centers, conf_intervals_time[0], conf_intervals_time[1], color=color, alpha=0.1, label='95% Confidence Interval (Time)')
    ax2.tick_params(axis='y', labelcolor=color)

    # 添加标题和图例
    fig.suptitle('Mean ED and Time with 95% Confidence Interval')
    fig.legend(loc='upper left')

    # 显示图形
    plt.show()

def Box_graph():
    mse_data_T10 = np.random.rand(15)  # T=10
    mse_data_T10 = np.array([0.45422836,0.40357719,0.40769836,0.39310638,0.39796959,0.43179105,
 0.39088894,0.38554914,0.40240354,0.39799752,0.41140481,0.38836792,
 0.40087591,0.38167781,0.39954195])
    
    mse_data_T20 = np.random.rand(15)  # T=20
    mse_data_T20 = np.array([0.40325532,0.41306204,0.36540259,0.40810119,0.4035872 ,0.40970114,
 0.41471355,0.40253481,0.40584805,0.40176394,0.45435863,0.40976507,
 0.35335018,0.4269599 ,0.38188462])

    mse_data_T30 = np.random.rand(15)  # T=30
    mse_data_T30 = np.array([0.39858415,0.40735086,0.4036552 ,0.39798962,0.39699095,0.40886193,
 0.3977887 ,0.39636868,0.4037408 ,0.4050258 ,0.40565208,0.39496996,
 0.39856381,0.39402283,0.39024201])
    
    mse_data_T40 = np.random.rand(15)
    mse_data_T40 = np.array([0.38167844,0.38340476,0.37527128,0.3865277 ,0.37960805,0.38009543,
 0.37639521,0.38318217,0.36986441,0.37424303,0.379445  ,0.37512269,
 0.37396228,0.38078978,0.38030473])

    # 组合MSE数据
    mse_data = [mse_data_T10, mse_data_T20, mse_data_T30, mse_data_T40]

    # 设置箱子宽度
    box_widths = 0.7

    # 设置箱体位置
    positions = [1, 2, 3, 4]

    # 设置颜色
    colors = ['#EA7773', '#6270F6', '#E6C677','#A8C942']

    # 设置字体为Times New Roman
    matplotlib.rcParams['font.family'] = 'Times New Roman'
    # 设置全局字体大小
    matplotlib.rcParams.update({'font.size': 13})
    plt.figure(figsize=(3.8,6))
    
    # 绘制箱线图
    for i, data in enumerate(mse_data):
        plt.boxplot(data, positions=[positions[i]], widths=box_widths, patch_artist=True, 
                    boxprops={'facecolor': colors[i], 'edgecolor': colors[i], 'linewidth': 1.5},
                    whiskerprops={'color': colors[i], 'linewidth': 1.5},
                    capprops={'color': colors[i], 'linewidth': 1.5},
                    medianprops={'color': 'white', 'linewidth': 1.5},)

    # 添加标题和标签
    plt.title('Boxplot of ED for N6')
    plt.xlabel('Model Name')
    plt.ylabel('ED')

    # 绘制箱体连接线
    for i in range(len(positions) - 1):
        plt.plot([positions[i], positions[i + 1]], [np.median(mse_data[i]), np.median(mse_data[i + 1])], color='gray', marker='o', markerfacecolor='white', markeredgecolor='none')

    # 设置横坐标标签的底色和边框
    plt.xticks(positions, ['Latin', 'Latin+DBC', 'Latin+ZPS','Full'], color='black', fontsize=10)

    # 添加网格线
    plt.grid(True, linestyle='-.', linewidth=0.5)

    # 显示图形
    plt.show()


# def Heat_map(w_values,l_values,mse_values):
#     # 绘制热力图
#     plt.imshow(mse_values, cmap='Blues', interpolation='nearest')
    
#     # 添加颜色条
#     plt.colorbar(label='Time')

#     # 设置横纵坐标标签
#     plt.xticks(np.arange(len(w_values)), w_values)
#     plt.yticks(np.arange(len(l_values)), l_values)

#     # 添加标题和标签
#     plt.title('Heatmap of Time for Different w and l Values')
#     plt.xlabel('w Values')
#     plt.ylabel('l Values')

#     # 显示图形
#     plt.show()

def point_comparation(y_values, x_values):
    # 设置圆点的大小和透明度
    point_size = 200  # 圆点大小
    alpha_value = 0.65  # 圆点透明度

    # 绘制散点图
    plt.scatter(y_values[:, 0], y_values[:, 1], c=x_values, cmap='viridis',
                s=point_size, alpha=alpha_value)
    plt.colorbar(label='X')

    # 设置坐标轴标签和标题
    plt.xlabel('Y1')
    plt.ylabel('Y2')
    plt.title('Comparison of Sampled Data and RBFNN Predicted Data')

    # 显示图形
    plt.show()





# # 绘制线图
# # MSE 数据
# choices = N_centers_choices = [10*i for i in range(1,16)]
# alpha_choices = [i/20 for i in range(13)]
# input_data = pd.read_csv('input.csv', sep='\t', header=None)
# input_data = input_data.values

# # 时间数据
# time_data = pd.read_csv('time.csv', sep='\t', header=None)
# time_data = time_data.values

# # 绘制图形
# Line_graph_time(alpha_choices, input_data, time_data)


# # 绘制箱图
# # 获取剪贴板内容
# clipboard_content = pyperclip.paste()

# # 将内容转换为一列数据
# data = pd.read_clipboard(header=None)
# column_data = data.iloc[:, 0].values

# # 将一列数据转换为NumPy数组

# print(np.array2string(column_data, separator=','))
# Box_graph()


# # 绘制热力图
# w_choices = np.array([i/10 for i in range(1,10)])  # 9个
# # w_choices = [int(i/10 * N_centers) for i in range(1,3)]
# # l_choices = np.array([float(1.5 * 10**-i) for i in range(1,11)])  # 10个
# l_choices = np.array([1.5*10**- i for i in range(1, 11)])  # 10个
# formatted_l_choices = [f'{num:.1e}' for num in l_choices]

# mse_values = pd.read_csv('input.csv', sep='\t', header=None)
# mse_values = np.array(mse_values.values).T

# time_values = pd.read_csv('time.csv', sep='\t', header=None)
# time_values = np.array(time_values.values).T

# Heat_map(w_choices, formatted_l_choices, time_values)


# 绘制预测-真实值对比图


# import numpy as np
# import matplotlib.pyplot as plt

# # 生成示例数据
# num_points = 100
# y_values = np.random.rand(num_points, 2)  # 两个维度的y值
# x_values = np.random.rand(num_points)     # x值

# point_comparation(y_values,x_values)