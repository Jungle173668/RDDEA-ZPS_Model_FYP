import numpy as np
import math
import torch

'''
需要求解的函数
输入输出均为list
'''

class Func_N1:
    def __init__(self):
        self.d = 2  # 维度
        self.Lb = -3 * np.ones(self.d)  # 每一维变量的下界
        self.Lb[0] = 0

        self.Ub = 3 * np.ones(self.d)   # 每一维变量的上界
        self.Ub[1] = 0
        self.name = 'N1'

        self.root = torch.tensor([0.5, -0.5]).reshape(1,self.d)
    
    def func_N(self, x):
        y1 = pow(x[0],2) - pow(x[1],2)
        y2 = 1 - abs(x[0]-x[1])

        return [y1,y2]


class Func_N2:
    def __init__(self):
        self.d = 2  # 维度
        self.Lb = -3 * np.ones(self.d)  # 每一维变量的下界
        self.Ub = 3 * np.ones(self.d)   # 每一维变量的上界
        self.Lb[0] = 0.71
        
        self.Lb[1] = -1.5
        self.Ub[1] = 0

        self.name = 'N2'

        self.root = torch.tensor([1, -1]).reshape(1,self.d)
    
    def func_N(self, x):
        y1 = pow(x[0],2) - x[1] - 2
        y2 = x[0] + np.sin(math.pi * x[1]/2)

        return [y1,y2]

# def func_N2(x):
#     y1 = pow(x[0],2) - x[1] - 2
#     y2 = x[0] + np.sin(math.pi * x[1]/2)

#     return [y1,y2]

class Func_N3:
    def __init__(self):
        self.d = 2  # 维度
        self.Lb = -0 * np.ones(self.d)  # 每一维变量的下界
        self.Ub = 3 * np.ones(self.d)   # 每一维变量的上界

        self.Ub[0] = 1.5
        self.Lb[1] = 1.5

        self.root = torch.tensor([0, 3]).reshape(1,self.d)

        self.name = 'N3'
    
    def func_N(self, x):
        y1 = x[0]+x[1]-3
        y2 = pow(x[0],2) + pow(x[1],2) - 9

        return [y1,y2]


class Func_N4:
    def __init__(self):
        self.d = 2  # 维度
        self.Lb = -1 * np.ones(self.d)  # 每一维变量的下界
        self.Ub = 0 * np.ones(self.d)   # 每一维变量的上界

        self.root = torch.tensor([-0.1733, -0.2561]).reshape(1,self.d)

        self.name = 'N4'
    
    def func_N(self, x):
        y1 = x[0] - np.sin(2*x[0] + 3*x[1]) - np.cos(3*x[0]-5*x[1])
        y2 = x[1] - np.sin(x[0]-2*x[1]) + np.cos(x[0]+3*x[1])

        return [y1,y2]


class Func_N5:
    def __init__(self):
        self.d = 10
        self.Lb = 0 * np.ones(self.d)  # 每一维变量的下界
        self.Ub = 0.5 * np.ones(self.d)   # 每一维变量的上界

        self.root = torch.tensor([0.2578, 0.3811, 0.2787, 0.2007, 0.4453,
                                  0.1492, 0.4320, 0.0734, 0.3460, 0.4273]).reshape(1,self.d)
        self.name = 'N5'
    
    def func_N(self, x):
        y1 = x[0] - 0.25428722 - 0.18324757 * x[3] * x[2] * x[8]
        y2 = x[1] - 0.37842197 - 0.16275449 * x[0] * x[9] * x[5]
        y3 = x[2] - 0.27162577 - 0.16955071 * x[0] * x[1] * x[9]
        y4 = x[3] - 0.19807914 - 0.15585316 * x[6] * x[0] * x[5]
        y5 = x[4] - 0.44166728 - 0.19950920 * x[6] * x[5] * x[2]
        y6 = x[5] - 0.14654113 - 0.18922793 * x[7] * x[4] * x[9]
        y7 = x[6] - 0.42937161 - 0.21180486 * x[1] * x[4] * x[7]
        y8 = x[7] - 0.07056438 - 0.17081208 * x[0] * x[6] * x[5]
        y9 = x[8] - 0.34504906 - 0.19612740 * x[9] * x[5] * x[7]
        y10 = x[9] - 0.42651102 - 0.21466544 * x[3] * x[7] * x[0]

        return [y1,y2,y3,y4,y5,y6,y7,y8,y9,y10]
    
class Func_N6:
    def __init__(self):
        self.d = 20
        self.Lb = 0.5 * np.ones(self.d)  # 每一维变量的下界
        self.Ub = 1.5 * np.ones(self.d)   # 每一维变量的上界

        self.root = torch.tensor([1,1,1,1,1,
                                  1,1,1,1,1,
                                  1,1,1,1,1,
                                  1,1,1,1,1]).reshape(1,self.d)
        self.name = 'N6'

    def func_N(self, x):
        y = []
        x = np.array(x)
        sum_of_x = np.sum(x)

        for i in range(len(x)-1):
            yi = x[i] + sum_of_x - (len(x)+1)
            y.append(yi)
        
        yd = np.prod(x) - 1
        y.append(yd)

        return y
