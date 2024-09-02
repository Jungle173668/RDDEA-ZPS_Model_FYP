import torch
import torch.nn as nn

import random

random.seed(1)
torch.manual_seed(1)

class RBFN(nn.Module):
    '''
    INPUT: 
    centers: RBF层的中心结点
    my_sigma: RBF函数的参数
    mean_sigma: 即mean_CP, 用于初始化网络参数
    n_out: 输出维度, 通过函数的d确定（func.d）
    '''
    def __init__(self, centers, my_sigma, mean_sigma, n_out):
        super(RBFN, self).__init__()
        self.sigma = my_sigma
        self.mean_sigma = mean_sigma
        self.n_out = n_out
        self.num_centers = centers.size(0)  # 隐藏层节点的个数
        self.dim_centure = centers.size(1)  # 节点的维度

        # RBF中的参数 # 线性输出层
        if torch.cuda.is_available():
            self.centers = nn.Parameter(centers.float(),requires_grad=True).cuda()  # 转换为可训练的参数
            self.beta = nn.Parameter(my_sigma.float(), requires_grad=True).cuda()
            self.linear = nn.Linear(self.num_centers,self.n_out, bias=True).cuda()
            # self.bn = nn.BatchNorm1d(self.n_out).cuda()  # 添加批标准化层
        else:
            # self.centers = centers.float()
            self.centers = nn.Parameter(centers.float(),requires_grad=True)
            self.beta = nn.Parameter(my_sigma.float(), requires_grad=True)
            self.linear = nn.Linear(self.num_centers,self.n_out, bias=True)
            # self.bn = nn.BatchNorm1d(self.n_out)  # 添加批标准化层
        
        # # ########## 结构2
        # self.linear = nn.Linear(self.num_centers + self.dim_centure, self.n_out, bias=True)

        self.initialize_weights()  # 正态分布进行初始化

    def kernel_fun(self, batches):
    # '''
    # INPUT: barches, 一批x, n_input*d
    # OUTPUT: c, 一个数字
    # '''
        n_input = batches.size(0)
        A = self.centers.view(self.num_centers, -1).repeat(n_input, 1, 1)
        # print('===== A =====')
        # print(A)
        # print(A.shape)

        B = batches.view(n_input, -1).unsqueeze(1).repeat(1, self.num_centers, 1)
        # print('===== B =====')
        # print(B)
        # print(B.shape)

        # C = torch.exp(-self.beta.mul((A - B).pow(2).sum(2, keepdim=False)))
        # C = torch.exp(-self.sigma.mul((A - B).pow(2).sum(2, keepdim=False)))
        C = torch.exp(-self.sigma.mul(torch.abs(A-B).sum(2, keepdim=False)))
        # C = torch.exp(-self.sigma.mul(torch.abs(A-B)).sum(2, keepdim=False))
        # print('===== C =====')
        # print(C)
        # print(C.shape)
        
        return C
    
    def forward(self, batches):
        if torch.cuda.is_available():
            radial_val = self.kernel_fun(batches).cuda()
        else:
            radial_val = self.kernel_fun(batches)
        
        # 结构1
        class_score = self.linear(radial_val)

        # # 结构2
        # class_score = self.linear(torch.cat([batches, radial_val], dim=1))

        # print('打印class_score：', class_score.shape)
        # print(class_score)
        
        # class_score = self.linear(torch.cat([batches,radial_val],dim=1))

        return class_score
    
    def predict(self,y):
        if torch.cuda.is_available():
            y = torch.tensor(y).cuda().to(torch.float32)
            radial_val = self.kernel_fun(y).cuda()
        else:
            y = torch.tensor(y).to(torch.float32)
            radial_val = self.kernel_fun(y)

        class_score = self.linear(radial_val)

        return class_score
    
    def initialize_weights(self, ):
        # self.centers.data = self.centers.data.float()
        # self.beta.data = self.beta.data.float()
        # self.linear.weight.data = self.linear.weight.data.float()
        # self.linear.bias.data = self.linear.bias.data.float()
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.xavier_normal_(m.weight.data,gain=0.1)
                m.weight.data.normal_(0, 0.05)
                m.bias.data.zero_()
            elif isinstance(m, nn.ConvTranspose2d):
                # nn.init.xavier_normal_(m.weight.data,gain=0.1)
                m.weight.data.normal_(0, 0.02)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data,a = self.mean_sigma)
                m.weight.data.normal_(0, self.mean_sigma)

                m.bias.data.zero_()

    
    def print_network(self):
        num_params = 0
        for param in self.parameters():
            num_params += param.numel()
        print(self)
        print('Total number of parameters: %d' % num_params)