from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import random
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from model import Net_D, Net_G, NetD_Aux


#  设置网络参数
def set_parameter():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='data_set/train', help='训练数据保存路径')
    parser.add_argument('--data_set', default='data_set', help='数据集文件名')
    parser.add_argument('--workers', type=int, default=0, help='工作线程数')
    parser.add_argument('--batchSize', type=int, default=64, help='输入图片批量大小')
    parser.add_argument('--imageSize', type=int, default=128, help='图片尺寸')
    parser.add_argument('--nz', type=int, default=100, help='初始噪音向量的大小')
    parser.add_argument('--ngf', type=int, default=64, help='生成网络中基础feature数目')
    parser.add_argument('--ndf', type=int, default=64, help='判别网络中基础feature数目')
    parser.add_argument('--nef', type=int, default=64, help='第一个转换层中的编码器滤波器')
    parser.add_argument('--nc', type=int, default=3, help='彩色图片通道数')
    parser.add_argument('--niter', type=int, default=200, help='网络训练过程中epoch数目')
    parser.add_argument('--lr', type=float, default=0.0002, help='初始学习率')
    parser.add_argument('--beta1', type=float, default=0.5, help='使用Adam优化算法中的β1参数值')
    parser.add_argument('--cuda', default=False, help='是否可以使用CUDA训练')
    parser.add_argument('--ngpu', type=int, default=1, help='使用GPU的数量')
    # model/netG_model.pth 使用模型继续训练
    parser.add_argument('--netG', default='', help="模型G的保存路径")
    # model/netD_model.pth   使用模型继续训练
    parser.add_argument('--netD', default='', help="模型G的保存路径")
    parser.add_argument('--outf', default='.', help='当前路径')
    parser.add_argument('--manualSeed', type=int, help='随机数')
    parser.add_argument('--nBottleneck', type=int, default=4000, help='训练的最大瓶颈')
    parser.add_argument('--overlapPred', type=int, default=4, help='重叠的边缘')
    parser.add_argument('--wtl2', type=float, default=0.998, help='0 意味着不用否则使用该权重')
    parser.add_argument('--wtlD', type=float, default=0.001, help='0 means do not use else use with this weight')
    parser.add_argument('--overlapL2Weight ', type=int, default=10, help="L2的权重")
    opt = parser.parse_args()
    return opt


# 创建文件目录
def make_file_dir():
    opt = set_parameter()
    try:
        os.makedirs("{}".format(opt.data_root))
        os.makedirs("result/train/cropped")
        os.makedirs("result/train/real")
        os.makedirs("result/train/recon")
        os.makedirs("model")
        print("文件目录创建成功！")
    except OSError:
        print("文件目录已存在")


# 加载数据集
def load_data_set():
    opt = set_parameter()
    # transforms:将多个transform组合起来使用
    transform = transforms.Compose([transforms.Scale(opt.imageSize),
                                    transforms.CenterCrop(opt.imageSize),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    # 将文件夹中的图像加载到 dataset 中，数据抽象
    # print(dataset.imgs) 保存所有图片
    dataset = dset.ImageFolder(root=opt.data_root, transform=transform)
    # 将 dataset 中图像加载到 dataloader 中
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                             shuffle=True, num_workers=opt.workers, drop_last=True)
    return dataloader


# 初始化权重
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


# 选择设备
def check_device():
    if torch.torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


# 设置随机种子
def set_seed():
    opt = set_parameter()
    # 生成随机数
    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)
    print("Random Seed: ", opt.manualSeed)
    # 设置不同随机数
    random.seed(opt.manualSeed)
    # 为CPU设置随机种子
    torch.manual_seed(opt.manualSeed)
    if opt.cuda:
        # 有多个 GPU 为GPU设置随机种子
        torch.cuda.manual_seed_all(opt.manualSeed)
    else:
        pass


# 训练网络
def train():
    dataloader = load_data_set()  # 读取数据集数据
    set_seed()  # 设置随机种子
    opt = set_parameter()  # 加载参数
    resume_epoch = 0  # 保存已经训练过的网络的 epoch
    real_label = random.uniform(0.9, 1)   # 定义真实图片的标签
    fake_label = random.uniform(0, 0.1)   # 定义假图的标签
    # 判断是否可以使用 GPU
    if torch.cuda.is_available() and opt.cuda:
        # 增加网络运行的效率
        cudnn.benchmark = True
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    # 选择设备
    device = check_device()
    # 加载网络 G
    netG = Net_G(opt).to(device)
    netG.apply(weights_init)
    if opt.netG != '':
        netG.load_state_dict(torch.load(opt.netG, map_location=lambda storage, location: storage)['state_dict'])
        resume_epoch = torch.load(opt.netG)['epoch']
    print(netG)  # 打印G网络
    # 加载网络 G
    netD = Net_D(opt).to(device)
    netD.apply(weights_init)
    if opt.netD != '':
        netD.load_state_dict(torch.load(opt.netD, map_location=lambda storage, location: storage)['state_dict'])
        resume_epoch = torch.load(opt.netD)['epoch']
    print(netD)  # 打印D网络
    # 加载辅助 D 网络
    netD_Aux = NetD_Aux(opt).to(device)
    netD_Aux.apply(weights_init)
    print(netD_Aux)  # 打印辅助D网络

    criterionBCE = nn.BCELoss()  # 定义交叉熵损失
    criterionMSE = nn.MSELoss()  # 均方损失函数

    # 定义Tensor
    input_real = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)  # 真实图片
    input_cropped = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)  # 有mask的图片
    real_center = torch.FloatTensor(opt.batchSize, 3, opt.imageSize // 2, opt.imageSize // 2)  # 真实中心
    r_label = torch.FloatTensor(opt.batchSize)  # 标签
    f_label = torch.FloatTensor(opt.batchSize)  # 标签
    # 如果CUDA可用将数据放到 CUDA（GPU）
    if opt.cuda:
        netD.cuda()
        netG.cuda()
        criterionBCE.cuda()
        criterionMSE.cuda()
        input_real = input_real.cuda()
        input_cropped = input_cropped.cuda()
        real_center = real_center.cuda()
        r_label = r_label.cuda()
        f_label = f_label.cuda()
    else:
        pass

    # 数据 Variable 化
    input_real = Variable(input_real)
    input_cropped = Variable(input_cropped)
    real_center = Variable(real_center)
    r_label = Variable(r_label)
    f_label = Variable(f_label)

    # 设置优化器
    # optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    # optimizerD_Aux = optim.Adam(netD_Aux.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    # optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

    optimizerD = optim.RMSprop(netD.parameters(), lr=opt.lr)
    optimizerD_Aux = optim.RMSprop(netD_Aux.parameters(), lr=opt.lr)
    optimizerG = optim.RMSprop(netG.parameters(), lr=opt.lr)


    # 开始训练
    for epoch in range(resume_epoch, opt.niter):
        # 对于可索引序列，enumerate可以同时获得索引和值
        for i, data in enumerate(dataloader, 0):  # i:索引 data:图像数据和tensor量
            real_cpu, _ = data  # 存储图片
            # vutils.save_image(real_cpu, 'result/temp/real_samplesi{}_epoch{}.png'.format(i, epoch + 1))
            real_center_cpu = real_cpu[:, :, 32:96, 32:96]   # 截取需要的部分Y y1:y2 X x1:x2
            batch_size = real_cpu.size(0)  # 设置尺寸64
            input_real.data.resize_(real_cpu.size()).copy_(real_cpu)  # 真实图片
            input_cropped.data.resize_(real_cpu.size()).copy_(real_cpu)  # 真实图片
            real_center.data.resize_(real_center_cpu.size()).copy_(real_center_cpu)  # 真实的面部中心
            # 构造加入mask的图片
            input_cropped.data[:, 0, 34:94, 34:94] = 2 * 117.0 / 255.0 - 1.0
            input_cropped.data[:, 1, 34:94, 34:94] = 2 * 104.0 / 255.0 - 1.0
            input_cropped.data[:, 2, 34:94, 34:94] = 2 * 123.0 / 255.0 - 1.0
            # vutils.save_image(input_cropped, 'result/temp/cropped_{}_epoch{}.png'.format(i+1, epoch + 1))

            # print("real_cpu:", real_cpu.size())
            # print("input_cropped:", input_cropped.size())
            # print("real_center:", real_center.size())

            # 训练D网络
            netD.zero_grad()   # 梯度归零
            netD_Aux.zero_grad()
            r_label.data.resize_(batch_size).fill_(real_label)  # 构造真标签
            # 将真实图片输入到 netD
            output_real_center_d = netD(real_center)   # 真实局部
            lossd_real_center = criterionBCE(output_real_center_d, r_label)  # 计算损失
            output_real = netD_Aux(input_real)   # 真实全局
            lossd_real = criterionBCE(output_real, r_label)
            lossreal = (lossd_real_center + lossd_real) * 1/2
            lossreal.backward()  # 损失回传
            lossd_real_ave = output_real_center_d.data.mean()  # D网络损失的均值

            # 将生成图片输入到 netD
            output_fake_g = netG(input_cropped)
            # print(output_fake_g.size())
            f_label.data.resize_(batch_size).fill_(fake_label)  # 构造假的标签
            # =============================================================================
            lossd_localBCE = criterionBCE(netD(output_fake_g.detach()), f_label)
            recon_global_img = input_cropped.clone()
            recon_global_img.data[:, :, 32:96, 32:96] = output_fake_g.data
            lossd_globalBCE = criterionMSE(netD_Aux(recon_global_img.detach()), f_label)
            # =============================================================================
            loss_d = (lossd_localBCE + lossd_globalBCE) * 1/2
            loss_d.backward()  # 损失回传
            lossd_fake_ave = output_fake_g.data.mean()  # D网络判别G损失的均值
            # 损失和
            loss_D = lossd_real_center + loss_d
            optimizerD.step()   # 逐步优化
            optimizerD_Aux.step()

            # 训练G网络
            netG.zero_grad()  # 梯度归零
            r_label.data.resize_(batch_size).fill_(real_label)  # 构造真标签
            fake = netG(input_cropped)
            loss_local_G_D = criterionBCE(netD(fake), r_label)  # 计算与标签损失
            lossg_local = criterionMSE(fake, real_center)  # 计算局部损失
            # ===================================================================
            recon_global = input_cropped.clone()
            recon_global.data[:, :, 32:96, 32:96] = fake.data
            loss_global_G_D = criterionBCE(netD_Aux(recon_global), r_label)
            lossg_global = criterionMSE(recon_global, input_real)  # 全局损失
            # ====================================================================
            # 计算损失和
            loss_g = (0.002 * 1/2 * (loss_local_G_D + loss_global_G_D)) + (0.998 * 1/2 * (lossg_local + lossg_global))
            loss_g.backward()   # 损失回传
            optimizerG.step()   # 优化
            # d_g = output.data.mean()

            # 打印结果
            print("Epoch:[{}/{}][{}/{}] real_D:{:.4f} fake_D:{:.4f} Loss_D:{:.4f} Loss_G:{:.4f}".format(
                    epoch+1, opt.niter, i+1, len(dataloader), lossd_real_ave, lossd_fake_ave, loss_D, loss_g
                ))

            if (epoch + 1) % 1 == 0:
                recon_image = input_cropped.clone()
                vutils.save_image(fake.data, 'result/temp/fake_{}_epoch{}.png'.format(i+1, epoch + 1))
                recon_image.data[:, :, 32:96, 32:96] = fake.data
                # 修复图
                vutils.save_image(recon_image.data, 'result/temp/reco_{}_epoch{}.png'.format(i+1, epoch + 1))
        # 存储模型
        if (epoch + 1) % 20 == 0:
            torch.save({'epoch': epoch + 1,
                        'state_dict': netG.state_dict()},
                       'model/netG_{}.pth'.format(epoch + 1))
            torch.save({'epoch': epoch + 1,
                        'state_dict': netD.state_dict()},
                       'model/netD_{}.pth'.format(epoch + 1))


# 局部对抗损失
def local_con_loss():
    pass


# 全局对抗损失
def globle_con_loss():
    pass


if __name__ == "__main__":
    train()
