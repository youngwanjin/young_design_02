from __future__ import print_function
import argparse
import torch.nn as nn
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import numpy as np
from model import Net_G


#  设置网络参数
def set_parameter():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='data_set/test', help='训练数据保存路径')
    parser.add_argument('--data_set', default='output', help='数据集文件名')
    parser.add_argument('--workers', type=int, default=0, help='工作线程数')
    parser.add_argument('--batchSize', type=int, default=64, help='输入图片批量大小')
    parser.add_argument('--imageSize', type=int, default=128, help='图片尺寸')
    parser.add_argument('--nz', type=int, default=100, help='初始噪音向量的大小')
    parser.add_argument('--ngf', type=int, default=64, help='生成网络中基础feature数目')
    parser.add_argument('--ndf', type=int, default=64, help='判别网络中基础feature数目')
    parser.add_argument('--nef', type=int, default=64, help='第一个转换层中的编码器滤波器')
    parser.add_argument('--nc', type=int, default=3, help='彩色图片通道数')
    parser.add_argument('--niter', type=int, default=50, help='网络训练过程中epoch数目')
    parser.add_argument('--lr', type=float, default=0.0002, help='初始学习率')
    parser.add_argument('--beta1', type=float, default=0.5, help='使用Adam优化算法中的β1参数值')
    parser.add_argument('--cuda', default=False, help='是否可以使用CUDA训练')
    parser.add_argument('--ngpu', type=int, default=1, help='使用GPU的数量')
    # model/netG_model.pth 使用模型继续训练
    parser.add_argument('--netG', default='model/netG_40.pth', help="模型G的保存路径")
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


# 开始训练
def test():
    # 获取参数
    opt = set_parameter()
    # 加载数据
    dataloader = load_data_set()
    # 加载网络
    netG = Net_G(opt)
    netG.load_state_dict(torch.load(opt.netG, map_location=lambda storage, location: storage)['state_dict'])
    netG.eval()
    # 数据转化为 tensor
    input_real = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)
    input_cropped = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)
    real_center = torch.FloatTensor(opt.batchSize, 3, int(opt.imageSize / 2), int(opt.imageSize / 2))
    # 定义损失
    criterionMSE = nn.MSELoss()   # 均方损失函数
    # 判断是否可用 GPU
    if opt.cuda:
        netG.cuda()
        input_real = input_real.cuda()
        input_cropped = input_cropped.cuda()
        criterionMSE.cuda()
        real_center = real_center.cuda()

    # 数据转化为 Variable
    input_real = Variable(input_real)
    input_cropped = Variable(input_cropped)
    real_center = Variable(real_center)

    dataiter = iter(dataloader)
    real_cpu, _ = dataiter.next()

    input_real.data.resize_(real_cpu.size()).copy_(real_cpu)
    input_cropped.data.resize_(real_cpu.size()).copy_(real_cpu)
    real_center_cpu = real_cpu[:, :, 32:96, 32:96]
    real_center.data.resize_(real_center_cpu.size()).copy_(real_center_cpu)

    input_cropped.data[:, 0, 34:94, 34:94] = 2 * 117.0 / 255.0 - 1.0
    input_cropped.data[:, 1, 34:94, 34:94] = 2 * 104.0 / 255.0 - 1.0
    input_cropped.data[:, 2, 34:94, 34:94] = 2 * 123.0 / 255.0 - 1.0

    vutils.save_image(input_cropped.data, 'result/test.png')
    # 使用网络G修复图片
    fake = netG(input_cropped)
    # 修复损失
    # errG = criterionMSE(fake, real_center)

    recon_image = input_cropped.clone()
    recon_image.data[:, :, 32:96, 32:96] = fake.data
    vutils.save_image(recon_image.data, 'result/test/recon_img.png', normalize=True)


if __name__ == "__main__":
    test()


