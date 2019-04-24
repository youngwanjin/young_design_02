from __future__ import print_function
import argparse
import utils
import torch.nn as nn
import torch.utils.data
import torchvision.transforms as transforms
from torch.autograd import Variable
from model import Net_G


# 设置参数
def set_parameter():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_image', default='./data_set/test_one/out1.png', help='测试图片的名称')
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
    parser.add_argument('--netG', default='model/netG_200.pth', help="模型G的保存路径")
    parser.add_argument('--outf', default='.', help='当前路径')
    parser.add_argument('--manualSeed', type=int, help='随机数')
    parser.add_argument('--nBottleneck', type=int, default=4000, help='训练的最大瓶颈')
    parser.add_argument('--overlapPred', type=int, default=4, help='重叠的边缘')
    parser.add_argument('--wtl2', type=float, default=0.998, help='0 意味着不用否则使用该权重')
    parser.add_argument('--wtlD', type=float, default=0.001, help='0 means do not use else use with this weight')
    parser.add_argument('--overlapL2Weight ', type=int, default=10, help="L2的权重")
    opt = parser.parse_args()
    return opt


# 加载数据
def load_data_set():
    # 获取数据
    opt = set_parameter()
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    image = utils.load_image(opt.test_image, opt.imageSize)
    image = transform(image)
    image = image.repeat(1, 1, 1, 1)
    return image


# 测试修复一张图的结果
def test_one():
    # 获取参数
    opt = set_parameter()
    # 获取待修复的图片
    test_img = load_data_set()
    # 加载网络参数
    netG = Net_G(opt)
    netG.load_state_dict(torch.load(opt.netG, map_location=lambda storage, location: storage)['state_dict'])
    # netG.requires_grad = False
    netG.eval()
    # 设置tensor变量
    input_real = torch.FloatTensor(1, 3, opt.imageSize, opt.imageSize)
    input_cropped = torch.FloatTensor(1, 3, opt.imageSize, opt.imageSize)
    real_center = torch.FloatTensor(1, 3, int(opt.imageSize / 2), int(opt.imageSize / 2))
    # 定义损失函数
    criterionMSE = nn.MSELoss()
    # 使用CUDA
    if opt.cuda:
        netG.cuda()
        input_real = input_real.cuda()
        input_cropped = input_cropped.cuda()
        criterionMSE.cuda()
        real_center = real_center.cuda()

    input_real = Variable(input_real)
    input_cropped = Variable(input_cropped)
    real_center = Variable(real_center)

    input_real.data.resize_(test_img.size()).copy_(test_img)
    input_cropped.data.resize_(test_img.size()).copy_(test_img)
    real_center_cpu = test_img[:, :, 32:96, 32:96]
    real_center.data.resize_(real_center_cpu.size()).copy_(real_center_cpu)

    input_cropped.data[:, 0, 34:94, 34:94] = 2 * 117.0 / 255.0 - 1.0
    input_cropped.data[:, 1, 34:94, 34:94] = 2 * 104.0 / 255.0 - 1.0
    input_cropped.data[:, 2, 34:94, 34:94] = 2 * 123.0 / 255.0 - 1.0
    # 使用G网络修复图像
    fake = netG(input_cropped)

    recon_image = input_cropped.clone()
    recon_image.data[:, :, 32:96, 32:96] = fake.data

    utils.save_image('result/test_one/real_{}.png'.format(opt.test_image[-8:-3]), test_img[0])
    # utils.save_image('result/test_one/cropped.png', input_cropped.data[0])
    utils.save_image('result/test_one/recon_{}.png'.format(opt.test_image[-8:-3]), recon_image.data[0])

    # 计算修复损失
    errG = criterionMSE(recon_image, input_real)
    print('修复损失为：{:.4f}'.format(errG.item()))
    print('图像修复：Successful!')


if __name__ == "__main__":
    test_one()
