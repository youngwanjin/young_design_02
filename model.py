import torch.nn as nn
import torch


# 生成网络结构
class Net_G(nn.Module):
    def __init__(self, opt):
        super(Net_G, self).__init__()
        self.ngpu = opt.ngpu
        self.main = nn.Sequential(
            # o = (i - k + 2p)/s + 1
            # 1.input:3 output:64  input_imgSize:128 x 128
            nn.Conv2d(opt.nc, opt.nef, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # 2.input:64 output:64  input_imgSize:64 x 64
            nn.Conv2d(opt.nef, opt.nef, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.nef),
            nn.LeakyReLU(0.2, inplace=True),
            # 3.input:64 output:128  input_imgSize:32 x 32
            nn.Conv2d(opt.nef, opt.nef*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.nef*2),
            nn.LeakyReLU(0.2, inplace=True),
            # 4.input:128 output:256  input_imgSize:16 x 16
            nn.Conv2d(opt.nef*2, opt.nef*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.nef*4),
            nn.LeakyReLU(0.2, inplace=True),
            # 5.input:256 output:512  input_imgSize:8 x 8
            nn.Conv2d(opt.nef*4, opt.nef*8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.nef*8),
            nn.LeakyReLU(0.2, inplace=True),
            # 6.input:512 output:4000  input_imgSize:4 x 4
            nn.Conv2d(opt.nef*8, opt.nBottleneck, 4, bias=False),
            nn.BatchNorm2d(opt.nBottleneck),
            nn.LeakyReLU(0.2, inplace=True),

            # o = s(i - 1) + k -2p
            # 1.input:4000 output:512  input_imgSize:1 x 1
            nn.ConvTranspose2d(opt.nBottleneck, opt.ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(opt.ngf * 8),
            nn.ReLU(True),
            # 2.input:512 output:256  input_imgSize:4 x 4
            nn.ConvTranspose2d(opt.ngf * 8, opt.ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.ngf * 4),
            nn.ReLU(True),
            # 3.input:256 output:128  input_imgSize:8 x 8
            nn.ConvTranspose2d(opt.ngf * 4, opt.ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.ngf * 2),
            nn.ReLU(True),
            # 4.input:128 output:64  input_imgSize:16 x 16
            nn.ConvTranspose2d(opt.ngf * 2, opt.ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.ngf),
            nn.ReLU(True),
            # ===========================================================================
            # 4.input:64 output:3  input_imgSize:32 x 32
            # nn.ConvTranspose2d(opt.ngf, opt.ngf, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(opt.ngf),
            # nn.ReLU(True),
            # ===========================================================================
            nn.ConvTranspose2d(opt.ngf, opt.nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # output_imgSize:64 x 64
        )

    def forward(self, input):
        # 判断一个对象是否是一个已知的类型
        # if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
        #     output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        # else:
        #     output = self.main(input)
        # return output
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output


class Net_D(nn.Module):
    def __init__(self, opt):
        super(Net_D, self).__init__()
        self.ngpu = opt.ngpu
        self.main = nn.Sequential(
            # 1.input：3 output:64 imgSize: 64 x 64
            nn.Conv2d(opt.nc, opt.ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # 2.input：64 output:128 imgSize: 32 x 32
            nn.Conv2d(opt.ndf, opt.ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # 3.input：128 output:256 imgSize:16 x 16
            nn.Conv2d(opt.ndf * 2, opt.ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # 4.input：256 output:512 imgSize: 8 x 8
            nn.Conv2d(opt.ndf * 4, opt.ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # ===========================================================================
            # nn.Conv2d(opt.ndf * 8, opt.ndf * 16, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(opt.ndf * 16),
            # nn.LeakyReLU(0.2, inplace=True),
            # ===========================================================================
            # 1.input：512 output:1 imgSize:4 x 4
            nn.Conv2d(opt.ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        # if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
        #     output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        # else:
        #     output = self.main(input)
        # return output.view(-1, 1)
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output.view(-1, 1).squeeze(1)


# 判别 128 x 128
class NetD_Aux(nn.Module):
    def __init__(self, opt):
        super(NetD_Aux, self).__init__()
        self.ngpu = opt.ngpu
        self.main = nn.Sequential(
            # 1.input：3 output:64 imgSize: 64 x 64
            nn.Conv2d(opt.nc, opt.ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # 2.input：64 output:128 imgSize: 32 x 32
            nn.Conv2d(opt.ndf, opt.ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # 3.input：128 output:256 imgSize:16 x 16
            nn.Conv2d(opt.ndf * 2, opt.ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # 4.input：256 output:512 imgSize: 8 x 8
            nn.Conv2d(opt.ndf * 4, opt.ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # ===========================================================================
            nn.Conv2d(opt.ndf * 8, opt.ndf * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # ===========================================================================
            # 1.input：512 output:1 imgSize:4 x 4
            nn.Conv2d(opt.ndf * 16, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        # if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
        #     output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        # else:
        #     output = self.main(input)
        # return output.view(-1, 1)
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output.view(-1, 1).squeeze(1)
