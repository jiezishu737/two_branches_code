import torch
from torch import nn
from torch.utils import model_zoo


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.vgg = VGG()
        self.load_vgg()
        self.amp = BackEnd()
        self.dmp = BackEnd()
        
        self.branch1 = Branch1()
        self.branch2 = Branch2()
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=8)
        self.upsample2 = nn.UpsamplingBilinear2d(scale_factor=4)
        self.conv1 = BaseConv(32, 32, 1, activation=nn.ReLU(), use_bn=True)
        self.conv3 = BaseConv(32, 32, 3, activation=nn.ReLU(), use_bn=True)
        self.conv5 = BaseConv(64, 1, 7, activation=None, use_bn=False)
        self.conv4 = BaseConv(64, 1, 3, activation=None, use_bn=False)
        self.mp = nn.MaxPool2d(2, 2)
        
        self.w1 = nn.Parameter(torch.tensor(0.5))
        self.w2 = nn.Parameter(torch.tensor(0.5))

        self.conv_att = BaseConv(64, 1, 1, activation=nn.Sigmoid(), use_bn=True)
        self.conv_out = BaseConv(32, 1, 1, activation=None, use_bn=False)

    def forward(self, input):
        input1 = self.vgg(input)
        #amp_out = self.amp(*input1)
        branch1 = self.branch1(*input1)
        #print('branch1=',branch1.shape)
        branch2 = self.branch2(*input1)
        branch1_test = self.conv5(branch1)
        branch2_test = self.conv4(branch2)
        #print('branch2=',branch2.shape)
        atm_1 = self.conv_att(branch1)
        atm_2 = self.conv_att(branch2)
        branch_new_1 = atm_1*branch1
        branch_new_2 = atm_2*branch2
        branch_new_1 = self.conv5(branch_new_1)
        branch_new_1 = self.mp(branch_new_1)
        branch_new_2 = self.conv4(branch_new_2)
        
        dmp_out = self.w1/(self.w1 + self.w2)*branch_new_1 + self.w2/(self.w1 + self.w2)*branch_new_2 
        atm_1 = self.mp(atm_1)
        atm_1 = self.upsample(atm_1)
        atm_2 = self.upsample(atm_2)
        branch_new_1 = self.upsample(branch_new_1)
        branch_new_2 = self.upsample(branch_new_2)
        dmp_out = self.upsample(dmp_out)
        branch1_test = self.upsample2(branch1_test)
        branch2_test = self.upsample(branch2_test)
        #dmp_out = self.conv1(dmp_out)
        #dmp_out = self.conv3(dmp_out)
        #dmp_out = self.upsample(dmp_out)
        #dmp_out = self.conv1(dmp_out)
        #dmp_out = self.conv3(dmp_out)
        #print('dmp_out=',dmp_out.shape)
        #amp_out = self.conv_att(amp_out)
        #print('amp_out=',amp_out.shape)
        #dmp_out = amp_out * dmp_out
        #dmp_out = self.conv_out(dmp_out)

        return dmp_out, atm_1, atm_2,self.w1,self.w2,branch_new_1,branch_new_2,branch1_test,branch2_test

    def load_vgg(self):
        #state_dict = model_zoo.load_url('https://download.pytorch.org/models/vgg16_bn-6c64b313.pth')
        state_dict = torch.load('/home/share/wangyongjie/new_vgg10/attention_map/vgg16_bn-6c64b313.pth')
        old_name = [0, 1, 3, 4, 7, 8, 10, 11, 14, 15, 17, 18, 20, 21, 24, 25, 27, 28, 30, 31]
        new_name = ['1_1', '1_2', '2_1', '2_2', '3_1', '3_2', '3_3', '4_1', '4_2', '4_3']
        new_dict = {}
        for i in range(10):
            new_dict['conv' + new_name[i] + '.conv.weight'] = \
                state_dict['features.' + str(old_name[2 * i]) + '.weight']
            new_dict['conv' + new_name[i] + '.conv.bias'] = \
                state_dict['features.' + str(old_name[2 * i]) + '.bias']
            new_dict['conv' + new_name[i] + '.bn.weight'] = \
                state_dict['features.' + str(old_name[2 * i + 1]) + '.weight']
            new_dict['conv' + new_name[i] + '.bn.bias'] = \
                state_dict['features.' + str(old_name[2 * i + 1]) + '.bias']
            new_dict['conv' + new_name[i] + '.bn.running_mean'] = \
                state_dict['features.' + str(old_name[2 * i + 1]) + '.running_mean']
            new_dict['conv' + new_name[i] + '.bn.running_var'] = \
                state_dict['features.' + str(old_name[2 * i + 1]) + '.running_var']

        self.vgg.load_state_dict(new_dict)


class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1_1 = BaseConv(3, 64, 3, activation=nn.ReLU(), use_bn=True)
        self.conv1_2 = BaseConv(64, 64, 3, activation=nn.ReLU(), use_bn=True)
        self.conv2_1 = BaseConv(64, 128, 3, activation=nn.ReLU(), use_bn=True)
        self.conv2_2 = BaseConv(128, 128, 3, activation=nn.ReLU(), use_bn=True)
        self.conv3_1 = BaseConv(128, 256, 3, activation=nn.ReLU(), use_bn=True)
        self.conv3_2 = BaseConv(256, 256, 3, activation=nn.ReLU(), use_bn=True)
        self.conv3_3 = BaseConv(256, 256, 3, activation=nn.ReLU(), use_bn=True)
        self.conv4_1 = BaseConv(256, 512, 3, activation=nn.ReLU(), use_bn=True)
        self.conv4_2 = BaseConv(512, 512, 3, activation=nn.ReLU(), use_bn=True)
        self.conv4_3 = BaseConv(512, 512, 3, activation=nn.ReLU(), use_bn=True)
        #self.conv5_1 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        #self.conv5_2 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        #self.conv5_3 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)

    def forward(self, input):
        input = self.conv1_1(input)
        input = self.conv1_2(input)
        input = self.pool(input)
        input = self.conv2_1(input)
        conv2_2 = self.conv2_2(input)

        input = self.pool(conv2_2)
        input = self.conv3_1(input)
        input = self.conv3_2(input)
        conv3_3 = self.conv3_3(input)

        input = self.pool(conv3_3)
        input = self.conv4_1(input)
        input = self.conv4_2(input)
        conv4_3 = self.conv4_3(input)

        #input = self.pool(conv4_3)
        #input = self.conv5_1(input)
        #input = self.conv5_2(input)
        #conv5_3 = self.conv5_3(input)

        return conv2_2, conv3_3, conv4_3


class BackEnd(nn.Module):
    def __init__(self):
        super(BackEnd, self).__init__()
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)

        self.conv1 = BaseConv(1024, 256, 1, activation=nn.ReLU(), use_bn=True)
        self.conv2 = BaseConv(256, 256, 3, activation=nn.ReLU(), use_bn=True)

        self.conv3 = BaseConv(768, 128, 1, activation=nn.ReLU(), use_bn=True)
        self.conv4 = BaseConv(128, 128, 3, activation=nn.ReLU(), use_bn=True)

        self.conv5 = BaseConv(256, 64, 1, activation=nn.ReLU(), use_bn=True)
        self.conv6 = BaseConv(64, 64, 3, activation=nn.ReLU(), use_bn=True)
        self.conv7 = BaseConv(64, 32, 3, activation=nn.ReLU(), use_bn=True)

    def forward(self, *input):
        conv2_2, conv3_3, conv4_3 = input

        #input = self.upsample(conv5_3)

        #input = torch.cat([input, conv4_3], 1)
        #input = self.conv1(input)
        #input = self.conv2(input)
        #input = self.upsample(input)
        
        input = self.upsample(conv4_3)
        
        input = torch.cat([input, conv3_3], 1)
        input = self.conv3(input)
        input = self.conv4(input)
        input = self.upsample(input)

        input = torch.cat([input, conv2_2], 1)
        input = self.conv5(input)
        input = self.conv6(input)
        input = self.conv7(input)

        return input
        
        
class Branch1(nn.Module):
    def __init__(self):
        super(Branch1, self).__init__()
        self.dc = BaseDeconv(512, 512, activation=nn.ReLU(), use_bn=True)
        self.conv1 = BaseConv(512, 512, 1, activation=nn.ReLU(), use_bn=True)
        self.conv2 = BaseConv(512, 256, 9, activation=nn.ReLU(), use_bn=True)
        self.conv3 = BaseConv(256, 128, 7, activation=nn.ReLU(), use_bn=True)
        self.conv4 = BaseConv(128, 64, 7, activation=nn.ReLU(), use_bn=True)
        #self.conv5 = BaseConv(64, 32, 7, activation=nn.ReLU(), use_bn=True)
        #self.mp = nn.MaxPool2d(2, 2)

    def forward(self, *input):
        conv2_2, conv3_3, conv4_3 = input
        input = self.dc(conv4_3)
        input = self.conv1(input)
        input = self.conv2(input)
        input = self.conv3(input)
        input = self.conv4(input)
        #input = self.conv5(input)
        #input = self.mp(input)
        return input
        
        
class Branch2(nn.Module):
    def __init__(self):
        super(Branch2, self).__init__()
        self.conv1 = BaseConv(512, 256, 5, activation=nn.ReLU(), use_bn=True)
        self.conv2 = BaseConv(256, 128, 3, activation=nn.ReLU(), use_bn=True)
        self.conv3 = BaseConv(128, 64, 3, activation=nn.ReLU(), use_bn=True)
        #self.conv4 = BaseConv(64, 32, 3, activation=nn.ReLU(), use_bn=True)

    def forward(self, *input):
        conv2_2, conv3_3, conv4_3 = input
        input = self.conv1(conv4_3)
        input = self.conv2(input)
        input = self.conv3(input)
        #input = self.conv4(input)
        return input    
        
        
class BaseDeconv(nn.Module):
    def __init__(self, in_channels, out_channels, activation=None, use_bn=False):
        super(BaseDeconv, self).__init__()
        self.use_bn = use_bn
        self.activation = activation
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, 2, 2)
        self.deconv.weight.data.normal_(0, 0.01)
        self.deconv.bias.data.zero_()
        self.bn = nn.BatchNorm2d(out_channels)
        self.bn.weight.data.fill_(1)
        self.bn.bias.data.zero_()

    def forward(self, input):
        input = self.deconv(input)
        if self.use_bn:
            input = self.bn(input)
        if self.activation:
            input = self.activation(input)

        return input
 
class BaseConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride=1, activation=None, use_bn=False):
        super(BaseConv, self).__init__()
        self.use_bn = use_bn
        self.activation = activation
        self.conv = nn.Conv2d(in_channels, out_channels, kernel, stride, kernel // 2)
        self.conv.weight.data.normal_(0, 0.01)
        self.conv.bias.data.zero_()
        self.bn = nn.BatchNorm2d(out_channels)
        self.bn.weight.data.fill_(1)
        self.bn.bias.data.zero_()

    def forward(self, input):
        input = self.conv(input)
        if self.use_bn:
            input = self.bn(input)
        if self.activation:
            input = self.activation(input)

        return input
#class BaseConv(nn.Module):
#    def __init__(self, in_channels, out_channels, kernel, stride=1, activation=None, use_bn=False):
#        super(BaseConv, self).__init__()
#        self.use_bn = use_bn
#        self.activation = activation
#        self.conv = nn.Conv2d(in_channels, out_channels, kernel, stride, kernel // 2)
#        self.conv.weight.data.normal_(0, 0.01)
#        self.conv.bias.data.zero_()
#        self.bn = nn.BatchNorm2d(out_channels)
#        self.bn.weight.data.fill_(1)
#        self.bn.bias.data.zero_()
#
#    def forward(self, input):
#        input = self.conv(input)
#        if self.use_bn:
#            input = self.bn(input)
#        if self.activation:
#            input = self.activation(input)
#
#        return input


if __name__ == '__main__':
    input = torch.randn(8, 3, 400, 400).cuda()
    model = Model().cuda()
    output, attention = model(input)
    print(input.size())
    print(output.size())
    print(attention.size())
