import copy
from torch.nn import init
import functools
from torch.optim import lr_scheduler
from ..losses.losses import *


###############################################################################
# Helper Functions
###############################################################################

def define_loss(type='CrossEntropyLoss', alpha=0, class_num=2):
    if type == 'CrossEntropyLoss':
        criterion = nn.CrossEntropyLoss()
    elif type == 'BCW':
        criterion = BCWLoss(class_num=class_num)
    elif type == 'Focal':
        criterion = FocalLoss(alpha=alpha)
    else:
        raise NotImplementedError('loss type [%s] is not recoginized' % type)
    return criterion


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=opt.lr_gamma)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
            elif init_type == 'xavier_uniform':
                init.xavier_uniform_(m.weight.data, gain=gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            if init_type == 'normal':
                init.normal_(m.weight.data, 1.0, gain)
                init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[], main_device=0):
    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available()), 'no available gpu devices'
        gpu_ids.pop(main_device)
        try:
            net = torch.nn.DataParallel(net, device_ids=[main_device] + gpu_ids)
            net.to(main_device)
        except:
            net = torch.nn.DataParallel(net)
            net.cuda()
    if init_type is not None:
        init_weights(net, init_type, gain=init_gain)
    return net


def define_net(input_nc, output_nc, net='unet', init_type='xavier_uniform', init_gain=1.0, gpu_ids=[], main_device=0):
    if net != 'unet':
        init_type = None
    if net == 'unet':
        net = Unet(input_nc=input_nc, output_nc=output_nc)
    else:
        net = Unet(input_nc=input_nc, output_nc=output_nc)
        # raise NotImplementedError('model name [%s] is not recoginized'%net)
    return init_net(net, init_type=init_type, init_gain=init_gain, gpu_ids=copy.deepcopy(gpu_ids),
                    main_device=main_device)


class Unet(nn.Module):
    def __init__(self, input_nc=1, output_nc=2):
        super(Unet, self).__init__()
        num_feat = [64, 128, 256, 512, 1024]
        # unet structure
        # innermost
        unet_block = UnetSkipConnectionBlock(num_feat[3], num_feat[4], innermost=True)

        unet_block = UnetSkipConnectionBlock(num_feat[2], num_feat[3], submodule=unet_block)
        unet_block = UnetSkipConnectionBlock(num_feat[1], num_feat[2], submodule=unet_block)
        unet_block = UnetSkipConnectionBlock(num_feat[0], num_feat[1], submodule=unet_block)
        # outermost
        unet_block = UnetSkipConnectionBlock(output_nc, num_feat[0], input_nc=input_nc,
                                             submodule=unet_block, outermost=True)
        self.model = unet_block

    def forward(self, input):
        return self.model(input)


class UnetSkipConnectionBlock(nn.Module):
    # define the submodule with skip connection
    # ----------------------------------------#
    # /-downsampling-/submodule/-upsampling-/ #
    # ----------------------------------------#
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if input_nc is None:
            input_nc = outer_nc
        down_maxpool = nn.MaxPool2d(kernel_size=2)
        conv1 = self.conv_3x3(input_nc, inner_nc)
        conv2 = self.conv_3x3(inner_nc * 2, inner_nc)
        up_conv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                     kernel_size=2,
                                     stride=2)
        conv_out = nn.Conv2d(inner_nc, outer_nc,
                             kernel_size=1)
        if outermost:
            down = [conv1]
            up = [conv2, conv_out]
        elif innermost:
            down = [down_maxpool, conv1]
            up = [up_conv]
        else:
            down = [down_maxpool, conv1]
            up = [conv2, up_conv]

        model = down + up if innermost else down + [submodule] + up
        self.model = nn.Sequential(*model)

    def forward(self, input):
        # print('input shape:',input.size())
        output = self.model(input)
        # print('output shape:',output.size())
        if not self.outermost:
            output = torch.cat([input, output], 1)
        return output

    def conv_3x3(self, input_nc, output_nc):
        conv_block1 = nn.Sequential(nn.Conv2d(input_nc, output_nc,
                                              kernel_size=3,
                                              stride=1,
                                              padding=1),
                                    nn.BatchNorm2d(output_nc),
                                    nn.ReLU(inplace=True))
        conv_block2 = nn.Sequential(nn.Conv2d(output_nc, output_nc,
                                              kernel_size=3,
                                              stride=1,
                                              padding=1),
                                    nn.BatchNorm2d(output_nc),
                                    nn.ReLU(inplace=True))
        conv_block = nn.Sequential(conv_block1, conv_block2)
        return conv_block
