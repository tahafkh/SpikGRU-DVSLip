"""
Implement custom layers
"""

import torch
import torch.nn as nn
import numpy as np

import torch.nn.functional as F
from DCLS.construct.modules import Dcls3_1d

Vth = 1.0
alpha_init_gru = 0.9
alpha_init_conv = 0.9
gamma = 10

class SpikeAct(torch.autograd.Function): 
    @staticmethod
    def forward(ctx, x_input):
        ctx.save_for_backward(x_input)
        output = torch.ge(x_input, Vth) 
        return output.float()

    @staticmethod
    def backward(ctx, grad_output):
        x_input, = ctx.saved_tensors 
        grad_input = grad_output.clone()
         ## derivative of arctan (scaled)
        grad_input = grad_input * 1 / (1 + gamma * (x_input - Vth)**2)
        return grad_input

class SpikeAct_signed(torch.autograd.Function): ## ternact
    @ staticmethod
    def forward(self, x):
        self.save_for_backward(x)
        x_forward = torch.clamp(torch.sign(x + Vth)+torch.sign(x - Vth), min=-1, max=1)
        return x_forward

    @ staticmethod
    def backward(self, grad_output):
        x_input, = self.saved_tensors
        grad_input = grad_output.clone()
        ## derivative of arctan (scaled)
        scale = 1 + 1/(1 + 4*Vth**2*gamma)
        grad_input = grad_input * 1/scale * (1/(1+ gamma * ((x_input - Vth)**2)) \
                                            + 1/(1+ gamma * ((x_input + Vth)**2))) 
        return grad_input
    
class Dcls3_1_SJ(Dcls3_1d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_count,
        learn_delay=True,
        stride=(1, 1),
        spatial_padding=(0, 0),
        dense_kernel_size=1,
        dilated_kernel_size=1,
        groups=1,
        bias=True,
        padding_mode='zeros',
        version='v1',
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_count,
            (*stride, 1),
            (*spatial_padding, 0),
            dense_kernel_size,
            dilated_kernel_size,
            groups,
            bias,
            padding_mode,  
            version,
        )
        self.learn_delay = learn_delay
        if not self.learn_delay:
            self.P.requies_grad = False
        else:
            torch.nn.init.constant_(self.P, (dilated_kernel_size[0] // 2)-0.01)
        if self.version == 'gauss':
            self.SIG.requires_grad = False
            self.sig_init = dilated_kernel_size[0]/2
            torch.nn.init.constant_(self.SIG, self.sig_init)
            
    def decrease_sig(self, epoch, epochs):
        if self.version == 'gauss':
            final_epoch = (1*epochs)//4
            final_sig = 0.23
            sig = self.SIG[0, 0, 0, 0, 0, 0].detach().cpu().item()
            alpha = (final_sig/self.sig_init)**(1/final_epoch)
            if epoch < final_epoch and sig > final_sig:
                self.SIG *= alpha

    def forward(self, x):
        x = x.permute(0, 2, 3, 4, 1) # [N, T, C, H, W] -> [N, C, H, W, T]
        x = F.pad(x, (self.dilated_kernel_size[0]-1, 0), mode='constant', value=0)
        x = super().forward(x)
        x = x.permute(0, 4, 1, 2, 3) # [N, C, H, W, T] -> [N, T, C, H, W]
        return x


class SCNNlayer(nn.Module):
    """ spiking 2D (or 3D if conv3d=True) convolution layer
        ann mode if ann=True
    """
    def __init__(self, args, height, width, in_channels, out_channels, kernel_size, dilation, stride, padding, useBN, ternact, conv3d=False, ann=False):
        super(SCNNlayer, self).__init__()
        self.conv3d = conv3d
        self.ann = ann
        self.height = height
        self.width = width
        if not self.ann:
            self.alpha = nn.Parameter(torch.zeros(out_channels).uniform_(alpha_init_conv,alpha_init_conv)) #1 per output channel only, as biases
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.stride = stride
        self.padding = padding
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.args = args
        if ternact:
            self.spikeact = SpikeAct_signed.apply
        else:
            self.spikeact = SpikeAct.apply
        self.useBN = useBN
        if self.conv3d:
            if self.useBN:
                self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=1, bias=False, padding_mode='zeros')
                self.bn = nn.BatchNorm3d(out_channels)
            else:
                self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=1, bias=True, padding_mode='zeros')
        else:
            if self.useBN:
                self.bn = nn.BatchNorm2d(out_channels)
                self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=1, bias=False, padding_mode='zeros')
            else:
                self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=1, bias=True, padding_mode='zeros')
            
        self.clamp()

        if self.ann:
            ## resnet init (kaiming normal mode fanout)
            n = out_channels * np.prod(kernel_size)
            nn.init.normal_(self.conv.weight, std= np.sqrt(2 / n))
            if not self.useBN:
                nn.init.zeros_(self.conv.bias)
        else:
            k = np.sqrt(6 / (in_channels*np.prod(kernel_size)))
            nn.init.uniform_(self.conv.weight, a=-k, b=k)

        if self.useBN:
            nn.init.constant_(self.bn.weight, 1)
            nn.init.constant_(self.bn.bias, 0)

    def forward(self, x):
        ## x : (B, T, Cin, Y, X)
        T = x.size(1)
        B = x.size(0)
        outputs = torch.zeros((B, T, self.out_channels, self.height, self.width), device = x.device)
        mem = torch.zeros((B, self.out_channels, self.height, self.width), device = x.device)
        output_prev = torch.zeros_like(mem)

        # parallel conv and batchnorm
        if self.conv3d:
            x = x.permute(0,2,1,3,4)
            conv_all = self.conv(x)
            if self.useBN:
                conv_all = self.bn(conv_all)
            conv_all = conv_all.permute(0,2,1,3,4)
        else:
            x = x.contiguous()
            x = x.view(-1, x.size(2), x.size(3), x.size(4)) #fuse T and B dim
            conv_all = self.conv(x)
            if self.useBN:
                conv_all = self.bn(conv_all)
            conv_all = conv_all.view(B, T, self.out_channels, self.height, self.width)

        if self.ann:
            conv_all = torch.relu(conv_all)
            outputs = conv_all
        else:
            for t in range(T):
                conv_xt = conv_all[:,t,:,:,:]

                ## SNN LIF
                mem = torch.einsum("abcd,b->abcd", mem, self.alpha) # with 1 time constant per output channel
                mem = mem + conv_xt - Vth * output_prev
                output_prev = self.spikeact(mem)
                outputs[:,t,:,:,:] = output_prev
               
        return outputs

    def clamp(self):
        if not self.ann:
            self.alpha.data.clamp_(0.,1.)

class DelayedConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, bias, delayed=True):
        super(DelayedConv, self).__init__()
        self.delayed = delayed
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, 
                                    stride=stride, padding=(kernel_size[0] // 2, kernel_size[0] // 2), bias=bias)
        if self.delayed:
            self.delay = Dcls3_1_SJ(
                in_channels=in_planes,
                out_channels=in_planes,
                kernel_count=1,
                learn_delay=True,
                spatial_padding=(1 // 2, 1 // 2),
                dense_kernel_size=1,
                dilated_kernel_size=(3, ),
                groups=in_planes,
                bias=False,
                version="v1",
            )
            torch.nn.init.constant_(self.delay.weight, 1)
            self.delay.weight.requies_grad = False

    def _get_dilated_factor(self):
        return self.delay.dilated_kernel_size[0] if self.delayed else 1

    def forward(self, x):
        if self.delayed:
            x = self.delay(x)
        
        T = x.size(1)
        B = x.size(0)
        x = x.reshape(-1, x.size(2), x.size(3), x.size(4))
        x = self.conv(x)
        return x.reshape(B, T, x.size(1), x.size(2), x.size(3))

    def clamp_parameters(self):
        if self.delayed:
            self.delay.clamp_parameters()

    def decrease_sig(self, epoch, epochs):
        if self.delayed:
            self.delay.decrease_sig(epoch, epochs)

class SBasicBlock(nn.Module):
    """ Spiking Resnet basic block
        ann mode if ann=True
    """ 
    def __init__(
            self,
            args,
            height,
            width,
            in_channels,
            out_channels,
            kernel_size, 
            dilation,
            stride, 
            padding, 
            useBN, 
            ternact, 
            ann=False, 
            delayed=False,
            axonal_delay=False,
        ):
        super(SBasicBlock, self).__init__()
        self.ann = ann
        self.height = height
        self.width = width
        if not self.ann:
            self.alpha1 = nn.Parameter(torch.zeros(out_channels).uniform_(alpha_init_conv, alpha_init_conv)) #1 per output channel only, as biases
            self.alpha2 = nn.Parameter(torch.zeros(out_channels).uniform_(alpha_init_conv, alpha_init_conv))
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.stride = stride
        self.padding = padding
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.args = args
        self.delayed = delayed
        if ternact:
            self.spikeact = SpikeAct_signed.apply
        else:
            self.spikeact = SpikeAct.apply
        self.useBN = useBN
        
        if self.useBN:
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.bn2 = nn.BatchNorm2d(out_channels)
            self.conv1 = self._add_conv_layer(
                delayed=delayed,
                axonal_delay=axonal_delay,
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False
            )
            self.conv2 = self._add_conv_layer(
                delayed=delayed,
                axonal_delay=axonal_delay,
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=(1, 1),
                padding=padding,
                bias=False
            )
        else:
            self.conv1 = self._add_conv_layer(
                delayed=delayed,
                axonal_delay=axonal_delay,
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=True
            )
            self.conv2 = self._add_conv_layer(
                delayed=delayed,
                axonal_delay=axonal_delay,
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=(1, 1),
                padding=padding,
                bias=True
            )

                
        if self.ann:
            ## resnet init (kaiming normal mode fanout)
            n = out_channels * np.prod(kernel_size)
            nn.init.normal_(self.conv1.conv.weight, std= np.sqrt(2.0 / n))
            nn.init.normal_(self.conv2.conv.weight, std= np.sqrt(2.0 / n))
            if not self.useBN:
                nn.init.zeros_(self.conv1.conv.bias)
                nn.init.zeros_(self.conv2.conv.bias)
        else:
            k1 = np.sqrt(6.0 /(self.in_channels*np.prod(self.kernel_size)*self._get_dilated_factor(self.conv1)))
            k2 = np.sqrt(6.0 /(self.out_channels*np.prod(self.kernel_size)*self._get_dilated_factor(self.conv2)))
            conv1_weights = self.conv1.conv if axonal_delay else self.conv1
            conv2_weights = self.conv2.conv if axonal_delay else self.conv2
            nn.init.uniform_(conv1_weights.weight, a=-k1, b=k1)
            nn.init.uniform_(conv2_weights.weight, a=-k2, b=k2)
            
        if self.useBN:
            nn.init.constant_(self.bn1.weight, 1)
            nn.init.constant_(self.bn1.bias, 0)
            nn.init.constant_(self.bn2.weight, 1)
            nn.init.constant_(self.bn2.bias, 0)

        if self.stride != (1,1):
            if self.useBN:
                self.downsample = self._add_conv_layer(
                    delayed=delayed,
                    axonal_delay=axonal_delay,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=(1, 1),
                    stride=stride,
                    padding=(0, 0),
                    bias=False
                )
                self.bn3 = nn.BatchNorm2d(out_channels)
                nn.init.constant_(self.bn3.weight, 1)
                nn.init.constant_(self.bn3.bias, 0)
            else:
                self.downsample = self._add_conv_layer(
                    delayed=delayed,
                    axonal_delay=axonal_delay,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=(1, 1),
                    stride=stride,
                    padding=(0, 0),
                    bias=True
                )
            if self.ann:
                # ## resnet init (kaiming normal mode fanout)
                n = out_channels
                nn.init.normal_(self.downsample.conv.weight, std= np.sqrt(2.0 / n))
                if not self.useBN:
                    nn.init.zeros_(self.downsample.conv.bias)
            else:
                k3 = np.sqrt(6.0 /(self.in_channels)*self._get_dilated_factor(self.downsample)) # kernel_size == (1,1)
                downsample_weights = self.downsample.conv if axonal_delay else self.downsample
                nn.init.uniform_(downsample_weights.weight, a=-k3, b=k3)

        self.clamp()

    def _get_dilated_factor(self, conv):
        return conv.dilated_kernel_size[0] if self.delayed else conv._get_dilated_factor()

    def _add_conv_layer(self, delayed, axonal_delay, in_channels, out_channels,
                        kernel_size, stride, padding, bias):
        if delayed:
            return Dcls3_1_SJ(in_channels=in_channels, out_channels=out_channels, kernel_count=1, learn_delay=True,
                              stride=stride, spatial_padding=padding, dense_kernel_size=kernel_size, dilated_kernel_size=(3, ),
                              groups=1, bias=bias, version='v1')
        else:
            return DelayedConv(in_channels, out_channels, kernel_size, stride, bias, delayed=axonal_delay)
        
    def _perform_conv(self, x, conv, bn=None, alpha=None, preidentity=None):
        T = x.size(1)
        B = x.size(0)
        outputs = torch.zeros(
            (B, T, self.out_channels, self.height, self.width), device=x.device
        )

        conv_all = conv(x)
        conv_all = conv_all.reshape(-1, conv_all.size(2), conv_all.size(3), conv_all.size(4))
        if bn is not None:
            conv_all = bn(conv_all)
        if preidentity is not None:
            preidentity = self.downsample(preidentity)
            preidentity = preidentity.reshape(-1, preidentity.size(2), preidentity.size(3), preidentity.size(4))
            if self.useBN:
                preidentity = self.bn3(preidentity)
            conv_all = conv_all + preidentity
        conv_all = conv_all.reshape(
            B, T, conv_all.size(1), conv_all.size(2), conv_all.size(3)
        )
        
        if self.ann:
            outputs = torch.relu(conv_all)
        else:
            mem = torch.zeros(
                (B, self.out_channels, self.height, self.width), device=x.device
            )
            output_prev = torch.zeros_like(mem)

            for t in range(T):
                ## SNN LIF
                mem = torch.einsum(
                    "abcd,b->abcd", mem, alpha
                )  # with 1 time constant per output channel #LIF NEURONS
                mem = mem + conv_all[:, t, :, :, :] - Vth * output_prev
                output_prev = self.spikeact(mem)
                outputs[:, t, :, :, :] = output_prev

        return outputs

    def forward(self, x):
        preidentity = x if self.stride != (1, 1) else None
        bn1 = self.bn1 if self.useBN else None
        alpha1 = self.alpha1 if not self.ann else None
        outputs1 = self._perform_conv(x, self.conv1, bn1, alpha1)

        bn2 = self.bn2 if self.useBN else None
        alpha2 = self.alpha2 if not self.ann else None
        outputs2 = self._perform_conv(outputs1, self.conv2, bn2, alpha2, preidentity)

        return outputs2, outputs1

    def decrease_sig(self, epoch, epochs):
        self.conv1.decrease_sig(epoch, epochs)
        self.conv2.decrease_sig(epoch, epochs)
        if self.stride != (1, 1):
            self.downsample.decrease_sig(epoch, epochs)

    def clamp(self):
        if not self.ann:
            self.alpha1.data.clamp_(0.,1.)
            self.alpha2.data.clamp_(0.,1.)
            
        self.conv1.clamp_parameters()
        self.conv2.clamp_parameters()
        if self.stride != (1, 1):
            self.downsample.clamp_parameters()


class SFCLayer(nn.Module):
    """ leaky integrator layer. if stateful=True, implement the stateful synapse version of the leaky integrator
        ann mode (=simple fully connected layer) if ann=True
    """
    def __init__(self, args, in_size, out_size, ann=False, stateful=False):
        super(SFCLayer, self).__init__()
        self.ann = ann
        self.in_size = in_size
        self.out_size = out_size
        self.dense = nn.Linear(in_size, out_size, bias=True)
        self.stateful = stateful
        if not self.ann:
            self.alpha = nn.Parameter(torch.zeros(out_size).uniform_(alpha_init_gru, alpha_init_gru))
        if stateful:
            self.beta = nn.Parameter(torch.zeros(out_size).uniform_(alpha_init_gru, alpha_init_gru))
        self.args = args

    def forward(self, x):
        # X : (B, T, N)
        T = x.size(1)
        B = x.size(0)
        outputs = torch.zeros((B, T, self.out_size), device = x.device)
        potential = torch.zeros((B, self.out_size), device = x.device)
        current = torch.zeros((B, self.out_size), device = x.device)

        if self.ann:
            outputs = self.dense(x)
        else:
            if self.stateful:
                for t in range(T):
                    out = self.dense(x[:,t,:])
                    current = self.beta * current + out
                    potential = self.alpha * potential + (1 - self.alpha) * current
                    outputs[:,t,:] = potential
            else: 
                for t in range(T):
                    out = self.dense(x[:,t,:])
                    potential = self.alpha * potential + out
                    outputs[:,t,:] = potential

        return outputs

    def clamp(self):
        if not self.ann:
            self.alpha.data.clamp_(0.,1.)




class GRUlayer(nn.Module):
    """ spiking GRU layer
        ann mode if ann=True
        SpikGRU2+ if twogates=True and ternact=True
    """
    def __init__(self, args, input_size, hidden_size, ann, ternact, twogates=False):
        super(GRUlayer, self).__init__()
        self.ann = ann
        self.twogates = twogates
        self.hidden_size = hidden_size
        self.wz = nn.Linear(input_size, hidden_size, bias=True)
        self.wi = nn.Linear(input_size, hidden_size, bias=True)
        self.uz = nn.Linear(hidden_size, hidden_size, bias=False)
        self.ui = nn.Linear(hidden_size, hidden_size, bias=True)
        if self.twogates:
            self.wr = nn.Linear(input_size, hidden_size, bias=True)
            self.ur = nn.Linear(hidden_size, hidden_size, bias=False)
        if not self.ann:
            self.alpha = nn.Parameter(torch.zeros(hidden_size).uniform_(alpha_init_gru, alpha_init_gru))
        self.clamp()
        self.args = args
        if ternact:
            self.spikeact = SpikeAct_signed.apply
        else:
            self.spikeact = SpikeAct.apply

        k_ff = np.sqrt(1./hidden_size)
        k_rec = np.sqrt(1./hidden_size)
        nn.init.uniform_(self.wi.weight, a=-k_ff, b=k_ff)
        nn.init.uniform_(self.wz.weight, a=-k_ff, b=k_ff)
        nn.init.uniform_(self.ui.weight, a=-k_rec, b=k_rec)
        nn.init.uniform_(self.uz.weight, a=-k_rec, b=k_rec)
        nn.init.uniform_(self.wi.bias, a=-k_ff, b=k_ff)
        nn.init.uniform_(self.wz.bias, a=-k_ff, b=k_ff)
        if self.twogates:
            nn.init.uniform_(self.wr.weight, a=-k_ff, b=k_ff)
            nn.init.uniform_(self.ur.weight, a=-k_rec, b=k_rec)
            nn.init.uniform_(self.wr.bias, a=-k_ff, b=k_ff)


    def forward(self, x):
        T = x.size(1)
        B = x.size(0)
        outputs = torch.zeros((B, T, self.hidden_size), device = x.device)
        output_prev = torch.zeros((B, self.hidden_size), device = x.device)
        temp = torch.zeros_like(output_prev)
        tempcurrent = torch.zeros_like(output_prev)

        for t in range(T): 
            
            tempZ = torch.sigmoid(self.wz(x[:,t,:]) + self.uz(output_prev)) 
            if self.twogates:
                tempR = torch.sigmoid(self.wr(x[:,t,:]) + self.ur(output_prev))
            if self.ann:
                if self.twogates:
                    tempcurrent = torch.tanh(self.wi(x[:,t,:]) + self.ui(output_prev) * tempR)
                else:
                    tempcurrent = torch.tanh(self.wi(x[:,t,:]) + self.ui(output_prev))
            else:
                if self.twogates:
                    tempcurrent = self.alpha * tempcurrent + self.wi(x[:,t,:]) + self.ui(output_prev) * tempR
                else:
                    tempcurrent = self.alpha * tempcurrent + self.wi(x[:,t,:]) + self.ui(output_prev)
                
            if self.ann:
                temp = tempZ * temp + (1 - tempZ) * tempcurrent
                output_prev = temp
            else:
                temp = tempZ * temp + (1 - tempZ) * tempcurrent - Vth * output_prev
                output_prev = self.spikeact(temp)

            outputs[:,t,:] = output_prev

        return outputs

    def clamp(self):
        if not self.ann:
            self.alpha.data.clamp_(0.,1.)



class SAdaptiveAvgPool2d(nn.Module):
    """ spiking adaptive avg pool 2d
        ann mode if ann=True
    """
    def __init__(self, args, kernel_size, channel_in, ternact, ann=False):
        super(SAdaptiveAvgPool2d, self).__init__()
        self.ann = ann
        self.avgpool = nn.AdaptiveAvgPool2d(kernel_size)
        self.kernel_size = kernel_size
        self.args = args
        if ternact:
            self.spikeact = SpikeAct_signed.apply
        else:
            self.spikeact = SpikeAct.apply

    def forward(self, x):
        # x: (B, T, Cin, Y, X)
        T = x.size(1)
        B = x.size(0)
        Cin = x.size(2)
        out = torch.zeros((B, T, Cin, self.kernel_size[0], self.kernel_size[0]), device = x.device)
        potential = torch.zeros((B, Cin, self.kernel_size[0], self.kernel_size[1]), device = x.device)
        output_prev = torch.zeros_like(potential)

        x = x.contiguous()
        x = x.view(-1, x.size(2), x.size(3), x.size(4)) #fuse T and B dim
        pool = self.avgpool(x)
        pool = pool.view(B, T, pool.size(1), pool.size(2), pool.size(3))

        if self.ann:
            out = pool
        else:
            for t in range(T):
                potential = potential + pool[:,t,:,:,:] - Vth * output_prev #IF neuron
                output_prev = self.spikeact(potential)
                out[:,t,:, :, :] = output_prev
        return out

class SAvgPool2d(nn.Module):
    """ spiking avg pool 2d
        ann mode if ann=True
    """
    def __init__(self, args, kernel, stride, padding, out_size, channel_in, ternact, ann=False):
        super(SAvgPool2d, self).__init__()
        self.ann = ann
        self.avgpool = nn.AvgPool2d(kernel, stride=stride, padding=padding)
        self.out_size = out_size
        self.args = args
        if ternact:
            self.spikeact = SpikeAct_signed.apply
        else:
            self.spikeact = SpikeAct.apply

    def forward(self, x):
        T = x.size(1)
        B = x.size(0)
        Cin = x.size(2)
        out = torch.zeros((B, T, Cin, self.out_size, self.out_size), device = x.device)
        potential = torch.zeros((B, Cin, self.out_size, self.out_size), device = x.device)
        output_prev = torch.zeros_like(potential)

        x = x.contiguous()
        x = x.view(-1, x.size(2), x.size(3), x.size(4)) #fuse T and B dim
        pool = self.avgpool(x)
        pool = pool.view(B, T, pool.size(1), pool.size(2), pool.size(3))

        if self.ann:
            out = pool
        else:
            for t in range(T):
                potential = potential + pool[:,t,:,:,:] - Vth * output_prev #IF neuron
                output_prev = self.spikeact(potential)
                out[:,t,:, :, :] = output_prev
        return out


class LiGRU(nn.Module):
    """ 3-layer bidrectionnal GRU backend
        ann mode if ann=True
    """
    def __init__(self, args, twogates, num_layers, bidirectional, dropout, input_size, hidden_size, ann, ternact):
        super(LiGRU, self).__init__()
        self.ann = ann
        self.hidden_size = hidden_size
        if ternact:
            self.spikeact = SpikeAct_signed.apply
        else:
            self.spikeact = SpikeAct.apply
        self.bidirectional = bidirectional
        self.args = args

        if num_layers != 3:
            print("Error in LiGRU: only defined with 3 layers")
        if self.bidirectional:
            self.grulayer1 = GRUlayer(args, input_size, hidden_size, ann=ann, ternact=ternact, twogates=twogates)
            self.grulayer2 = GRUlayer(args, hidden_size * 2, hidden_size, ann=ann, ternact=ternact, twogates=twogates)
            self.grulayer3 = GRUlayer(args, hidden_size * 2, hidden_size, ann=ann, ternact=ternact, twogates=twogates)
            self.grulayer1_b = GRUlayer(args, input_size, hidden_size, ann=ann, ternact=ternact, twogates=twogates)
            self.grulayer2_b = GRUlayer(args, hidden_size * 2, hidden_size, ann=ann, ternact=ternact, twogates=twogates)
            self.grulayer3_b = GRUlayer(args, hidden_size * 2, hidden_size, ann=ann, ternact=ternact, twogates=twogates)
        else:
            self.grulayer1 = GRUlayer(args, input_size, hidden_size, ann=ann, ternact=ternact, twogates=twogates)
            self.grulayer2 = GRUlayer(args, hidden_size, hidden_size, ann=ann, ternact=ternact, twogates=twogates)
            self.grulayer3 = GRUlayer(args, hidden_size, hidden_size, ann=ann, ternact=ternact, twogates=twogates)
            
        self.dropout = nn.Dropout(p=dropout)
        self.clamp()

    def forward(self, x):
        # x: [B, T, N]
        if self.bidirectional:
            x_b = torch.flip(x, [1])
            out1 = self.grulayer1(x)
            out1_b = self.grulayer1_b(x_b)
            out2 = self.grulayer2(self.dropout(torch.cat((out1, torch.flip(out1_b, [1])), 2)))
            out2_b = self.grulayer2_b(self.dropout(torch.cat((torch.flip(out1, [1]), out1_b), 2)))
            out3 = self.grulayer3(self.dropout(torch.cat((out2, torch.flip(out2_b, [1])), 2)))
            out3_b = self.grulayer3_b(self.dropout(torch.cat((torch.flip(out2, [1]), out2_b), 2)))
            outputs = torch.cat((out3, out3_b), 2)
        else:
            out1 = self.grulayer1(x)
            out2 = self.grulayer2(self.dropout(out1))
            out3 = self.grulayer3(self.dropout(out2))
            outputs = out3
        if self.bidirectional:
            return outputs, torch.cat((out2, out2_b), 2), torch.cat((out1, out1_b), 2)
        else:
            return outputs, out2, out1

    def clamp(self):
        if not self.ann:
            self.grulayer1.clamp()
            self.grulayer2.clamp()
            self.grulayer3.clamp()
            if self.bidirectional:
                self.grulayer1_b.clamp()
                self.grulayer2_b.clamp()
                self.grulayer3_b.clamp()