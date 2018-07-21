import mxnet as mx
import numpy as np

def Relu6(data):
    return mx.sym.minimum(mx.sym.maximum(data, 0), 6)

def permutation(data, groups):
	data = mx.sym.reshape(data, shape=(0, -4, groups, -1, -2))
	data = mx.sym.swapaxes(data, 1, 2)
	data = mx.sym.reshape(data, shape=(0, -3, -2))
	return data

def ConvNolinear(data, num_filter, kernel, stride, pad, group, use_global_stats, name):
    conv = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=kernel, stride=stride, pad=pad, num_group=group, no_bias=True, name=name)
    bn = mx.sym.BatchNorm(data=conv, fix_gamma=False, use_global_stats=use_global_stats,momentum=0.9,cudnn_off=True)
    relu6 = Relu6(bn)
    return relu6
    #return bn

def ConvDepthwise(data, num_filter, kernel, stride, pad, use_global_stats, name):
    conv = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=kernel, stride=stride, pad=pad, num_group=num_filter, no_bias=True, name=name)
    bn = mx.sym.BatchNorm(data=conv, fix_gamma=False, use_global_stats=use_global_stats,momentum=0.9,cudnn_off=True)
    relu6 = Relu6(bn)
    return relu6

def ConvLinear(data, num_filter, kernel, stride, pad, name, group=1, use_global_stats=False):
#def ConvLinear(data, num_filter, kernel, stride, pad, use_global_stats=False):
    conv = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=kernel, stride=stride, pad=pad, num_group=group, no_bias=True, name=name)
    bn = mx.sym.BatchNorm(data=conv, fix_gamma=False, use_global_stats=use_global_stats,momentum=0.9,cudnn_off=True)
    return bn

def GroupConv(data, in_c, out_c, kernel, stride, pad, name, use_global_stats, M=2, linear=True,cudnn_off=True):
    #conv_0 = mx.sym.Convolution(data=data, num_filter=M*in_c, kernel=(1,1), stride=(1,1), pad=(0,0), num_group=1, no_bias=True, name=name+'_gc_0')
    conv_1 = mx.sym.Convolution(data=data, num_filter=in_c, kernel=kernel, stride=stride, pad=pad, num_group=in_c, no_bias=True, name=name+'_gc_1')
    conv_2 = mx.sym.Convolution(data=conv_1, num_filter=out_c, kernel=(1,1), stride=(1,1), pad=(0,0), num_group=1, no_bias=True, name=name+'_gc_2')
    #sf_data_1 = channel_shuffle(conv_2,M)
    bn = mx.sym.BatchNorm(data=conv_2, fix_gamma=False, use_global_stats=use_global_stats,momentum=0.9,cudnn_off=True)
    if linear:
        return bn
    else:
        relu6 = Relu6(bn)
        return relu6

def IGC1x1(data, in_c, out_c, name, L=8, M=8, use_global_stats=False,linear=False):
    conv_1 = ConvLinear(data, num_filter=in_c, kernel=(1,1), stride=(1,1), pad=(0,0), group=L, use_global_stats=use_global_stats,name=name+'_ln_1')
    #conv_1 = GroupConv(data, in_c=in_c, out_c=in_c, kernel=(1,1), stride=(1,1), pad=(0,0), M=M, name=name+'_ln_1', use_global_stats=use_global_stats)
    sf_data_1 = permutation(conv_1,M)
    conv_2 = ConvNolinear(sf_data_1, num_filter=out_c, kernel=(1,1), stride=(1,1), pad=(0,0), group=M, use_global_stats=use_global_stats,name=name+'_nl_1')
    sf_data_2 = permutation(conv_2,out_c/M)
    return sf_data_2

def bottleneck(data, in_c, out_c, t, L, M, s, use_global_stats, name):
    if s == 1:
        conv_nolinear = ConvNolinear(data, num_filter=t*in_c, kernel=(1,1), stride=(1,1), pad=(0,0), group=L, use_global_stats=use_global_stats, name=name+'nl_1')
        conv_dw = ConvDepthwise(conv_nolinear, num_filter=t*in_c, kernel=(3,3), stride=(1,1), pad=(1,1), use_global_stats=use_global_stats, name=name+'_dw')
        sf_data_1 = permutation(conv_dw,M)
        conv_linear = ConvLinear(sf_data_1, num_filter=out_c, kernel=(1,1), stride=(1,1), pad=(0,0), group=M, use_global_stats=use_global_stats, name=name+'_ln_1')
        #conv_linear_1 = GroupConv(sf_data, in_c=in_c*t, out_c=in_c*t/M, kernel=(1,1), stride=(1,1), pad=(0,0), M=M, name=name+'_ln_1', use_global_stats=use_global_stats)
        sf_data_2 = permutation(conv_linear,out_c/M)

        if in_c == out_c:
            shortcut = data
        else:
            #shortcut = mx.sym.Convolution(data, num_filter=out_c, kernel=(1,1), stride=(1,1), pad=(0,0), no_bias=True)
            #shortcut = mx.sym.BatchNorm(shortcut, fix_gamma=False, use_global_stats=use_global_stats,momentum=0.9)
            #shortcut = Relu6(shortcut)
            shortcut = IGC1x1(data,in_c,out_c,name=name+'_igc',L=4,M=4)
        if True:
            shortcut._set_attr(mirror_stage='True')
        return shortcut + sf_data_2

    if s == 2:
        conv_nolinear = ConvNolinear(data, num_filter=t*in_c, kernel=(1,1), stride=(1,1), pad=(0,0), group=L, use_global_stats=use_global_stats, name=name+'nl_1')
        conv_dw = ConvDepthwise(conv_nolinear, num_filter=t*in_c, kernel=(3,3), stride=(1,1), pad=(1,1), use_global_stats=use_global_stats, name=name+'_dw')
        sf_data_1 = permutation(conv_dw,M)
        conv_linear = ConvLinear(sf_data_1, num_filter=out_c, kernel=(1,1), stride=(2,2), pad=(0,0), group=M, use_global_stats=use_global_stats, name=name+'_ln_1')
        #conv_linear_1 = GroupConv(sf_data, in_c=in_c*t, out_c=in_c*t/M, kernel=(1,1), stride=(1,1), pad=(0,0), M=M, name=name+'_ln_1', use_global_stats=use_global_stats)
        sf_data_2 = permutation(conv_linear,out_c/M)
        return sf_data_2

def get_fixed_param(n,c_L,t_L,M=2):
    fixed_param_name = []
    fixed_weigths = {}
    for i in xrange(n):
        name_igc_1 = 'blk%d_nl_1_gc_2_weight' % i
        fixed_param_name.append(name_igc_1)
        t = t_L[i]/M

        w_igc_1 = np.zeros((t*c_L[i],M*t*c_L[i],1,1))
        for j in xrange(t*c_L[i]):
            w_igc_1[j,M*j:M*(j+1),:,:] = 1.

        """
        w_igc_2_0 = np.zeros((M*c_L[i+1],c_L[i+1],1,1))
        for j in xrange(c_L[i+1]):
            w_igc_2_0[M*j:M*(j+1),j,:,:] = 1.
        w_igc_2 = np.zeros((c_L[i+1],M*c_L[i+1],1,1))
        for j in xrange(c_L[i+1]):
            ind = j
            if j%2 == 0:
                ind = j*2
            else:
                ind = 2*j-1
            w_igc_2[j,ind,:,:] = 1.
            w_igc_2[j,ind+M,:,:] = 1.
        """

        fixed_weigths[name_igc_1] = w_igc_1

    return fixed_param_name,fixed_weigths


def get_symbol(num_classes, **kwargs):
    expansion_factor=6
    use_global_stats = False
    s = [2,1,2,2,2,1,2,1]

    L =  [4,  2,  2,  2,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4]
    M =  [4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4]  

    c_L = [32, 32, 48, 48, 64, 64, 64, 128, 128, 128, 128, 192, 192, 192, 320, 320, 320, 640, 1280]

    t_L = [1]
    for i in xrange(1,17):
        t_L.append(expansion_factor)
        if len(L) == 1:
            L.append(L[0])
            M.append(M[0])
    repeat = [1, 2, 3, 4, 3, 3, 1]

    data = mx.sym.var(name='data')
    conv0 = mx.sym.Convolution(data=data, num_filter=32, kernel=(3,3), stride=(s[0],s[0]), pad=(1,1), name='conv0',no_bias=True)
    bn0 = mx.sym.BatchNorm(data=conv0, fix_gamma=False, use_global_stats=use_global_stats, name='bn0',momentum=0.9,cudnn_off=True)
    relu0 = Relu6(bn0)

    inp = relu0
    layer = 0

    for r in xrange(len(repeat)):
        n = repeat[r]
        bottleneck_0 = bottleneck(inp, in_c=c_L[layer], out_c=c_L[layer+1], t=t_L[layer], L= L[layer], M=M[layer], s=s[r+1], use_global_stats=use_global_stats, name='blk%d' % layer)
        inp = bottleneck_0
        layer += 1
        for i in xrange(1,n):
            bottleneck_1 = bottleneck(inp, in_c=c_L[layer], out_c=c_L[layer+1], t=t_L[layer], L= L[layer], M=M[layer], s=1, use_global_stats=use_global_stats, name='blk%d' % layer)
            inp = bottleneck_1
            layer += 1


    final_conv = IGC1x1(data=inp,in_c=c_L[17],out_c=c_L[18],name='final_conv',L=4,M=4)


    global_pooling = mx.sym.Pooling(data=final_conv, kernel=(7,7), pool_type='avg', global_pool=True)
    flatten = mx.sym.flatten(data=global_pooling)

    fc = mx.sym.FullyConnected(data=flatten, num_hidden=num_classes, no_bias=False)
    softmax = mx.sym.SoftmaxOutput(data=fc, name='softmax')

    #fixed_param_name,fixed_weigths = get_fixed_param(17,c_L,t_L,M=M)

    #return softmax,fixed_param_name,fixed_weigths

    return softmax

if __name__ == '__main__':
    L =  [4,  2,  4,  2,  4,  4,  4,  8,  8,  8,  4,  8,   8,   4,  10,  10,  10]
    M =  [4,  4,  6,  4,  8,  8,  8,  8,  8,  8,  8,  12,  12,  8,  16,  16,  16]
    net = get_symbol(1000,4,L=L,M=M)
    mod = mx.mod.Module(context=mx.cpu(), symbol=net)
    mod.bind(data_shapes=[('data', (1,3,224,224))])
    mod.init_params()
    mod.save_checkpoint('./igc_mn_v2', 0)
    print net.infer_shape(data=(1,3,224,224))