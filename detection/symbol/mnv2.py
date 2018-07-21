import mxnet as mx
import numpy as np

def Relu6(data):
    return mx.sym.minimum(mx.sym.maximum(data, 0), 6)

def channel_shuffle(data, groups):
	data = mx.sym.reshape(data, shape=(0, -4, groups, -1, -2))
	data = mx.sym.swapaxes(data, 1, 2)
	data = mx.sym.reshape(data, shape=(0, -3, -2))
	return data

def ConvNolinear(data, num_filter, kernel, stride, pad, use_global_stats, name):
    conv = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=kernel, stride=stride, pad=pad, no_bias=True, name=name+'_nl')
    bn = mx.sym.BatchNorm(data=conv, fix_gamma=False, use_global_stats=use_global_stats,cudnn_off=True)
    relu6 = Relu6(bn)
    return relu6

def ConvDepthwise(data, num_filter, kernel, stride, pad, use_global_stats, name):
    conv = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=kernel, stride=stride, pad=pad, num_group=num_filter, no_bias=True, name=name+'_dw')
    bn = mx.sym.BatchNorm(data=conv, fix_gamma=False, use_global_stats=use_global_stats,cudnn_off=True)
    relu6 = Relu6(bn)
    return relu6

def ConvLinear(data, num_filter, kernel, stride, pad, name, group=1, use_global_stats=False):
#def ConvLinear(data, num_filter, kernel, stride, pad, use_global_stats=False):
    conv = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=kernel, stride=stride, pad=pad, num_group=group, no_bias=True, name=name+'_ln')
    bn = mx.sym.BatchNorm(data=conv, fix_gamma=False, use_global_stats=use_global_stats,cudnn_off=True)
    return bn

def bottleneck(data, in_c, out_c, t, s, use_global_stats, name):
    if s == 1:
        conv_nolinear = ConvNolinear(data, num_filter=in_c*t, kernel=(1,1), stride=(1,1), pad=(0,0), use_global_stats=use_global_stats, name=name)
        conv_dw = ConvDepthwise(conv_nolinear, num_filter=t*in_c, kernel=(3,3), stride=(1,1), pad=(1,1), use_global_stats=use_global_stats, name=name)
        conv_linear = ConvLinear(conv_dw, num_filter=out_c, kernel=(1,1), stride=(1,1), pad=(0,0), group=1, use_global_stats=use_global_stats, name=name)

        if in_c == out_c:
            shortcut = data
        else:
            shortcut = mx.sym.Convolution(data, num_filter=out_c, kernel=(1,1), stride=(1,1), pad=(0,0), no_bias=True,name=name+"_skip")
            shortcut = mx.sym.BatchNorm(shortcut, fix_gamma=False, use_global_stats=use_global_stats,cudnn_off=True)
            shortcut = Relu6(shortcut)

        if True:
            shortcut._set_attr(mirror_stage='True')
        return shortcut + conv_linear

    if s == 2:
        conv_nolinear = ConvNolinear(data, num_filter=in_c*t, kernel=(1,1), stride=(1,1), pad=(0,0), use_global_stats=use_global_stats, name=name)
        conv_dw = ConvDepthwise(conv_nolinear, num_filter=in_c*t, kernel=(3,3), stride=(2,2), pad=(1,1), use_global_stats=use_global_stats, name=name)
        conv_linear = ConvLinear(conv_dw, num_filter=out_c, kernel=(1,1), stride=(1,1), pad=(0,0), group=1, use_global_stats=use_global_stats, name=name)
        return conv_linear

def get_fixed_param(n,c_L,t_L):
    fixed_param_name = []
    fixed_weigths = {}
    for i in xrange(n):
        name = 'blk%d_nl_weight' % i
        fixed_param_name.append(name)
        weights = np.zeros((t_L[i]*c_L[i],c_L[i],1,1))
        for j in xrange(c_L[i]):
            weights[t_L[i]*j:t_L[i]*(j+1),j,:,:] = 1.
        fixed_weigths[name] = weights
    return fixed_param_name,fixed_weigths


def get_symbol(num_classes, **kwargs):
    use_global_stats = False
    s1 = 2
    s2 = 2
    s3 = 2
    s4 = 2
    # if stride == 16:
    #     s1 = 1
    # if stride == 8:
    #     s1 = 1
    #     s2 = 1
    # if stride == 4:
    #     s1 = 1
    #     s2 = 1
    #     s4 = 1
    # if stride == 2:
    #     s1 = 1
    #     s2 = 1
    #     s3 = 1
    #     s4 = 1
    T = 6
    t_L = [1]
    for i in xrange(16):
        t_L.append(T)
    c_L = [32,16,24,24,32,32,32,64, 64, 64, 64, 96, 96, 96, 160,160,160,320,1280]
    #c_L = [32,32,48,48,64,64,64,128,128,128,128,192,192,192,320,320,320,640,1280]
    #c_L = [32,32,64,64,128,128,128,256,256,256,256,512,512,512,1024,1024,1024,1024,2048]
    data = mx.sym.var(name='data')
    conv0 = mx.sym.Convolution(data=data, num_filter=c_L[0], kernel=(3,3), stride=(2,2), pad=(1,1), name='conv0',no_bias=True)
    bn0 = mx.sym.BatchNorm(data=conv0, fix_gamma=False, use_global_stats=use_global_stats, name='bn0',cudnn_off=True)
    relu0 = Relu6(bn0)

    bottleneck0 = bottleneck(relu0, in_c=c_L[0], out_c=c_L[1], t=t_L[0], s=1, use_global_stats=use_global_stats, name='blk0')

    bottleneck1_0 = bottleneck(bottleneck0, in_c=c_L[1], out_c=c_L[2], t=t_L[1], s=2, use_global_stats=use_global_stats,name='blk1')
    bottleneck1_1 = bottleneck(bottleneck1_0, in_c=c_L[2], out_c=c_L[3], t=t_L[2], s=1, use_global_stats=use_global_stats,name='blk2')

    bottleneck2_0 = bottleneck(bottleneck1_1, in_c=c_L[3], out_c=c_L[4], t=t_L[3], s=2, use_global_stats=use_global_stats,name='blk3')
    bottleneck2_1 = bottleneck(bottleneck2_0, in_c=c_L[4], out_c=c_L[5], t=t_L[4], s=1, use_global_stats=use_global_stats,name='blk4')
    bottleneck2_2 = bottleneck(bottleneck2_1, in_c=c_L[5], out_c=c_L[6], t=t_L[5], s=1, use_global_stats=use_global_stats,name='blk5')

    bottleneck3_0 = bottleneck(bottleneck2_2, in_c=c_L[6], out_c=c_L[7], t=t_L[6], s=2, use_global_stats=use_global_stats,name='blk6')
    bottleneck3_1 = bottleneck(bottleneck3_0, in_c=c_L[7], out_c=c_L[8], t=t_L[7], s=1, use_global_stats=use_global_stats,name='blk7')
    bottleneck3_2 = bottleneck(bottleneck3_1, in_c=c_L[8], out_c=c_L[9], t=t_L[8], s=1, use_global_stats=use_global_stats,name='blk8')
    bottleneck3_3 = bottleneck(bottleneck3_2, in_c=c_L[9], out_c=c_L[10], t=t_L[9], s=1, use_global_stats=use_global_stats,name='blk9')

    bottleneck4_0 = bottleneck(bottleneck3_3, in_c=c_L[10], out_c=c_L[11], t=t_L[10], s=1, use_global_stats=use_global_stats,name='blk10')
    bottleneck4_1 = bottleneck(bottleneck4_0, in_c=c_L[11], out_c=c_L[12], t=t_L[11], s=1, use_global_stats=use_global_stats,name='blk11')
    bottleneck4_2 = bottleneck(bottleneck4_1, in_c=c_L[12], out_c=c_L[13], t=t_L[12], s=1, use_global_stats=use_global_stats,name='blk12')

    bottleneck5_0 = bottleneck(bottleneck4_2, in_c=c_L[13], out_c=c_L[14], t=t_L[13], s=2, use_global_stats=use_global_stats,name='blk13')
    bottleneck5_1 = bottleneck(bottleneck5_0, in_c=c_L[14], out_c=c_L[15], t=t_L[14], s=1, use_global_stats=use_global_stats,name='blk14')
    bottleneck5_2 = bottleneck(bottleneck5_1, in_c=c_L[15], out_c=c_L[16], t=t_L[15], s=1, use_global_stats=use_global_stats,name='blk15')

    bottleneck6_0 = bottleneck(bottleneck5_2, in_c=c_L[16], out_c=c_L[17], t=t_L[16], s=1, use_global_stats=use_global_stats,name='blk16')

    final_conv = mx.sym.Convolution(data=bottleneck6_0, num_filter=c_L[18], kernel=(1,1), stride=(1,1), pad=(0,0), num_group=1, no_bias=True,name='finalconv')
    final_bn = mx.sym.BatchNorm(data=final_conv, fix_gamma=False, use_global_stats=use_global_stats,cudnn_off=True)
    final_relu = Relu6(final_bn)

    global_pooling = mx.sym.Pooling(data=final_relu, kernel=(7,7), pool_type='avg', global_pool=True)
    flatten = mx.sym.flatten(data=global_pooling)
    #dropout = mx.sym.Dropout(data=flatten, p=0.2)
    fc = mx.sym.FullyConnected(data=flatten, num_hidden=num_classes, no_bias=False)
    softmax = mx.sym.SoftmaxOutput(data=fc, name='softmax')

    fixed_param_name,fixed_weigths = get_fixed_param(17,c_L[0:17],t_L)

    return softmax

if __name__ == '__main__':
    net = get_symbol(1000)
    mod = mx.mod.Module(context=mx.cpu(), symbol=net)
    mod.bind(data_shapes=[('data', (1,3,224,224))])
    mod.init_params()
    mod.save_checkpoint('./mobilenet_v2', 0)
    print net.infer_shape(data=(1,3,224,224))