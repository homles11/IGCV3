import mxnet as mx
import numpy as np

def Relu6(data):
    return mx.sym.minimum(mx.sym.maximum(data, 0), 6)

def channel_shuffle(data, groups):
	data = mx.sym.reshape(data, shape=(0, -4, groups, -1, -2))
	data = mx.sym.swapaxes(data, 1, 2)
	data = mx.sym.reshape(data, shape=(0, -3, -2))
	return data

def conv_act_layer(from_layer, name, num_filter, kernel=(1,1), pad=(0,0), \
    stride=(1,1), act_type="relu", use_batchnorm=False):
    """
    wrapper for a small Convolution group

    Parameters:
    ----------
    from_layer : mx.symbol
        continue on which layer
    name : str
        base name of the new layers
    num_filter : int
        how many filters to use in Convolution layer
    kernel : tuple (int, int)
        kernel size (h, w)
    pad : tuple (int, int)
        padding size (h, w)
    stride : tuple (int, int)
        stride size (h, w)
    act_type : str
        activation type, can be relu...
    use_batchnorm : bool
        whether to use batch normalization

    Returns:
    ----------
    (conv, relu) mx.Symbols
    """
    # bias = mx.symbol.Variable(name="{}_conv_bias".format(name),
    #     init=mx.init.Constant(0.0), attr={'__lr_mult__': '2.0'})
    conv = mx.symbol.Convolution(data=from_layer, kernel=kernel, pad=pad, \
        stride=stride, num_filter=num_filter, name="{}_conv".format(name), no_bias=True,num_group=2)
    conv = channel_shuffle(conv,2)
    if use_batchnorm:
        conv = mx.symbol.BatchNorm(data=conv, name="{}_bn".format(name),momentum=0.9,eps=0.001,cudnn_off=True)
    if act_type=="relu":
        conv = Relu6(data=conv)
    return conv

def conv_act_layer_dw(from_layer, name, kin=0,kout=0, \
    stride=(1,1), act_type="relu" ,use_batchnorm=False):
    """
    wrapper for a small Convolution group

    Parameters:
    ----------
    from_layer : mx.symbol
        continue on which layer
    name : str
        base name of the new layers
    num_filter : int
        how many filters to use in Convolution layer
    kernel : tuple (int, int)
        kernel size (h, w)
    pad : tuple (int, int)
        padding size (h, w)
    stride : tuple (int, int)
        stride size (h, w)
    act_type : str
        activation type, can be relu...
    use_batchnorm : bool
        whether to use batch normalization

    Returns:
    ----------
    (conv, relu) mx.Symbols
    """
    conv = mx.symbol.Convolution(data=from_layer, kernel=(3,3), pad=(1,1), \
        stride=stride, num_filter=kin, name="{}_conv_dw".format(name), no_bias=True, num_group=kin)
    conv = mx.symbol.BatchNorm(data=conv,name="{}_bn1".format(name),momentum=0.9,eps=0.001,cudnn_off=True)
    conv = mx.symbol.Convolution(data=conv, kernel=(1,1), pad=(0,0), \
                                 stride=(1,1), num_filter=kout, name="{}_conv_1x1".format(name),no_bias=True,num_group=2)
    conv = channel_shuffle(conv,kout/2)
    if use_batchnorm:
        conv = mx.symbol.BatchNorm(data=conv, name="{}_bn".format(name),momentum=0.9,eps=0.001,cudnn_off=True)
        conv = Relu6(data=conv)
    return conv

# def legacy_conv_act_layer(from_layer, name, num_filter, kernel=(1,1), pad=(0,0), \
#     stride=(1,1), act_type="relu", use_batchnorm=False):
#     """
#     wrapper for a small Convolution group
#
#     Parameters:
#     ----------
#     from_layer : mx.symbol
#         continue on which layer
#     name : str
#         base name of the new layers
#     num_filter : int
#         how many filters to use in Convolution layer
#     kernel : tuple (int, int)
#         kernel size (h, w)
#     pad : tuple (int, int)
#         padding size (h, w)
#     stride : tuple (int, int)
#         stride size (h, w)
#     act_type : str
#         activation type, can be relu...
#     use_batchnorm : bool
#         whether to use batch normalization
#
#     Returns:
#     ----------
#     (conv, relu) mx.Symbols
#     """
#     # assert not use_batchnorm, "batchnorm not yet supported"
#     bias = mx.symbol.Variable(name="conv{}_bias".format(name),
#         init=mx.init.Constant(0.0), attr={'__lr_mult__': '2.0'})
#     conv = mx.symbol.Convolution(data=from_layer, no_bias=True, kernel=kernel, pad=pad, \
#         stride=stride, num_filter=num_filter, name="conv{}".format(name))
#     relu = mx.symbol.Activation(data=conv, act_type=act_type, \
#         name="{}{}".format(act_type, name))
#     if use_batchnorm:
#         relu = mx.symbol.BatchNorm(data=relu, name="bn{}".format(name))
#     return conv, relu

def multi_layer_feature(body, from_layers, num_filters, strides, pads, min_filter=16):
    """Wrapper function to extract features from base network, attaching extra
    layers and SSD specific layers

    Parameters
    ----------
    from_layers : list of str
        feature extraction layers, use '' for add extra layers
        For example:
        from_layers = ['relu4_3', 'fc7', '', '', '', '']
        which means extract feature from relu4_3 and fc7, adding 4 extra layers
        on top of fc7
    num_filters : list of int
        number of filters for extra layers, you can use -1 for extracted features,
        however, if normalization and scale is applied, the number of filter for
        that layer must be provided.
        For example:
        num_filters = [512, -1, 512, 256, 256, 256]
    strides : list of int
        strides for the 3x3 convolution appended, -1 can be used for extracted
        feature layers
    pads : list of int
        paddings for the 3x3 convolution, -1 can be used for extracted layers
    min_filter : int
        minimum number of filters used in 1x1 convolution

    Returns
    -------
    list of mx.Symbols

    """
    # arguments check
    assert len(from_layers) > 0
    assert isinstance(from_layers[0], str) and len(from_layers[0].strip()) > 0
    assert len(from_layers) == len(num_filters) == len(strides) == len(pads)

    internals = body.get_internals()
    layers = []
    for k, params in enumerate(zip(from_layers, num_filters, strides, pads)):
        from_layer, num_filter, s, p = params
        if from_layer.strip():
            # extract from base network
            layer = internals[from_layer.strip() + '_output']
            layer = Relu6(data=layer)
            layers.append(layer)
        else:
            # attach from last feature layer
            assert len(layers) > 0
            assert num_filter > 0
            layer = layers[-1]
            num_1x1 = max(min_filter, num_filter // 2)
            conv_1x1 = conv_act_layer(layer, 'multi_feat_%d_conv_1x1' % (k),
                num_1x1, kernel=(1, 1), pad=(0, 0), stride=(1, 1), act_type='relu',use_batchnorm=True)
            conv_3x3 = conv_act_layer_dw(conv_1x1, 'multi_feat_%d_conv_3x3_dw' % (k),
                kin=num_1x1,kout=num_filter, stride=(s, s), act_type='relu',use_batchnorm=True)
            layers.append(conv_3x3)
    return layers

def multibox_layer(from_layers, num_classes, sizes=[.2, .95],
                    ratios=[1], normalization=-1, num_channels=[],
                    clip=False, interm_layer=0, steps=[]):
    """
    the basic aggregation module for SSD detection. Takes in multiple layers,
    generate multiple object detection targets by customized layers

    Parameters:
    ----------
    from_layers : list of mx.symbol
        generate multibox detection from layers
    num_classes : int
        number of classes excluding background, will automatically handle
        background in this function
    sizes : list or list of list
        [min_size, max_size] for all layers or [[], [], []...] for specific layers
    ratios : list or list of list
        [ratio1, ratio2...] for all layers or [[], [], ...] for specific layers
    normalizations : int or list of int
        use normalizations value for all layers or [...] for specific layers,
        -1 indicate no normalizations and scales
    num_channels : list of int
        number of input layer channels, used when normalization is enabled, the
        length of list should equals to number of normalization layers
    clip : bool
        whether to clip out-of-image boxes
    interm_layer : int
        if > 0, will add a intermediate Convolution layer
    steps : list
        specify steps for each MultiBoxPrior layer, leave empty, it will calculate
        according to layer dimensions

    Returns:
    ----------
    list of outputs, as [loc_preds, cls_preds, anchor_boxes]
    loc_preds : localization regression prediction
    cls_preds : classification prediction
    anchor_boxes : generated anchor boxes
    """
    assert len(from_layers) > 0, "from_layers must not be empty list"
    assert num_classes > 0, \
        "num_classes {} must be larger than 0".format(num_classes)

    assert len(ratios) > 0, "aspect ratios must not be empty list"
    if not isinstance(ratios[0], list):
        # provided only one ratio list, broadcast to all from_layers
        ratios = [ratios] * len(from_layers)
    assert len(ratios) == len(from_layers), \
        "ratios and from_layers must have same length"

    assert len(sizes) > 0, "sizes must not be empty list"
    if len(sizes) == 2 and not isinstance(sizes[0], list):
        # provided size range, we need to compute the sizes for each layer
         assert sizes[0] > 0 and sizes[0] < 1
         assert sizes[1] > 0 and sizes[1] < 1 and sizes[1] > sizes[0]
         tmp = np.linspace(sizes[0], sizes[1], num=(len(from_layers)-1))
         min_sizes = [start_offset] + tmp.tolist()
         max_sizes = tmp.tolist() + [tmp[-1]+start_offset]
         sizes = zip(min_sizes, max_sizes)
    assert len(sizes) == len(from_layers), \
        "sizes and from_layers must have same length"

    if not isinstance(normalization, list):
        normalization = [normalization] * len(from_layers)
    assert len(normalization) == len(from_layers)

    assert sum(x > 0 for x in normalization) <= len(num_channels), \
        "must provide number of channels for each normalized layer"

    if steps:
        assert len(steps) == len(from_layers), "provide steps for all layers or leave empty"

    loc_pred_layers = []
    cls_pred_layers = []
    anchor_layers = []
    num_classes += 1 # always use background as label 0

    for k, params in enumerate(zip(from_layers, num_channels)):
        from_layer, num_filter = params
        from_name = from_layer.name
        # normalize
        if normalization[k] > 0:
            from_layer = mx.symbol.L2Normalization(data=from_layer, \
                mode="channel", name="{}_norm".format(from_name))
            scale = mx.symbol.Variable(name="{}_scale".format(from_name),
                shape=(1, num_channels.pop(0), 1, 1),
                init=mx.init.Constant(normalization[k]),
                attr={'__wd_mult__': '0.1'})
            from_layer = mx.symbol.broadcast_mul(lhs=scale, rhs=from_layer)
        if interm_layer > 0:
            bias1 = mx.symbol.Variable(name="{}_inter_conv_bias1".format(from_name),
                                       init=mx.init.Constant(0.0), attr={'__lr_mult__': '2.0'})
            bias2 = mx.symbol.Variable(name="{}_inter_conv_bias2".format(from_name),
                                       init=mx.init.Constant(0.0), attr={'__lr_mult__': '2.0'})
            from_layer = mx.symbol.Convolution(data=from_layer, kernel=(3,3), \
                stride=(1,1), pad=(1,1),bias=bias1, num_filter=num_filter, \
                name="{}_inter_conv_dw".format(from_name),num_group=num_filter)
            from_layer = mx.symbol.Convolution(data=from_layer, kernel=(1, 1), \
                                               stride=(1, 1), pad=(0, 0),bias=bias2, num_filter=interm_layer, \
                                               name="{}_inter_conv".format(from_name),no_bias=True)
            from_layer = mx.symbol.Activation(data=from_layer, act_type="relu", \
                name="{}_inter_relu".format(from_name))

        # estimate number of anchors per location
        # here I follow the original version in caffe
        # TODO: better way to shape the anchors??
        size = sizes[k]
        assert len(size) > 0, "must provide at least one size"
        size_str = "(" + ",".join([str(x) for x in size]) + ")"
        ratio = ratios[k]
        assert len(ratio) > 0, "must provide at least one ratio"
        ratio_str = "(" + ",".join([str(x) for x in ratio]) + ")"
        num_anchors = len(size) -1 + len(ratio)

        # create location prediction layer
        num_loc_pred = num_anchors * 4

        # bias = mx.symbol.Variable(name="{}_loc_pred_conv_bias".format(from_name),
        #                                init=mx.init.Constant(0.0), attr={'__lr_mult__': '2.0'})
        loc_pred = mx.symbol.Convolution(data=from_layer, kernel=(3, 3), \
                                         stride=(1, 1), pad=(1, 1), num_filter=num_loc_pred, \
                                         name="{}_loc_pred_conv_dw".format(from_name))
        loc_pred = mx.sym.BatchNorm(data=loc_pred, name="{}_loc_pred_bn1".format(from_name), momentum=0.9, eps=0.001,
                                    cudnn_off=True)
        loc_pred = mx.symbol.transpose(loc_pred, axes=(0, 2, 3, 1))
        loc_pred = mx.symbol.Flatten(data=loc_pred)
        loc_pred_layers.append(loc_pred)

        # create class prediction layer
        num_cls_pred = num_anchors * num_classes
        cls_pred = mx.symbol.Convolution(data=from_layer, kernel=(3, 3), \
                                         stride=(1, 1), pad=(1, 1), num_filter=num_cls_pred, \
                                         name="{}_cls_pred_conv_dw".format(from_name))  # ,num_group=num_filter if interm_layer<=0 else interm_layer
        cls_pred = mx.sym.BatchNorm(data=cls_pred, name="{}_cls_pred_bn1".format(from_name),momentum=0.9,eps=0.001,cudnn_off=True)
        cls_pred = mx.symbol.transpose(cls_pred, axes=(0, 2, 3, 1))
        cls_pred = mx.symbol.Flatten(data=cls_pred)
        cls_pred_layers.append(cls_pred)
        # create anchor generation layer
        if steps:
            step = (steps[k], steps[k])
        else:
            step = '(-1.0, -1.0)'
        anchors = mx.contrib.symbol.MultiBoxPrior(from_layer, sizes=size_str, ratios=ratio_str, \
            clip=clip, name="{}_anchors".format(from_name), steps=step)
        anchors = mx.symbol.Flatten(data=anchors)
        anchor_layers.append(anchors)

    loc_preds = mx.symbol.Concat(*loc_pred_layers, num_args=len(loc_pred_layers), \
        dim=1, name="multibox_loc_pred")
    cls_preds = mx.symbol.Concat(*cls_pred_layers, num_args=len(cls_pred_layers), \
        dim=1)
    cls_preds = mx.symbol.Reshape(data=cls_preds, shape=(0, -1, num_classes))
    cls_preds = mx.symbol.transpose(cls_preds, axes=(0, 2, 1), name="multibox_cls_pred")
    anchor_boxes = mx.symbol.Concat(*anchor_layers, \
        num_args=len(anchor_layers), dim=1)
    anchor_boxes = mx.symbol.Reshape(data=anchor_boxes, shape=(0, -1, 4), name="multibox_anchors")
    return [loc_preds, cls_preds, anchor_boxes]
