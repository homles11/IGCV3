"""
Compile MXNet Models
====================
**Author**: `Joshua Z. Zhang <https://zhreshold.github.io/>`_

This article is an introductory tutorial to deploy mxnet models with NNVM.

For us to begin with, mxnet module is required to be installed.

A quick solution is
```
pip install mxnet --user
```
or please refer to offical installation guide.
https://mxnet.incubator.apache.org/versions/master/install/index.html
"""
# some standard imports
import os, sys

import mxnet as mx
import nnvm
import tvm
import numpy as np
import time

print(mx.__file__)
print(nnvm.__file__)
print(tvm.__file__)

target = 'opencl'
target_to_device={'opencl':tvm.cl(0), 'llvm':tvm.cpu(0), 'cuda':tvm.gpu(0), }

######################################################################
# Download Resnet18 model from Gluon Model Zoo
# ---------------------------------------------
# In this section, we download a pretrained imagenet model and classify an image.
# from mxnet.gluon.model_zoo.vision import get_model
from symbols.mobilenetv2 import get_symbol
# from mxnet.gluon.utils import download
from PIL import Image
from matplotlib import pyplot as plt

model_name = 'models/mobilenetv2-1_0'
# symbol = get_symbol()
# img_name = 'data/cat.jpg'
img_name = 'data/beagle.jpg'
synset_url = ''.join(['https://gist.githubusercontent.com/zhreshold/',
                      '4d0b62f3d01426887599d4f7ede23ee5/raw/',
                      '596b27d23537e5a1b5751d2b0481ef172f58b539/',
                      'imagenet1000_clsid_to_human.txt'])
synset_name = 'data/imagenet1k-synset.txt'
# download('https://github.com/dmlc/mxnet.js/blob/master/data/cat.png?raw=true', img_name)
# download(synset_url, synset_name)
with open(synset_name) as f:
    synset = f.readlines()
image = Image.open(img_name).resize((224, 224))
# image = Image.open(img_name).resize((320, 640))
# plt.imshow(image)
# plt.show()

def transform_image(image):
    image = np.array(image) - np.array([123., 117., 104.])
    # image /= np.array([58.395, 57.12, 57.375])
    image = image.transpose((2, 0, 1))
    image = image[np.newaxis, :]
    return image

x = transform_image(image)
print('x', x.shape)

######################################################################
# Compile the Graph
# -----------------
# Now we would like to port the Gluon model to a portable computational graph.
# It's as easy as several lines.
# We support MXNet static graph(symbol) and HybridBlock in mxnet.gluon
# sym, params = nnvm.frontend.from_mxnet(symbol)
# # we want a probability so add a softmax operator
# sym = nnvm.sym.softmax(sym)
mx_sym, args, auxs = mx.model.load_checkpoint(model_name, 0)
# now we use the same API to get NNVM compatible symbol
nnvm_sym, nnvm_params = nnvm.frontend.from_mxnet(mx_sym, args, auxs)
# repeat the same steps to run this model using TVM

######################################################################
# now compile the graph
import nnvm.compiler
shape_dict = {'data': x.shape}
graph, lib, params = nnvm.compiler.build(nnvm_sym, target, shape_dict, params=nnvm_params)

######################################################################
# Execute the portable graph on TVM
# ---------------------------------
# Now, we would like to reproduce the same forward computation using TVM.
from tvm.contrib import graph_runtime
ctx = target_to_device[target]
dtype = 'float32'
m = graph_runtime.create(graph, lib, ctx)
m.set_input(**params)
for i in range(3):
    # set inputs
    m.set_input('data', tvm.nd.array(x.astype(dtype)))
    tic = time.time()
    # execute
    m.run()
    # get outputs
    tvm_output = m.get_output(0, tvm.nd.empty((1000,), dtype))
    toc1 = time.time()
    top1 = np.argsort(tvm_output.asnumpy())[::-1][:5]
    toc2 = time.time()
    print('elapsed: %.1fms (%.1fms)' % ((toc2-tic)*1000.,(toc1-tic)*1000.,))
    for i in range(5):
        print('TVM prediction top-%d:'%(i+1,), top1[i], synset[top1[i]])

######################################################################
# Use MXNet symbol with pretrained weights
# ----------------------------------------
# MXNet often use `arg_prams` and `aux_params` to store network parameters
# separately, here we show how to use these weights with existing API
# def block2symbol(block):
#     data = mx.sym.Variable('data')
#     sym = block(data)
#     args = {}
#     auxs = {}
#     for k, v in block.collect_params().items():
#         args[k] = mx.nd.array(v.data().asnumpy())
#     return sym, args, auxs
# mx_sym, args, auxs = block2symbol(block)
# # usually we would save/load it as checkpoint
# mx.model.save_checkpoint(model_name, 0, mx_sym, args, auxs)
# # there are 'resnet18_v1-0000.params' and 'resnet18_v1-symbol.json' on disk

######################################################################
# for a normal mxnet model, we start from here
mx_sym, args, auxs = mx.model.load_checkpoint(model_name, 0)
# now we use the same API to get NNVM compatible symbol
nnvm_sym, nnvm_params = nnvm.frontend.from_mxnet(mx_sym, args, auxs)
# repeat the same steps to run this model using TVM
