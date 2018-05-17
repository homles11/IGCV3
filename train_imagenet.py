# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import os
import argparse
import logging
logging.basicConfig(level=logging.DEBUG)
import time
from common import find_mxnet, data, fit
from common.util import download_file
import mxnet as mx

thisdir = os.path.dirname(os.path.abspath(__file__))

if __name__ == '__main__':
    # parse args
    parser = argparse.ArgumentParser(description="train imagenet-1k",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    fit.add_fit_args(parser)
    data.add_data_args(parser)
    data.add_data_aug_args(parser)
    # use a large aug level
    data.set_data_aug_level(parser, 2)
    parser.set_defaults(
        # network
        network          = 'igcv3',
        multiplier       = 1.0, 
        model_prefix     = os.path.join(thisdir,'models/igcv3'), 

        # data
        data_train       = './imagenet_data/imagenet_train.rec',
        # data_train_idx   = '/home/shared/ILSVRC2012_img_train_output.idx',
        data_val         = './imagenet_data/imagenet_val.rec',
        # data_val_idx     = '/home/shared/ILSVRC2012_img_val_output.idx',
        num_classes      = 1000,
        num_examples     = 1281167,
        image_shape      = '3,224,224',
        max_random_scale = 1.0,
        min_random_scale = 0.533, # if input image has min size k, suggest to use
                                  # 256.0/x, e.g. 0.533 for 480
        # train
        num_epochs       = 480, # default=480 epochs
        lr               = 0.045, # default=0.045 
        lr_factor        = 0.98, # default=0.98
        lr_step_epochs   = ','.join([str(i) for i in range(1,480)]),
        wd               = 0.00004, 
        dtype            = 'float32', 
        batch_size       = 96,
        gpus             = '0,1,2,3',
        optimizer        = 'sgd',
        # monitor          = 20, 
        load_epoch       = None, # default=None
        top_k            = 5,
    )
    args = parser.parse_args()
    from pprint import pprint
    pprint(vars(args))

    # load network
    from importlib import import_module
    net = import_module('symbols.'+args.network)
    sym = net.get_symbol(num_classes=args.num_classes, multiplier=args.multiplier)
    # print(sym.get_internals()['mobilenetv20_features_conv0_weight'].attr_dict()['mobilenetv20_features_conv0_weight']['__shape__'])
    # exit()

    # set up logger
    logger = logging.getLogger()
    fh = logging.FileHandler(os.path.join('log',time.strftime('%F-%T',time.localtime()).replace(':','-')+'.log'))
    fh.setLevel(logging.DEBUG)
    # ch = logging.StreamHandler()
    # ch.setLevel(logging.INFO)
    logger.addHandler(fh)
    # logger.addHandler(ch)

    # train
    fit.fit(args, sym, data.get_rec_iter, logger)
