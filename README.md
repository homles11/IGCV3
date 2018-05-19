# IGCV3
**IGCV3:Interleaved Low-Rank Group Convolutions for Efficient Deep Neural Networks.**  
IGCV3 code and pretrained model based on https://github.com/liangfu/mxnet-mobilenet-v2.


## Requirements
- Install [MXNet](https://mxnet.incubator.apache.org/install/index.html)

## How to Train
Current code supports training IGCNV3 on ImageNet, such as `IGCV3`s, `MobileNet-V2`. All the networks are contained in the `symbol` folder.


For example, running the following command can train the `IGCV3` network on ImageNet.

```shell
python train_imagenet.py --network=IGCV3 --multiplier=1.0 --gpus=0,1,2,3,4,5,6,7 --batch-size=96 --data-dir=<dataset location>
```

## Citation

Please cite our papers in your publications if it helps your research:

```
@article{WangWZZ16,
  author    = {Jingdong Wang and
               Zhen Wei and
               Ting Zhang and
               Wenjun Zeng},
  title     = {Deeply-Fused Nets},
  journal   = {CoRR},
  volume    = {abs/1605.07716},
  year      = {2016},
  url       = {http://arxiv.org/abs/1605.07716}
}
```

```
@article{ZhaoWLTZ16,
  author    = {Liming Zhao and
               Jingdong Wang and
               Xi Li and
               Zhuowen Tu and
               Wenjun Zeng},
  title     = {On the Connection of Deep Fusion to Ensembling},
  journal   = {CoRR},
  volume    = {abs/1611.07718},
  year      = {2016},
  url       = {http://arxiv.org/abs/1611.07718}
}
```

```
@article{DBLP:journals/corr/ZhangQ0W17,
  author    = {Ting Zhang and
               Guo{-}Jun Qi and
               Bin Xiao and
               Jingdong Wang},
  title     = {Interleaved Group Convolutions for Deep Neural Networks},
  journal   = {CoRR},
  volume    = {abs/1707.02725},
  year      = {2017},
  url       = {http://arxiv.org/abs/1707.02725}
}
```

```
@article{DBLP:journals/corr/abs-1804-06202,
  author    = {Guotian Xie and
               Jingdong Wang and
               Ting Zhang and
               Jianhuang Lai and
               Richang Hong and
               Guo{-}Jun Qi},
  title     = {{IGCV2:} Interleaved Structured Sparse Convolutional Neural Networks},
  journal   = {CoRR},
  volume    = {abs/1804.06202},
  year      = {2018},
  url       = {http://arxiv.org/abs/1804.06202},
  archivePrefix = {arXiv},
  eprint    = {1804.06202},
  timestamp = {Wed, 02 May 2018 15:55:01 +0200},
  biburl    = {https://dblp.org/rec/bib/journals/corr/abs-1804-06202},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
