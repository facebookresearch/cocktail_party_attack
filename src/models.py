#!/usr/bin/env python3
#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import numpy as np
from functools import partial
import logging
from utils import MyDataParallel

from vgg16 import vgg16
from datasets import xshape_dict, nclasses_dict

act_dict = {"relu": nn.ReLU(), "leaky_relu": nn.LeakyReLU(0.2)}

supported_ds = {
    "fc2": ["cifar10", "cifar100", "tiny_imagenet"],
    "vgg16": ["imagenet"],
}

model_type_dict = {"fc2": "fc", "vgg16": "conv"}


def get_model(
    model_name,
    ds,
    h_dim=256,
    dataparallel=True,
    load_path=None,
):
    logger = logging.getLogger()
    if ds not in supported_ds[model_name]:
        raise ValueError(f"{model_name} does not support {ds}")

    model = model_class_dict[model_name](
        ds=ds,
        h_dim=h_dim,
    )
    pretrained = model.pretrained

    if pretrained:
        logger.info(f"\nUsing Pretrained model for {model_name}")
    elif load_path is not None and pretrained == False:
        logger.info(f"\nLoading model from {load_path}")
        state_dict = torch.load(load_path)
        model.load_state_dict(state_dict)

    if dataparallel:
        device_list = [i for i in range(torch.cuda.device_count())]
        model = MyDataParallel(model, device_ids=device_list)

    return model


class FC2(nn.Module):
    def __init__(self, ds="cifar10", act="relu", h_dim=256, **kwargs):
        super().__init__()
        act_fn = act_dict[act]

        x_dim = np.prod(xshape_dict[ds])
        n_classes = nclasses_dict[ds]
        self.net = nn.Sequential(
            nn.Linear(x_dim, h_dim),
            act_fn,
        )
        self.final = nn.Linear(h_dim, n_classes)
        self.cpa_attack_layer_index = 0
        self.model_type = "fc"
        self.attack_index = 0
        self.pretrained = False

    def forward(self, x, **kwargs):
        x = torch.flatten(x, start_dim=1, end_dim=-1)
        x = self.net(x)
        return self.final(x)


model_class_dict = {
    "fc2": FC2,
    "vgg16": partial(vgg16, pretrained=True),
}
