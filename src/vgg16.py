#!/usr/bin/env python3
#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torchvision.models import VGG
from functools import partial
from typing import Union, List, Dict, Any, Optional, cast
from torchvision.transforms._presets import ImageClassification
from torchvision.utils import _log_api_usage_once
from torchvision.models._api import WeightsEnum, Weights
from torchvision.models._meta import _IMAGENET_CATEGORIES
from torchvision.models._utils import handle_legacy_interface, _ovewrite_named_param
import torch.nn as nn

__all__ = [
    "VGG",
    "VGG11_Weights",
    "VGG11_BN_Weights",
    "VGG13_Weights",
    "VGG13_BN_Weights",
    "VGG16_Weights",
    "VGG16_BN_Weights",
    "VGG19_Weights",
    "VGG19_BN_Weights",
    "vgg11",
    "vgg11_bn",
    "vgg13",
    "vgg13_bn",
    "vgg16",
    "vgg16_bn",
    "vgg19",
    "vgg19_bn",
]


cfgs: Dict[str, List[Union[str, int]]] = {
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "D": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "E": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        512,
        "M",
    ],
}

_COMMON_META = {
    "min_size": (32, 32),
    "categories": _IMAGENET_CATEGORIES,
    "recipe": (
        "https://github.com/pytorch/vision/tree/main/references/classification#alexnet-and-vgg"
    ),
    "_docs": """These weights were trained from scratch by using a simplified training recipe.""",
}

# Modified VGG with return_z option
class VGG_mod(VGG):
    def __init__(self, *args, **kwargs):
        super().__init__(*args)
        layers = []
        # Make dropout probability 0
        for layer in self.classifier:
            if isinstance(layer, nn.Dropout):
                layer = nn.Dropout(p=0)
            layers.append(layer)
        self.classifier = nn.Sequential(*layers)
        self.model_type = "conv"
        self.pretrained = True
        self.attack_index = 26

    def forward(self, x: torch.Tensor, return_z=False) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        z = torch.flatten(x, 1)
        x = self.classifier(z)
        if return_z:
            return x, z
        else:
            return x


def make_layers(cfg: List[Union[str, int]], batch_norm: bool = False) -> nn.Sequential:
    layers: List[nn.Module] = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = cast(int, v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


def _vgg(
    cfg: str, batch_norm: bool, weights: Optional[WeightsEnum], progress: bool, **kwargs: Any
) -> VGG:
    if weights is not None:
        kwargs["init_weights"] = False
        if weights.meta["categories"] is not None:
            _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))
    model = VGG_mod(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)
    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress))
    return model


class VGG16_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(
        url="https://download.pytorch.org/models/vgg16-397923af.pth",
        transforms=partial(ImageClassification, crop_size=224),
        meta={
            **_COMMON_META,
            "num_params": 138357544,
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 71.592,
                    "acc@5": 90.382,
                }
            },
        },
    )
    IMAGENET1K_FEATURES = Weights(
        # Weights ported from https://github.com/amdegroot/ssd.pytorch/
        url="https://download.pytorch.org/models/vgg16_features-amdegroot-88682ab5.pth",
        transforms=partial(
            ImageClassification,
            crop_size=224,
            mean=(0.48235, 0.45882, 0.40784),
            std=(1.0 / 255.0, 1.0 / 255.0, 1.0 / 255.0),
        ),
        meta={
            **_COMMON_META,
            "num_params": 138357544,
            "categories": None,
            "recipe": "https://github.com/amdegroot/ssd.pytorch#training-ssd",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": float("nan"),
                    "acc@5": float("nan"),
                }
            },
            "_docs": """
                These weights can't be used for classification because they are missing values in the `classifier`
                module. Only the `features` module has valid values and can be used for feature extraction. The weights
                were trained using the original input standardization method as described in the paper.
            """,
        },
    )
    DEFAULT = IMAGENET1K_V1


@handle_legacy_interface(weights=("pretrained", VGG16_Weights.IMAGENET1K_V1))
def vgg16(*, weights: Optional[VGG16_Weights] = None, progress: bool = True, **kwargs: Any) -> VGG:
    weights = VGG16_Weights.verify(weights)

    return _vgg("D", False, weights, progress, **kwargs)
