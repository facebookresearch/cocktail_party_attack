#!/usr/bin/env python3
#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch

attack_name_dict = {"cp": "Cocktail Party Attack", "gm": "Gradient Matching Attack"}


class GradientMetric:
    def __init__(self, indices="all") -> None:
        self.indices = indices

    def determine_indices(self, input_gradient):
        indices = self.indices
        if isinstance(indices, list):
            pass
        elif indices == "all":
            indices = torch.arange(len(input_gradient))
        else:
            raise ValueError()
        return indices


class CosineDistance(GradientMetric):
    def __init__(self, indices="all") -> None:
        super().__init__(indices=indices)

    def __call__(self, trial_gradients, input_gradients):
        with torch.no_grad():
            indices = self.determine_indices(input_gradients)
            filtered_trial_gradients = [trial_gradients[i] for i in indices]
            filtered_input_gradients = [input_gradients[i] for i in indices]

        costs = sum(
            (x * y).sum() for x, y in zip(filtered_input_gradients, filtered_trial_gradients)
        )

        trial_norm = sum(x.pow(2).sum() for x in filtered_trial_gradients).sqrt()
        input_norm = sum(y.pow(2).sum() for y in filtered_input_gradients).sqrt()
        costs = 1 - (costs / trial_norm / input_norm)
        return costs
