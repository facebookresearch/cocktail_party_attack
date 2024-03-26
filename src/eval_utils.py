#!/usr/bin/env python3
#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
import torch.nn as nn
from torchmetrics.functional import (
    peak_signal_noise_ratio,
    structural_similarity_index_measure,
)
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from utils import normalize

# lpips requires inputs in the range [-1,1]
def normalize_lpips(x):
    return (x - 0.5) * 2


class Eval:
    def __init__(self, S, fix_order_method=None, fix_sign_method=None, ds="cifar10"):
        self.S = S
        self.device = S.device
        self.compute_lpips = LearnedPerceptualImagePatchSimilarity(net_type="vgg").to(
            self.device
        )
        self.eps = 1e-20
        self.fix_order_method = fix_order_method
        self.fix_sign_method = fix_sign_method
        self.n = self.S.shape[0]
        self.ds = ds

    def fix_order(self, S_hat):
        assert S_hat.shape[0] == self.n

        if self.fix_order_method is None:
            return S_hat
        elif self.fix_order_method == "ssim":
            S_hat_inv = 1 - S_hat
            scores = []
            for S_i in self.S.unsqueeze(1):
                for S_hat_i, S_hat_inv_i in zip(
                    S_hat.unsqueeze(1), S_hat_inv.unsqueeze(1)
                ):
                    score = (
                        structural_similarity_index_measure(S_hat_i, S_i).cpu().item()
                    )
                    score_inv = (
                        structural_similarity_index_measure(S_hat_inv_i, S_i)
                        .cpu()
                        .item()
                    )
                    scores.append(np.max([score, score_inv]))
            similarity_matrix = torch.tensor(np.array(scores), device=self.device).view(
                self.n, self.n
            )
        elif self.fix_order_method == "lpips":
            S_hat_inv = 1 - S_hat
            scores = []
            for S_i in self.S.unsqueeze(1):
                for S_hat_i, S_hat_inv_i in zip(
                    S_hat.unsqueeze(1), S_hat_inv.unsqueeze(1)
                ):

                    score = (
                        self.compute_lpips(
                            normalize_lpips(S_hat_i), normalize_lpips(S_i)
                        )
                        .cpu()
                        .item()
                    )
                    score_inv = (
                        self.compute_lpips(
                            normalize_lpips(S_hat_inv_i), normalize_lpips(S_i)
                        )
                        .cpu()
                        .item()
                    )
                    scores.append(np.min([score, score_inv]))
            # invert because lower is better
            similarity_matrix = -torch.tensor(
                np.array(scores), device=self.device
            ).view(self.n, self.n)
        elif self.fix_order_method == "cs":
            # Note: We are matching based on absolute value of cosine similarity.
            # Ideally this should be done only for CPA (where sign is not preserved) and not for GMA
            S_unit = self.S / self.S.norm(dim=-1, keepdim=True)
            S_hat_unit = S_hat / S_hat.norm(dim=-1, keepdim=True)
            cs_matrix = torch.matmul(S_unit, S_hat_unit.T).abs()
            similarity_matrix = cs_matrix
        else:
            raise ValueError("Unknown method: ", self.fix_order_method)

        sorted_vals, sorted_indices = torch.sort(
            similarity_matrix.view(-1), descending=True
        )
        ordered_indices = -np.ones(self.n)
        rec_used = np.zeros(self.n)
        for index in sorted_indices.cpu().detach().numpy():
            source_index = int(index / self.n)
            rec_index = int(index % self.n)
            if ordered_indices[source_index] == -1 and rec_used[rec_index] == 0:
                ordered_indices[source_index] = rec_index
                rec_used[rec_index] = 1

        S_hat_reordered = S_hat[ordered_indices]
        return S_hat_reordered

    def fix_sign(self, S_hat):
        # Flatten
        S_flat = self.S.view([self.n, -1])
        S_hat_flat = S_hat.view([self.n, -1])

        if self.fix_sign_method is None:
            return S_hat
        elif self.fix_sign_method == "best":
            mse = ((S_flat - S_hat_flat) ** 2).mean(dim=-1)
            mse_inv = ((S_flat - (1 - S_hat_flat)) ** 2).mean(dim=-1)
            flip_mask = (mse_inv < mse).float().view([self.n, 1, 1, 1])
            S_hat_sign_fixed = S_hat * (1 - flip_mask) + (1 - S_hat) * flip_mask
            return S_hat_sign_fixed
        else:
            raise ValueError("Unknown method: ", self.fix_sign_method)


class ImageEval(Eval):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.S = normalize(self.S, method="ds", ds=self.ds)
        self.compute_lpips = LearnedPerceptualImagePatchSimilarity(net_type="vgg").to(
            self.device
        )

    def __call__(self, S_hat):
        S_hat = self.fix_order(S_hat)
        S_hat = self.fix_sign(S_hat)
        S_hat = S_hat + self.eps

        psnr_batch, ssim_batch, lpips_batch = [], [], []

        for S_hat_i, S_i in zip(S_hat.unsqueeze(1), self.S.unsqueeze(1)):
            psnr_batch.append(
                peak_signal_noise_ratio(S_hat_i.view(-1), S_i.view(-1)).cpu().item()
            )
            ssim_batch.append(
                structural_similarity_index_measure(S_hat_i, S_i).cpu().item()
            )
            lpips_batch.append(
                self.compute_lpips(normalize_lpips(S_hat_i), normalize_lpips(S_i))
                .cpu()
                .item()
            )

        metrics_batch = {
            "psnr": psnr_batch,
            "ssim": ssim_batch,
            "lpips": lpips_batch,
        }
        metrics_avg = {
            "psnr": np.mean(psnr_batch),
            "ssim": np.mean(ssim_batch),
            "lpips": np.mean(lpips_batch),
        }

        return metrics_avg, metrics_batch, S_hat


class EmbEval(Eval):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cs = nn.CosineSimilarity(dim=0, eps=self.eps)

    def __call__(self, S_hat):
        S_hat = self.fix_order(S_hat)
        S_hat = self.fix_sign(S_hat)
        cs_batch = []
        for S_hat_i, S_i in zip(S_hat, self.S):
            cs_i = self.cs(S_hat_i, S_i).abs()
            cs_batch.append(cs_i.cpu().numpy())

        metrics_batch = {
            "cs": cs_batch,
        }
        metrics_avg = {
            "cs": np.mean(cs_batch),
        }

        return metrics_avg, metrics_batch, S_hat


def get_eval(inp, emb, model_type, ds_type, attack, fi=False):
    if attack == "gm":
        if ds_type == "image":
            eval = ImageEval(
                inp,
                # fix_order_method="lpips", #DBG
                fix_order_method="ssim",
            )
        else:
            eval = EmbEval(inp, fix_order_method="cs")
    else:
        if ds_type == "image":
            if model_type == "fc":
                eval = ImageEval(inp, fix_order_method="ssim", fix_sign_method="best")
                # DBG
                # eval = ImageEval(inp, fix_order_method="lpips", fix_sign_method="best")
            elif fi is False:  # model_type is "conv"
                eval = EmbEval(emb, fix_order_method="cs", fix_sign_method=None)
            else:  # fi
                eval = ImageEval(inp, fix_order_method=None, fix_sign_method=None)
        else:  # Text
            eval = EmbEval(inp, fix_order_method="cs", fix_sign_method=None)

    return eval
