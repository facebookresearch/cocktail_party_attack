#!/usr/bin/env python3
#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from utils import get_device, get_opt, get_sch, normalize
import torch
from datasets import ds_mean, ds_std, nclasses_dict, xshape_dict
from gradient_inversion_utils import CosineDistance


class FeatureInversion:
    def __init__(self, Z, model, args, grads, labels):
        self.args = args
        self.device = get_device()
        self.cosine_similarity = torch.nn.CosineSimilarity(dim=-1, eps=1e-10)
        self.model = model
        self.model.eval()

        self.Z = Z
        self.n = Z.shape[0]
        self.tv = self.args.tv
        self.opt = self.args.opt_fi
        self.sch = self.args.sch_fi
        self.lr = self.args.lr_fi
        self.ds = self.args.ds
        self.n_iter_fi = self.args.n_iter_fi
        self.mean = torch.tensor(ds_mean[self.ds], device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor(ds_std[self.ds], device=self.device).view(1, 3, 1, 1)
        self.grads = grads
        self.labels = labels
        self.Y_hat = None
        self.cosine_distance = CosineDistance()
        self.use_labels = self.args.use_labels
        self.inp_shape = [self.n] + xshape_dict[self.ds]
        self.gm = self.args.gm
        self.fi = self.args.fi
        self.start_iter = 0

    def total_variation(self, x):
        dx = torch.mean(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:]))
        dy = torch.mean(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]))
        return dx + dy


class GradientMatching(FeatureInversion):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.params = []
        self.criterion = torch.nn.CrossEntropyLoss()
        self.S_hat = torch.nn.Parameter(
            torch.randn(
                self.inp_shape,
                requires_grad=True,
                device=self.device,
            )
        )
        self.params.append(self.S_hat)
        if self.use_labels:
            self.Y_hat = self.labels
        else:
            self.Y_hat = torch.nn.Parameter(
                torch.randn(
                    (self.n, nclasses_dict[self.ds]),
                    requires_grad=True,
                    device=self.device,
                )
            )
            self.params.append(self.Y_hat)
        self.opt = get_opt(self.params, self.opt, lr=self.lr)
        self.sch = get_sch(self.sch, self.opt, self.n_iter_fi)

    def get_attack_state(self):
        state_dict = {}
        state_dict["S_hat"] = self.S_hat
        state_dict["Y_hat"] = self.Y_hat
        state_dict["opt"] = self.opt.state_dict()
        if self.sch is not None:
            state_dict["sch"] = self.sch.state_dict()
        return state_dict

    def set_attack_state(self, state_dict):
        self.S_hat.data = state_dict["S_hat"]
        self.Y_hat.data = state_dict["Y_hat"]
        self.opt.load_state_dict(state_dict["opt"])
        if self.sch is not None:
            self.sch.load_state_dict(state_dict["sch"])

    def step(self):
        loss_fi = loss_tv = torch.tensor(0.0, device=self.device)
        # Box Image
        self.S_hat.data = torch.max(
            torch.min(self.S_hat, (1 - self.mean) / self.std),
            -self.mean / self.std,
        )
        self.opt.zero_grad()
        pred, z_hat = self.model(self.S_hat, return_z=True)
        loss_hat = self.criterion(pred, self.Y_hat)
        grad_hat = torch.autograd.grad(
            loss_hat, self.model.parameters(), create_graph=True
        )
        loss_gm = self.cosine_distance(grad_hat, self.grads)

        if self.fi > 0:
            loss_fi = (1 - self.cosine_similarity(self.Z, z_hat)).mean()

        if self.tv > 0:
            loss_tv = self.total_variation(self.S_hat)

        loss = self.fi * loss_fi + self.tv * loss_tv + self.gm * loss_gm

        loss.backward()
        self.opt.step()
        if self.sch:
            self.sch.step()

        return {
            "loss": loss.detach().cpu().item(),
            "loss_fi": loss_fi.detach().cpu().item(),
            "loss_gm": loss_gm.detach().cpu().item(),
            "loss_tv": loss_tv.detach().cpu().item(),
        }

    def get_rec(self):
        with torch.no_grad():
            S_hat = normalize(self.S_hat, method="ds").detach()
        return S_hat


class Direct(FeatureInversion):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.params = []
        self.criterion = torch.nn.CrossEntropyLoss()
        self.S_hat = torch.nn.Parameter(
            torch.randn(
                self.inp_shape,
                requires_grad=True,
                device=self.device,
            )
        )
        for p in self.model.parameters():
            p.requires_grad = False
        self.params.append(self.S_hat)
        self.opt = get_opt(self.params, self.opt, lr=self.lr)
        self.sch = get_sch(self.sch, self.opt, self.n_iter_fi)

    def get_attack_state(self):
        state_dict = {}
        state_dict["S_hat"] = self.S_hat
        state_dict["opt"] = self.opt.state_dict()
        if self.sch is not None:
            state_dict["sch"] = self.sch.state_dict()
        return state_dict

    def set_attack_state(self, state_dict):
        self.S_hat.data = state_dict["S_hat"]

        self.opt.load_state_dict(state_dict["opt"])
        if self.sch is not None:
            self.sch.load_state_dict(state_dict["sch"])

    def step(self):
        loss_fi = loss_gm = loss_tv = torch.tensor(0.0, device=self.device)
        # Box Image
        self.S_hat.data = torch.max(
            torch.min(self.S_hat, (1 - self.mean) / self.std),
            -self.mean / self.std,
        )
        self.opt.zero_grad()
        _, z_hat = self.model(self.S_hat, return_z=True)
        if self.fi > 0:
            loss_fi = (1 - self.cosine_similarity(self.Z, z_hat)).mean()

        if self.tv > 0:
            loss_tv = self.total_variation(self.S_hat)

        loss = self.fi * loss_fi + self.tv * loss_tv

        loss.backward()
        self.opt.step()
        if self.sch:
            self.sch.step()

        return {
            "loss": loss.detach().cpu().item(),
            "loss_fi": loss_fi.detach().cpu().item(),
            "loss_gm": loss_gm.detach().cpu().item(),
            "loss_tv": loss_tv.detach().cpu().item(),
        }

    def get_rec(self):
        with torch.no_grad():
            S_hat = normalize(self.S_hat, method="ds").detach()
        return S_hat


fi_method_dict = {"direct": Direct, "gm": GradientMatching}


def get_fi(fi_method, z, model, args, grads, labels, attack_log):
    if fi_method not in fi_method_dict.keys():
        raise ValueError(f"Unknown feature inversion method: {fi_method}")
    else:
        fi = fi_method_dict[fi_method](z, model, args, grads, labels)

    if attack_log.restore_required:
        fi.set_attack_state(attack_log.attack_state)
        attack_log.restore_required = False
        fi.start_iter = attack_log.iter

    return fi
