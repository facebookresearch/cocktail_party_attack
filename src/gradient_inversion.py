#!/usr/bin/env python3
#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F
from utils import (
    get_opt,
    get_sch,
    get_device,
    write_pickle,
    read_pickle,
    normalize,
    get_attack_exp_path,
)
from gradient_inversion_utils import CosineDistance
from datasets import xshape_dict, ds_type_dict, ds_mean, ds_std, nclasses_dict
from abc import abstractmethod
from os.path import exists


class GradientInversionAttack:
    def __init__(self, model, grads, labels, args, batch_id):

        self.args = args
        self.model = model
        self.model.train()
        self.grads = grads
        self.labels = labels
        self.use_labels = args.use_labels
        self.device = get_device()
        self.n_iter = args.n_iter
        self.n_comp = args.batch_size
        self.sigma = args.sigma * args.C
        self.eps = torch.tensor(1e-20, device=self.device)
        self.ds = args.ds

        self.tv = args.tv
        self.l1 = args.l1
        self.nv = args.nv
        self.ne = args.ne
        self.T = args.T
        self.decor = args.decor
        self.set_inp_type()
        self.mean = torch.tensor(ds_mean[self.ds], device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor(ds_std[self.ds], device=self.device).view(1, 3, 1, 1)
        self.batch_id = batch_id
        self.start_iter = 0

    def total_variation(self, x):
        dx = torch.mean(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:]))
        dy = torch.mean(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]))
        return dx + dy

    def make_dict(self, keys, vals):
        return {k: v.cpu().detach().item() for k, v in zip(keys, vals)}

    @abstractmethod
    def set_inp_type(self):
        pass


class CocktailPartyAttack(GradientInversionAttack):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        exp_path = get_attack_exp_path(self.args)
        self.w_pkl_file = f"{exp_path}/w_{self.n_comp}_{self.batch_id}.pkl"
        self.X = self.grads[self.model.attack_index]
        self.X_zc, self.X_mu = self.zero_center(self.X)

        if exists(self.w_pkl_file):
            w_data = read_pickle(self.w_pkl_file)
            self.X_w, self.W_w = w_data["X_w"].to(self.device), w_data["W_w"].to(
                self.device
            )
            print(f"loaded w data from {self.w_pkl_file}")
        else:
            self.X_w, self.W_w = self.whiten(self.X_zc)
            write_pickle(
                {"X_w": self.X_w.detach().cpu(), "W_w": self.W_w.detach().cpu()},
                self.w_pkl_file,
            )

        self.W_hat = torch.empty(
            [self.n_comp, self.n_comp],
            dtype=torch.float,
            requires_grad=True,
            device=self.device,
        )
        torch.nn.init.eye_(self.W_hat)
        param_list = [self.W_hat]

        self.opt = get_opt(param_list, self.args.opt, lr=self.args.lr)
        self.sch = get_sch(self.args.sch, self.opt, epochs=self.args.n_iter)
        self.a = 1

    def set_inp_type(self):
        if ds_type_dict[self.ds] == "image" and self.model.model_type == "fc":
            self.inp_type = "image"
            self.inp_shape = [self.n_comp] + xshape_dict[self.ds]
        else:
            self.inp_type = "emb"
            self.inp_shape = [self.n_comp, -1]

    def zero_center(self, x):
        x_mu = x.mean(dim=-1, keepdims=True)
        return x - x_mu, x_mu

    def whiten(self, x):
        cov = torch.matmul(x, x.T) / (x.shape[1] - 1)
        eig_vals, eig_vecs = torch.linalg.eig(cov)
        topk_indices = torch.topk(eig_vals.float().abs(), self.n_comp)[1]

        U = eig_vecs.float()
        lamb = eig_vals.float()[topk_indices].abs()
        lamb_inv_sqrt = torch.diag(1 / (torch.sqrt(lamb) + self.eps)).float()
        W = torch.matmul(lamb_inv_sqrt, U.T[topk_indices]).float()
        x_w = torch.matmul(W, x)
        return x_w, W

    def get_attack_state(self):
        state_dict = {}
        state_dict["W_hat"] = self.W_hat
        state_dict["opt"] = self.opt.state_dict()
        if self.sch is not None:
            state_dict["sch"] = self.sch.state_dict()
        return state_dict

    def set_attack_state(self, state_dict):
        self.W_hat.data = state_dict["W_hat"].data
        self.opt.load_state_dict(state_dict["opt"])
        if self.sch is not None:
            self.sch.load_state_dict(state_dict["sch"])

    def step(self):
        loss_ne = loss_decor = loss_nv = loss_tv = loss_l1 = torch.tensor(
            0.0, device=self.device
        )

        self.opt.zero_grad()
        W_hat_norm = self.W_hat / (self.W_hat.norm(dim=-1, keepdim=True) + self.eps)

        # Neg Entropy Loss
        X_w = self.X_w
        S_hat = torch.matmul(W_hat_norm, X_w)
        

        if torch.isnan(S_hat).any():
            raise ValueError(f"S_hat has NaN")

        if self.ne > 0:
            loss_ne = -(
                (
                    (1 / self.a)
                    * torch.log(torch.cosh(self.a * S_hat) + self.eps).mean(dim=-1)
                )
                ** 2
            ).mean()
            loss_ne = torch.tensor(0.0, device=self.device)

        # Undo centering, whitening
        S_hat = S_hat + torch.matmul(torch.matmul(W_hat_norm, self.W_w), self.X_mu)

        # Decorrelation Loss (decorrelate i-th row with j-th row, s.t. j>i)
        if self.decor > 0:
            cos_matrix = torch.matmul(W_hat_norm, W_hat_norm.T).abs()
            loss_decor = (torch.exp(cos_matrix * self.T) - 1).mean()

        # Prior Loss
        if self.tv > 0 and self.nv == 0:  # if nv > 0, tv is meant for the generator
            loss_tv = self.total_variation(S_hat.view(self.inp_shape))

        if self.nv > 0:
            loss_nv = torch.minimum(
                F.relu(-S_hat).norm(dim=-1), F.relu(S_hat).norm(dim=-1)
            ).mean()

        if self.l1 > 0:
            loss_l1 = torch.abs(S_hat).mean()

        loss = (
            loss_ne
            + (self.decor * loss_decor)
            + (self.tv * loss_tv)
            + (self.nv * loss_nv)
            + (self.l1 * loss_l1)
        )

        loss.backward()
        self.opt.step()
        if self.sch:
            self.sch.step()
        loss_dict = self.make_dict(
            [
                "loss",
                "loss_ne",
                "loss_decor",
                "loss_tv",
                "loss_nv",
                "loss_l1",
            ],
            [
                loss,
                loss_ne,
                loss_decor,
                loss_tv,
                loss_nv,
                loss_l1,
            ],
        )

        return loss_dict

    def get_rec(self):
        with torch.no_grad():
            W_hat_norm = self.W_hat / (self.W_hat.norm(dim=-1, keepdim=True) + self.eps)
            S_hat = torch.matmul(W_hat_norm, self.X_w)
            S_hat = S_hat + torch.matmul(torch.matmul(W_hat_norm, self.W_w), self.X_mu)
            S_hat = S_hat.detach().view(self.inp_shape)
            if self.inp_type == "image":
                S_hat = normalize(S_hat, method="infer")
        return S_hat


class GradientMatchingAttack(GradientInversionAttack):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.params = []
        if self.use_labels:
            self.Y_hat = self.labels
        else:
            self.Y_hat = torch.nn.Parameter(
                torch.randn(
                    (self.n_comp, nclasses_dict[self.ds]),
                    requires_grad=True,
                    device=self.device,
                )
            )
            self.params.append(self.Y_hat)
        self.S_hat = torch.nn.Parameter(
            torch.randn(
                self.inp_shape,
                requires_grad=True,
                device=self.device,
            )
        )
        self.params.append(self.S_hat)
        self.opt = get_opt(self.params, self.args.opt, lr=self.args.lr)
        self.sch = get_sch(self.args.sch, self.opt, epochs=self.args.n_iter)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.cosine_distance = CosineDistance()

    def get_attack_state(self):
        state_dict = {}
        state_dict["S_hat"] = self.S_hat
        state_dict["Y_hat"] = self.Y_hat
        state_dict["opt"] = self.opt.state_dict()
        if self.sch is not None:
            state_dict["sch"] = self.sch.state_dict()
        return state_dict

    def set_attack_state(self, state_dict):
        self.S_hat.data = state_dict["S_hat"].data
        self.Y_hat.data = state_dict["Y_hat"].data
        self.opt.load_state_dict(state_dict["opt"])
        if self.sch is not None:
            self.sch.load_state_dict(state_dict["sch"])

    def set_inp_type(self):
        if ds_type_dict[self.ds] == "image":
            self.inp_type = "image"
        else:
            self.inp_type = "emb"
        self.inp_shape = [self.n_comp] + xshape_dict[self.ds]

    def step(self):
        loss_cd = loss_tv = torch.tensor(0.0, device=self.device)

        # Box Image
        self.S_hat.data = torch.max(
            torch.min(self.S_hat, (1 - self.mean) / self.std),
            -self.mean / self.std,
        )

        self.opt.zero_grad()
        pred = self.model(self.S_hat)
        loss_hat = self.criterion(pred, self.Y_hat)
        grad_hat = torch.autograd.grad(
            loss_hat, self.model.parameters(), create_graph=True
        )
        loss_cd = self.cosine_distance(grad_hat, self.grads)

        if self.tv > 0:
            loss_tv = self.total_variation(self.S_hat)

        loss = loss_cd + (self.tv * loss_tv)
        loss.backward()
        self.opt.step()

        if self.sch:
            self.sch.step()
        loss_dict = self.make_dict(
            ["loss", "loss_cd", "loss_tv"], [loss, loss_cd, loss_tv]
        )

        return loss_dict

    def get_rec(self):
        with torch.no_grad():
            if self.inp_type == "image":
                S_hat = normalize(self.S_hat, method="ds").detach()
        return S_hat


attack_dict = {"cp": CocktailPartyAttack, "gm": GradientMatchingAttack}


def get_gi(attack, model, grads, labels, args, batch_id, attack_log):
    gi = attack_dict[attack](model, grads, labels, args, batch_id)
    if attack_log.restore_required:
        gi.set_attack_state(attack_log.attack_state)
        attack_log.restore_required = False
        gi.start_iter = attack_log.iter
    return gi
