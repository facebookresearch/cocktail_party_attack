#!/usr/bin/env python3
#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import os
import math
import torch.nn as nn
import numpy as np
import random
from torch.backends import cudnn
from tqdm import tqdm
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
import logging
import sys
from datasets import ds_mean, ds_std
import pickle

exp_path_base = "./exp"
submitit_logdir = "./exp/submititlogs"
slurm_partition = ""
wandb_entity = ""


def read_pickle(pkl_file):
    with open(pkl_file, "rb") as handle:
        data = pickle.load(handle)
    return data


def get_grads_file(ds, model, defense, batch_size, C, sigma):
    exp_path = f"{exp_path_base}/{ds}/{model}/grads/{defense}"
    if defense == "nodef":
        grads_file = f"{exp_path}/{batch_size}.pickle"
    else:
        grads_file = f"{exp_path}/{batch_size}_C_{C}_sigma_{sigma}.pickle"
    return grads_file


def get_updates_file(ds, model, n_samples):
    exp_path = f"{exp_path_base}/{ds}/{model}/updates/nodef"
    updates_file = f"{exp_path}/{n_samples}.pickle"
    return updates_file


def write_pickle(data, pkl_file):
    with open(pkl_file, "wb") as handle:
        data = pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def subsample(input_list, n, n_sample):
    if n <= n_sample:
        return input_list
    else:
        assert n_sample < n
        return [input[:n_sample] for input in input_list]


def normalize(inp, method=None, ds="imagenet"):
    device = inp.device
    if method is None:
        pass
    elif method == "infer":
        orig_shape = inp.shape
        n = orig_shape[0]
        inp = inp.view([n, -1])
        inp = (inp - inp.min(dim=-1, keepdim=True)[0]) / (
            inp.max(dim=-1, keepdim=True)[0] - inp.min(dim=-1, keepdim=True)[0]
        )
        inp = inp.view(orig_shape)
    elif method == "ds":
        mean = torch.tensor(ds_mean[ds], device=device).view(1, 3, 1, 1)
        std = torch.tensor(ds_std[ds], device=device).view(1, 3, 1, 1)
        inp = torch.clamp((inp * std) + mean, 0, 1)
    else:
        raise ValueError(f"Unknown method {method}")
    return inp


class MyDataParallel(nn.DataParallel):
    """
    Allow nn.DataParallel to call model's attributes.
    """

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


def setup(exp_path, log_file):
    set_seed(42)
    enable_gpu_benchmarking()
    if not os.path.exists(exp_path):
        os.makedirs(exp_path)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    output_file_handler = logging.FileHandler(f"{exp_path}/{log_file}")
    stdout_handler = logging.StreamHandler(sys.stdout)
    logger.addHandler(output_file_handler)
    logger.addHandler(stdout_handler)
    return logger


def get_pbar(iter, disable=False, enu=False):
    t = len(iter)
    if enu:
        iter = enumerate(iter)
    if disable:
        pbar = tqdm(iter, total=t, leave=False, disable=True)
    else:
        pbar = tqdm(iter, total=t, leave=False)
    return pbar


def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device


def wait_for_job(job):
    output = job.result()
    print(output)


def enable_gpu_benchmarking():
    if torch.cuda.is_available():
        cudnn.enabled = True
        cudnn.benchmark = True


def set_seed(seed: int = 0):
    random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def save_model(model: nn.Module, path: str = "./"):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)
    try:
        torch.save(model.module.state_dict(), path)
    except AttributeError:
        torch.save(model.state_dict(), path)

    logger = logging.getLogger()
    logger.info(f"\nSaving model in {path}")


def train_epoch(
    model,
    train_loader,
    opt,
    criterion=nn.CrossEntropyLoss(),
    task_type="classification",
    disable_pbar=False,
):
    device = get_device()
    model = model.to(device)
    model.train()
    running_loss = correct_total = 0.0
    n_batches = len(train_loader)
    pbar = get_pbar(train_loader, disable=disable_pbar, enu=True)

    for it, (x, y) in pbar:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        pred = model(x)
        loss = criterion(pred.view(-1, pred.size(-1)), y.view(-1))
        loss.backward()
        opt.step()
        running_loss += loss.item()

        if task_type == "classification":
            pred_class = torch.argmax(pred, dim=-1)
            correct = (pred_class == y).sum().item()
            correct_total += correct
            batch_size = len(x)

            pbar.set_description(
                f"Iter {it}: train loss {loss.item():.3f} train acc {100 * correct/batch_size:.3f}"
            )
        elif task_type == "language_model":
            train_ppl = math.exp(loss.item())
            pbar.set_description(
                f"Iter {it}: train loss {loss.item():.3f} train ppl {train_ppl:.3f}"
            )

    loss = running_loss / n_batches
    acc = correct_total / len(train_loader.dataset)
    return loss, acc


def test(model, data_loader, criterion, n_iter=None, task_type="classification"):
    device = get_device()
    model = model.to(device)
    model.eval()
    correct = 0.0
    n_inputs = 0
    loss_list = []
    with torch.no_grad():
        for it, (x, y) in enumerate(data_loader):
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = criterion(pred.view(-1, pred.size(-1)), y.view(-1))
            loss_list.append(loss.item())
            if task_type == "classification":
                pred_class = torch.argmax(pred, dim=1)
                correct += (pred_class == y).sum().item()
            n_inputs += len(x)
            if n_iter is not None and it == n_iter:
                break
        acc = correct / n_inputs
    return np.mean(loss_list), acc


def get_opt(params, opt, lr, weight_decay=0, momentum=0):
    if opt == "adam":
        return Adam(params, lr=lr, weight_decay=weight_decay)
    elif opt == "sgd":
        return SGD(params, lr=lr, weight_decay=weight_decay, momentum=momentum)
    else:
        raise ValueError(f"Unknown optimizer {opt}")


def get_sch(sch, opt, epochs):
    if sch == "none" or sch is None:
        return None
    elif sch == "cosine":

        return CosineAnnealingLR(opt, epochs, last_epoch=-1)
    else:
        raise ValueError(f"Unknown scheduler {sch}")


def get_attack_exp_path(args):
    if args.exp_name is not None:
        exp_name = args.exp_name
    else:
        if args.defense == "dp":
            exp_name = f"C_{args.C}_sigma_{args.sigma}_decor_{args.decor}_t_{args.T}_tv_{args.tv}_nv_{args.nv}_l1_{args.l1}"
        else:
            exp_name = (
                f"decor_{args.decor}_t_{args.T}_tv_{args.tv}_nv_{args.nv}_l1_{args.l1}"
            )
    if args.attack == "gm":
        attack_name = "gm"
    else:
        attack_name = f"cp_{args.fi_method}"

    if args.fl_alg == "fedavg":
        exp_name = exp_name + "_uia"

    exp_path = f"{exp_path_base}/{args.ds}/{args.model}/attack/{attack_name}/{args.defense}/{exp_name}"
    return exp_path
