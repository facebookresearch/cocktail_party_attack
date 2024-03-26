#!/usr/bin/env python3
#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import copy
import time
import torch.nn as nn
import torch
import submitit
import logging
from opacus.grad_sample import GradSampleModule
from torch.utils.data import TensorDataset, DataLoader

JOB_NAME = "collect_grads"

from utils import (
    get_device,
    exp_path_base,
    setup,
    write_pickle,
    submitit_logdir,
    get_updates_file,
    slurm_partition,
)
from datasets import (
    get_dataloaders,
    ds_dict,
)
from models import get_model, model_class_dict


def apply_dp_noise(sample_grad_list, C=1, sigma=0):
    # sample_grad_list: list of sample_wise_gradients for each parameter, where each gradient has dimensionality: batch_size x param_size
    # C: L2 norm for clipping
    # sigma: stddev of noise
    batch_size = sample_grad_list[0].shape[0]
    device = sample_grad_list[0].device
    C = torch.tensor(C, device=device)
    sample_grad_clip_list = []

    with torch.no_grad():

        for b in range(batch_size):
            # concatenate grad from each parameter into a single vector
            g_b_cat = torch.cat([g[b].view(-1) for g in sample_grad_list])

            # Compute scaling factor for "clipping"
            scaling = 1 / torch.maximum(
                torch.tensor(1.0, device=device), g_b_cat.norm() / C
            )

            # scale the gradients of all the parameters using the scaling factor
            for i, g in enumerate(sample_grad_list):
                g_b_clip = g[b] * scaling
                if b == 0:
                    sample_grad_clip_list.append([g_b_clip])
                else:
                    sample_grad_clip_list[i].append(g_b_clip)

        # Compute the sum of sample-wise clipped gradients, add noise, compute mean
        grad_list_dp = []
        for g_batch_clip in sample_grad_clip_list:
            g_sum_clip = torch.sum(torch.stack(g_batch_clip, dim=0), dim=0)
            g_dp = (g_sum_clip / batch_size) + (
                torch.randn(g_sum_clip.shape, device=device) * sigma * C
            )
            grad_list_dp.append(g_dp)

    return grad_list_dp


def get_model_params(model):
    param_list = []
    for name, param in model.named_parameters():
        if "tok_emb" not in name:
            param_list.append(param)
    return param_list


def collect_grads(args):
    n_samples_list = [32, 64, 128, 256]

    # === Setup ===
    exp_path = f"{exp_path_base}/{args.ds}/{args.model}/updates/nodef"
    logger = setup(exp_path=exp_path, log_file="collect_updates.log")
    logger.info(
        f"Collecting gradients with {args.model} model on {args.ds} with dataset sizes:"
        f" {n_samples_list} for {args.n_epochs} epochs"
    )
    device = get_device()

    # === Model ===
    dataparallel = False
    model_orig = get_model(
        model_name=args.model,
        ds=args.ds,
        h_dim=args.h_dim,
        load_path=f"{exp_path_base}/{args.ds}/{args.model}/model.pt",
        dataparallel=dataparallel,
    )
    model_type = model_orig.type

    # Criterion
    criterion = nn.CrossEntropyLoss()

    # === Collect grads ===

    for n_samples in n_samples_list:
        _, dl_test = get_dataloaders(
            args.ds,
            batch_size=n_samples,
            shuffle_test=True,
        )

        logger.info(f"\nCollecting updates with n_samples: {n_samples}")

        data = {"x": [], "y": [], "z": [], "grad": []}

        for i in range(args.n_rounds):
            x, y = iter(dl_test).next()

            ds = TensorDataset(x, y)
            dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True)

            data["x"].append(x.numpy())
            data["y"].append(y.numpy())
            model = copy.deepcopy(model_orig)
            model = model.to(device)
            opt = torch.optim.SGD(model.parameters(), lr=args.lr)

            for e in range(args.n_epochs):
                for x, y in dl:
                    x, y = x.to(device), y.to(device)
                    opt.zero_grad()
                    x, y = x.to(device), y.to(device)
                    if model_type == "conv" or args.ds == "imdb":
                        p, z = model(x=x, return_z=True)
                        data["z"].append(z.detach().cpu().numpy())
                    else:
                        p = model(x)
                    loss = criterion(p, y)
                    loss.backward()
                    opt.step()
            updates = [
                (p0 - p1).detach().numpy()
                for p0, p1 in zip(model.cpu().parameters(), model_orig.parameters())
            ]
            data["grad"].append(updates)

        # Save data
        updates_pkl_file = get_updates_file(args.ds, args.model, n_samples)
        write_pickle(data, updates_pkl_file)
        logger.info(f"Done!\nData saved in {updates_pkl_file}")


if __name__ == "__main__":
    logger = logging.getLogger()
    start = time.time()
    parser = argparse.ArgumentParser(description="Collect Gradients")
    parser.add_argument(
        "--ds",
        type=str,
        default="cifar10",
        help="dataset",
        choices=ds_dict.keys(),
    )

    parser.add_argument(
        "--n_rounds", type=int, default=10, help="number of rounds to collect updates"
    )
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument(
        "--n_epochs", type=int, default=10, help="number of epochs for training"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="fc2",
        help="model name",
        choices=model_class_dict.keys(),
    )
    parser.add_argument("--h_dim", type=int, default=256, help="hidden dim")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")

    parser.add_argument("--submitit", action="store_true", help="submit to queue")
    parser.add_argument("--n_cpu", type=int, default=40, help="cpus per task")
    parser.add_argument("--n_gpu", type=int, default=8, help="gpus per node")

    args = parser.parse_args()

    # Submit it
    if args.submitit:
        executor = submitit.AutoExecutor(folder=submitit_logdir)
        executor.update_parameters(
            timeout_min=600,
            nodes=1,
            slurm_partition=slurm_partition,
            slurm_job_name=JOB_NAME,
            gpus_per_node=args.n_gpu,
            cpus_per_task=args.n_cpu,
            slurm_mem="400G",
        )
        print(JOB_NAME)
        job = executor.submit(collect_grads, args)
        print(f"Scheduled { job }.")
    else:
        collect_grads(args)

    end = time.time()
    runtime = end - start

    logger.info(f"\nRuntime: {runtime:.2f} s")
