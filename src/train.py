#!/usr/bin/env python3
#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import time
import torch.nn as nn
import submitit

JOB_NAME = "train"

from utils import (
    get_device,
    test,
    get_opt,
    get_sch,
    train_epoch,
    save_model,
    exp_path_base,
    setup,
    submitit_logdir,
    slurm_partition,
)
from log_utils import TrainLog
from datasets import get_dataloaders, ds_dict
from models import model_class_dict, get_model
import logging


def train(args):
    device = get_device()

    # === Setup ===
    exp_path = f"{exp_path_base}/{args.ds}/{args.model}/"
    logger = setup(exp_path=exp_path, log_file="train.log")
    logger.info(
        f"\nTraining {args.model} model on {args.ds} dataset for {args.epochs} epochs"
    )
    device = get_device()

    # === Data ===
    dl_train, dl_test = get_dataloaders(args.ds, batch_size=args.batch_size)

    # === Model ===
    model = get_model(
        model_name=args.model,
        ds=args.ds,
        h_dim=args.h_dim,
    )
    model = model.to(device)

    # === Opt, Sch, Criterion ===
    opt = get_opt(
        model.parameters(),
        args.opt,
        args.lr,
        weight_decay=args.weight_decay,
        momentum=args.momentum,
    )
    sch = get_sch(args.sch, args.opt, args.epochs)
    criterion = nn.CrossEntropyLoss()

    # === TrainLog ===
    trainLog = TrainLog(
        metrics=["epoch", "train_loss", "test_loss", "train_acc", "test_acc"],
        pkl_file=exp_path + "/train.pickle",
        args=args,
        wandb_tags=[args.ds, args.model, "train"],
        project=args.project,
        dry_run=args.dry_run,
    )

    # === Training ===
    for epoch in range(1, args.epochs + 1):
        s = time.time()
        train_loss, train_acc = train_epoch(
            model,
            dl_train,
            opt,
            criterion,
            disable_pbar=args.disable_pbar,
        )
        if sch:
            sch.step()
        test_loss, test_acc = test(model, dl_test, criterion)
        e = time.time()

        logger.info(
            f"Epoch: {epoch} train_loss: {train_loss:.3f} test_loss:"
            f" {test_loss:.3f} train_acc: {train_acc*100:.2f}% test_acc:"
            f" {test_acc*100:.2f}% time: {e-s:.1f}"
        )
        trainLog.append([epoch, train_loss, test_loss, train_acc, test_acc])

    # === Save results ===
    trainLog.save_to_disk()
    save_model(model, exp_path + "model.pt")


if __name__ == "__main__":
    logger = logging.getLogger()
    start = time.time()
    parser = argparse.ArgumentParser(description="Train")
    parser.add_argument(
        "--model",
        type=str,
        default="fc2",
        help="model name",
        choices=model_class_dict.keys(),
    )
    parser.add_argument(
        "--ds",
        type=str,
        default="cifar10",
        help="dataset",
        choices=ds_dict.keys(),
    )

    parser.add_argument(
        "--opt",
        type=str,
        default="adam",
        choices=["sgd", "adam"],
        help="optimizer",
    )
    parser.add_argument(
        "--sch",
        type=str,
        default="none",
        choices=["none", "cosine"],
        help="scheduler",
    )
    parser.add_argument("--batch_size", type=int, default=128, help="batch size")
    parser.add_argument("--epochs", type=int, default=10, help="epochs")
    parser.add_argument("--h_dim", type=int, default=256, help="hidden dim")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="weight decay")
    parser.add_argument("--momentum", type=float, default=0, help="momentum")

    parser.add_argument(
        "--project",
        type=str,
        default="cpa_test",
        help="wandb project",
        choices=["cpa_paper", "cpa_test"],
    )
    parser.add_argument("--disable_pbar", action="store_true", help="disable pbar")
    parser.add_argument("--dry_run", action="store_true", help="disables wandb logging")

    parser.add_argument("--submitit", action="store_true", help="submit to queue")
    parser.add_argument("--n_cpu", type=int, default=40, help="cpus per task")
    parser.add_argument("--n_gpu", type=int, default=4, help="gpus per node")
    args = parser.parse_args()

    # Submit to cluster
    if args.submitit:
        args.disable_pbar = True
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
        job = executor.submit(train, args)
        print(f"\nScheduled { job }.")
    # Train locally
    else:
        train(args)
        end = time.time()
        runtime = end - start
        logger.info("\nRuntime: {:.2f} s".format(runtime))
