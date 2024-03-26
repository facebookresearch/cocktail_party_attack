#!/usr/bin/env python3
#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import time
import logging
import torch
import warnings
import submitit
from datasets import get_dataloaders

JOB_NAME = "attack"

from utils import (
    get_device,
    wait_for_job,
    setup,
    get_pbar,
    exp_path_base,
    submitit_logdir,
    subsample,
    normalize,
    read_pickle,
    get_attack_exp_path,
    get_grads_file,
    get_updates_file,
    slurm_partition,
)
from eval_utils import get_eval
from datasets import ds_dict, ds_type_dict
from models import model_class_dict, get_model
from gradient_inversion import get_gi

from feature_inversion import get_fi, fi_method_dict
from log_utils import AttackLog


def attack(args):
    warnings.filterwarnings("ignore")
    # === Setup ===
    exp_path = get_attack_exp_path(args)
    logger = setup(exp_path=exp_path, log_file=f"{args.batch_size}.log")
    if args.fl_alg == "fedsgd":
        grads_file = get_grads_file(
            args.ds, args.model, args.defense, args.batch_size, args.C, args.sigma
        )
    else:
        grads_file = get_updates_file(args.ds, args.model, args.batch_size)

    grad_data = read_pickle(grads_file)
    device = get_device()

    # === Model ===
    model = get_model(
        model_name=args.model,
        ds=args.ds,
        h_dim=args.h_dim,
        load_path=f"{exp_path_base}/{args.ds}/{args.model}/model.pt",
    )

    model = model.to(device)
    model.eval()

    # === Attack ===

    attack_log = AttackLog(
        args=args, exp_path=exp_path, project=args.project, model=model
    )
    logger.info(f"\nReading gradient data from {grads_file}")

    for batch in range(attack_log.batch, args.n_batch):
        # get a batch of inputs
        inp_key = "x"
        inp = torch.tensor(grad_data[inp_key][batch], device=device)
        emb = torch.tensor(
            grad_data["z"][batch] if len(grad_data["z"]) > 0 else [], device=device
        )
        labels = torch.tensor(grad_data["y"][batch], device=device)
        grads = [torch.tensor(g, device=device) for g in grad_data["grad"][batch]]

        attack_log.update_batch(batch, inp, emb, grads)

        if attack_log.attack_mode == "gi":
            # === Gradient Inversion ===
            gi = get_gi(args.attack, model, grads, labels, args, batch, attack_log)
            pbar = get_pbar(range(gi.start_iter, args.n_iter), disable=True)
            eval = get_eval(
                inp, emb, model.model_type, ds_type_dict[args.ds], args.attack, fi=False
            )

            for iter in pbar:
                loss_dict = gi.step()
                if (iter % args.n_log == 0) or (iter == args.n_iter - 1):
                    rec_gi = gi.get_rec()
                    eval_avg, eval_batch, rec_gi_reord = eval(rec_gi)
                    attack_log.update_iter(iter, rec_gi_reord, loss_dict, eval_avg)
                    attack_log.checkpoint(gi.get_attack_state())
                    if iter == args.n_iter - 1:
                        attack_log.update_summary(eval_batch)
            rec_emb = torch.tensor([], device=device)
            rec = rec_gi_reord

        # === Feature Inversion ===
        if model.model_type == "conv" and args.n_iter_fi > 0 and args.attack == "cp":
            attack_log.attack_mode = "fi"

            if attack_log.restore_required:
                rec_emb = attack_log.rec_emb.to(device)
            else:
                if args.ideal_emb_rec:
                    rec_emb = emb
                else:
                    rec_emb = rec_gi_reord.abs()
                attack_log.rec_emb = rec_emb

            inp, emb, rec_emb = subsample(
                [inp, emb, rec_emb], n=args.batch_size, n_sample=args.n_sample_fi
            )
            eval_fi = get_eval(
                inp, emb, model.model_type, ds_type_dict[args.ds], args.attack, fi=True
            )

            attack_log.update_batch(batch, inp=inp, fi=True)

            fi = get_fi(
                args.fi_method,
                rec_emb,
                model,
                args,
                grads,
                labels,
                attack_log=attack_log,
            )
            pbar = get_pbar(range(fi.start_iter, args.n_iter_fi), disable=True)
            for iter_fi in pbar:
                loss_dict_fi = fi.step()
                if (iter_fi % args.n_log_fi == 0) or (iter_fi == args.n_iter_fi - 1):
                    rec_fi = fi.get_rec()
                    eval_avg_fi, eval_batch_fi, _ = eval_fi(rec_fi)
                    attack_log.update_iter(
                        iter_fi, rec_fi, loss_dict_fi, eval_avg_fi, fi=True
                    )
                    attack_log.checkpoint(fi.get_attack_state())
                    if iter_fi == args.n_iter_fi - 1:
                        attack_log.update_summary(eval_batch_fi, fi=True)

            rec = rec_fi

        if ds_type_dict[args.ds] == "image":
            inp = normalize(inp, method="ds", ds=args.ds)

        attack_log.update_rec(
            inp.cpu().numpy(),
            emb.cpu().numpy(),
            rec_emb.cpu().numpy(),
            rec.cpu().numpy(),
        )
        attack_log.attack_mode = "gi"  # for next batch

    attack_log.save_to_disk()


if __name__ == "__main__":
    logger = logging.getLogger()

    start = time.time()

    # Model, dataset
    parser = argparse.ArgumentParser(description="Gradient Inversion Attack")
    parser.add_argument(
        "--model",
        type=str,
        default="vgg16",
        help="model name",
        choices=model_class_dict.keys(),
    )
    parser.add_argument("--h_dim", type=int, default=256, help="hidden dim")
    parser.add_argument(
        "--ds",
        type=str,
        default="imagenet",
        help="dataset",
        choices=ds_dict.keys(),
    )
    parser.add_argument("--batch_size", type=int, default=8, help="batch size")

    # Gradient Inversion

    parser.add_argument(
        "--attack", type=str, default="cp", help="attack", choices=["cp", "gm"]
    )
    parser.add_argument(
        "--decor", type=float, default=1, help="decorrelation weight (CPA)"
    )
    parser.add_argument(
        "--tv", type=float, default=0, help="Total Variation prior weight"
    )
    parser.add_argument("--ne", type=float, default=1, help="Neg Entropy")
    parser.add_argument("--l1", type=float, default=0, help="L1 prior")
    parser.add_argument("--nv", type=float, default=0, help="negative value penalty")

    parser.add_argument(
        "--T",
        type=float,
        default=5,
        help="Temperature for cosine similarity when computing decor loss in CPA",
    )
    parser.add_argument(
        "--n_batch",
        type=int,
        default=10,
        help="number of batches of gradients to invert",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=25000,
        help="number of iterations of optimization for attack",
    )
    parser.add_argument(
        "--n_log", type=int, default=1000, help="log frequency (every n iterations)"
    )
    parser.add_argument(
        "--opt",
        type=str,
        default="adam",
        choices=["sgd", "adam"],
        help="optimizer",
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument(
        "--lr_N", type=float, default=1e-6, help="learning rate for Noise"
    )
    parser.add_argument(
        "--sch",
        type=str,
        default="none",
        choices=["none", "cosine"],
        help="scheduler",
    )

    # Feature Inversion

    parser.add_argument(
        "--fi_method",
        type=str,
        default="gm",
        choices=fi_method_dict.keys(),
        help="dip",
    )

    parser.add_argument(
        "--n_iter_fi",
        type=int,
        default=25000,
        help="number of iterations of optimization for attack",
    )
    parser.add_argument(
        "--n_log_fi",
        type=int,
        default=2000,
        help="log frequency for feature inversion (every n iterations)",
    )
    parser.add_argument(
        "--opt_fi",
        type=str,
        default="adam",
        choices=["sgd", "adam"],
        help="optimizer",
    )
    parser.add_argument(
        "--lr_fi", type=float, default=1e-1, help="learning rate for feature inversion"
    )
    parser.add_argument(
        "--sch_fi",
        type=str,
        default="cosine",
        choices=["none", "cosine"],
        help="scheduler",
    )
    parser.add_argument(
        "--n_sample_fi",
        type=int,
        default=-1,
        help="number of samples for performing feature inversion",
    )
    parser.add_argument(
        "--gm",
        type=float,
        default=1,
        help="gradient matching weight for feature inversion",
    )
    parser.add_argument("--fi", type=float, default=1, help="feature inversion weight")
    parser.add_argument(
        "--fl_alg",
        type=str,
        default="fedsgd",
        choices=["fedsgd", "fedavg"],
        help="algorithm for federated learning",
    )

    # DP
    parser.add_argument("--C", type=float, default=1.0, help="Clipping Norm")
    parser.add_argument("--sigma", type=float, default=0.0, help="sigmal for DP noise")

    # Defense
    parser.add_argument(
        "--defense",
        type=str,
        default="nodef",
        help="Defense for gradient sharing (not implemented!!)",
        choices=["nodef", "dp", "clip"],
    )

    # Logging
    parser.add_argument(
        "--exp_name",
        type=str,
        default=None,
        help="custom name for the experiment",
    )

    parser.add_argument(
        "--project",
        type=str,
        default="cpa_test",
        help="wandb project",
        choices=["cpa_paper", "cpa_test"],
    )

    parser.add_argument(
        "--fresh_start",
        action="store_true",
        help="Does not load checkpoint (fresh start)",
    )
    parser.add_argument("--dry_run", action="store_true", help="disables wandb logging")
    parser.add_argument("--submitit", action="store_true", help="submit to slurm")
    parser.add_argument("--n_gpu", type=int, default=1, help="gpus per node")
    parser.add_argument(
        "--timeout", type=int, default=600, help="timeout minutes on gpu"
    )
    parser.add_argument(
        "--wait", action="store_true", help="wait till slurm jobs finish"
    )
    parser.add_argument("--disable_pbar", action="store_true", help="disable pbar")

    # Unrealistic assumptions

    parser.add_argument(
        "--use_labels", action="store_true", help="Assume labels are known"
    )
    parser.add_argument(
        "--ideal_emb_rec",
        action="store_true",
        help="Assume ideal embedding reconstruction",
    )

    args = parser.parse_args()

    if args.n_sample_fi == -1 or args.n_sample_fi > args.batch_size:
        args.n_sample_fi = args.batch_size

    # Submit it
    if args.submitit:
        args.disable_pbar = True
        executor = submitit.AutoExecutor(folder=submitit_logdir)
        executor.update_parameters(
            timeout_min=args.timeout,
            nodes=1,
            slurm_partition=slurm_partition,
            slurm_job_name=JOB_NAME,
            gpus_per_node=args.n_gpu,
            cpus_per_task=20,
            slurm_mem="100G",
        )
        job = executor.submit(attack, args)
        print(f"[{JOB_NAME}] Scheduled { job }.")
        if args.wait == True:
            wait_for_job(job)

    else:
        attack(args)
        end = time.time()
        runtime = end - start
        logger.info(f"\nRuntime: {runtime:.2f} s")
