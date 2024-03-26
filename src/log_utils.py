#!/usr/bin/env python3
#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import matplotlib.pyplot as plt
import numpy as np
from datasets import xshape_dict, ds_type_dict
from utils import read_pickle, write_pickle
import wandb
import PIL
import os
import pandas as pd
import logging
from gradient_inversion_utils import attack_name_dict
from torchvision.utils import make_grid
import torchvision.transforms as T
import time
from utils import wandb_entity


def remove_ticks(ax):
    plt.grid(False)
    ax.grid(False)
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)


def get_pil_plot(X, ax_dim=3, max_images=16, ds="cifar10"):
    if X.shape[0] > max_images:
        X = X[:max_images]
    x_shape = [-1] + xshape_dict[ds]
    X = X.view(x_shape)

    X = X.cpu().numpy()
    n = X.shape[0]
    fig, ax = plt.subplots(1, n, figsize=(ax_dim * n, ax_dim))
    for i in range(n):
        img = np.rollaxis(X[i], 0, 3)
        img = (img - img.min()) / (img.max() - img.min())
        if n == 1:
            axi = ax
        else:
            axi = ax[i]
        axi.imshow(img)
        remove_ticks(axi)
    fig.canvas.draw()
    pil_image = PIL.Image.frombytes(
        "RGB", fig.canvas.get_width_height(), fig.canvas.tostring_rgb()
    )
    return pil_image


class TrainLog:
    def __init__(
        self,
        metrics,
        pkl_file,
        project="cpa_test",
        entity=wandb_entity,
        args=None,
        wandb_tags=None,
        dry_run=False,
    ):
        if dry_run:
            os.environ["WANDB_MODE"] = "dryrun"
        print("\n")
        wandb.init(entity=entity, project=project, tags=wandb_tags)
        print("\n")

        if args is not None:
            wandb.config.update(args)

        self.metrics = metrics
        self.pkl_file = pkl_file
        d = {metric: [] for metric in metrics}
        self.df = pd.DataFrame(d)
        self.step = 1

    def append(self, values=None, image_dict=None):
        self.df.loc[len(self.df)] = values
        d = {k: v for k, v in zip(self.metrics, values)}
        if image_dict is not None:
            transform = T.ToPILImage()
            for caption, images in image_dict.items():
                img_grid = make_grid(images)
                d[caption] = wandb.Image(transform(img_grid))
        wandb.log(d, step=self.step)
        self.step += 1

    def save_to_disk(self):
        self.df.to_pickle(self.pkl_file)


class AttackLog:
    def __init__(
        self,
        args=None,
        exp_path=None,
        project="cpa_test",
        entity=wandb_entity,
        model=None,
    ):
        self.args = args
        self.batch = 0
        self.iter = 0
        self.attack = args.attack
        self.model = model
        self.model_type = self.model.model_type
        self.ds = args.ds
        self.ds_type = ds_type_dict[args.ds]
        self.attack_mode = "gi"
        self.restore_required = False
        self.attack_state = None
        self.rec_emb = None

        # Set up data structures to log results
        (
            self.tags_iter,
            self.tags_iter_fi,
            self.tags_batch,
            self.tags_batch_fi,
        ) = self.get_tags()

        self.df_iter = pd.DataFrame({tag: [] for tag in self.tags_iter})
        self.df_summary = pd.DataFrame({tag: [] for tag in self.tags_batch})
        self.rec_dict = {"inp": [], "emb": [], "rec_emb": [], "rec": []}

        self.results_iter_pkl_file = f"{exp_path}/{args.batch_size}_iter.pkl"
        self.results_summary_pkl_file = f"{exp_path}/{args.batch_size}_summary.pkl"
        self.ckpt_pkl_file = f"{exp_path}/{args.batch_size}_ckpt.pkl"
        self.rec_file = f"{exp_path}/{args.batch_size}_rec.pkl"

        self.df_iter_fi = self.df_summary_fi = None
        if self.tags_iter_fi is not None:
            self.df_iter_fi = pd.DataFrame({tag: [] for tag in self.tags_iter_fi})
            self.df_summary_fi = pd.DataFrame({tag: [] for tag in self.tags_batch_fi})
            self.results_iter_fi_pkl_file = f"{exp_path}/{args.batch_size}_fi_iter.pkl"
            self.results_summary_fi_pkl_file = (
                f"{exp_path}/{args.batch_size}_fi_summary.pkl"
            )

        self.logger = logging.getLogger()

        self.time_prev = time.time()
        self.wandb_id = None

        if self.args.fresh_start:
            self.restore_required = False
            if os.path.exists(self.ckpt_pkl_file):
                os.remove(self.ckpt_pkl_file)
        else:
            self.restore_required = self.check_preemption()

        # Wandb
        if args.dry_run:
            os.environ["WANDB_MODE"] = "dryrun"
        wandb_run_tags = [args.ds, args.model, self.attack, str(args.batch_size)]
        self.logger.info("\n")
        wandb.init(
            entity=entity, project=project, tags=wandb_run_tags, id=self.wandb_id
        )
        self.logger.info(f"\nWandb: {wandb.run.get_url()}")
        self.logger.info(f"\nexp_path: {exp_path}")

        if self.wandb_id is None:
            self.wandb_id = wandb.run.id
        if args is not None:
            wandb.config.update(args)

    def check_preemption(self):
        if os.path.exists(self.ckpt_pkl_file):
            ckpt = read_pickle(self.ckpt_pkl_file)
            self.wandb_id = ckpt["wandb_id"]
            self.batch = ckpt["batch"]
            self.iter = ckpt["iter"]
            self.attack_mode = ckpt["attack_mode"]
            self.df_iter = ckpt["df_iter"]
            self.df_summary = ckpt["df_summary"]
            self.df_iter_fi = ckpt["df_iter_fi"]
            self.df_summary_fi = ckpt["df_summary_fi"]
            self.rec_dict = ckpt["rec_dict"]
            self.attack_state = ckpt["attack_state"]
            self.rec_emb = ckpt["rec_emb"]
            self.logger.info(
                f"\n--Restoring from Checkpoint batch: {self.batch} iter: {self.iter} attack mode:"
                f" {self.attack_mode}--\n"
            )
            return True
        return False

    def checkpoint(self, attack_state=None):
        ckpt = {
            "wandb_id": self.wandb_id,
            "batch": self.batch,
            "iter": self.iter,
            "attack_mode": self.attack_mode,
            "df_iter": self.df_iter,
            "df_summary": self.df_summary,
            "df_iter_fi": self.df_iter_fi,
            "df_summary_fi": self.df_summary_fi,
            "rec_dict": self.rec_dict,
            "attack_state": attack_state,
            "rec_emb": self.rec_emb,
        }
        write_pickle(ckpt, self.ckpt_pkl_file)

    def update_batch(self, batch, inp=None, emb=None, grads=None, fi=False):
        if self.restore_required:
            return
        self.batch = batch

        if fi:
            self.logger.info(
                "\n===Starting Feature Inversion on batch:"
                f" {batch} with batch_size: {self.args.batch_size} and"
                f" n_iter={self.args.n_iter_fi}===\n"
            )
            d = {"batch": self.batch}
            d["S"] = wandb.Image(get_pil_plot(inp, ds=self.ds))
            wandb.log(d)
        else:
            if self.ds_type == "image":
                d = {"batch": self.batch}
                d["S"] = wandb.Image(get_pil_plot(inp, ds=self.ds))
                if self.attack == "cp" and self.model_type == "fc":
                    X = grads[self.model.attack_index]
                    X = X.reshape([-1] + xshape_dict[self.args.ds])
                    d["X"] = wandb.Image(get_pil_plot(X, ds=self.ds))
                wandb.log(d)
            self.logger.info(
                f"\n===Starting {attack_name_dict[self.attack]} on batch:"
                f" {batch} with batch_size: {self.args.batch_size} and"
                f" n_iter={self.args.n_iter}===\n"
            )

    def update_iter(self, iter, rec, loss_dict, eval_dict, print_val=True, fi=False):

        self.attack_mode = "fi" if fi else "gi"

        self.iter = iter
        time_per_interval = time.time() - self.time_prev
        batch_iter_dict = {
            "batch": self.batch,
            "iter": self.iter,
            "time": int(time_per_interval),
        }

        self.time_prev = time.time()
        combined_dict = {**batch_iter_dict, **loss_dict, **eval_dict}
        values = []
        tags = self.tags_iter_fi if fi else self.tags_iter

        for k in tags:
            values.append(combined_dict[k])

        if fi:
            self.df_iter_fi.loc[len(self.df_iter_fi)] = values
        else:
            self.df_iter.loc[len(self.df_iter)] = values

        d = {k: v for k, v in zip(tags, values)}

        if print_val == True:
            self.logger.info(
                "  ".join("{}: {}".format(k, round(v, 3)) for k, v in d.items())
            )

        if self.ds_type == "image" and (
            self.attack == "gm" or self.model_type == "fc" or fi
        ):
            d["S_hat"] = wandb.Image(get_pil_plot(rec, ds=self.ds))

        wandb.log(d)

    def update_summary(self, eval_dict, fi=False):
        tags = self.tags_batch_fi if fi else self.tags_batch

        tags_nobatch = tags.copy()
        tags_nobatch.remove("batch")
        for vals in zip(*[eval_dict[k] for k in tags_nobatch]):
            vals = [self.batch] + list(vals)
            if fi:
                self.df_summary_fi.loc[len(self.df_summary_fi)] = vals
            else:
                self.df_summary.loc[len(self.df_summary)] = vals

    def save_to_disk(self):
        self.df_iter.to_pickle(self.results_iter_pkl_file)
        self.df_summary.to_pickle(self.results_summary_pkl_file)
        write_pickle(self.rec_dict, self.rec_file)

        if self.tags_iter_fi is not None:
            self.df_iter_fi.to_pickle(self.results_iter_fi_pkl_file)
            self.df_summary_fi.to_pickle(self.results_summary_fi_pkl_file)

        if os.path.exists(self.ckpt_pkl_file):
            os.remove(self.ckpt_pkl_file)

    def get_tags(self):
        tags_iter_fi = None
        tags_batch_fi = None

        if self.args.attack == "gm":
            if self.ds_type == "image":
                tags_iter = [
                    "batch",
                    "iter",
                    "loss",
                    "loss_cd",
                    "loss_tv",
                    "psnr",
                    "ssim",
                    "lpips",
                    "time",
                ]
                tags_batch = ["batch", "psnr", "ssim", "lpips"]
            else:
                tags_iter = ["batch", "iter", "loss", "loss_cd", "cs"]
                tags_batch = ["batch", "cs"]
        elif self.args.attack == "cp":
            if self.ds_type == "image" and self.model_type == "fc":
                tags_iter = [
                    "batch",
                    "iter",
                    "loss",
                    "loss_ne",
                    "loss_decor",
                    "loss_tv",
                    "psnr",
                    "ssim",
                    "lpips",
                    "time",
                ]
                tags_batch = ["batch", "psnr", "ssim", "lpips"]
            elif self.ds_type == "image" and self.model_type == "conv":
                tags_iter = [
                    "batch",
                    "iter",
                    "loss",
                    "loss_ne",
                    "loss_decor",
                    "loss_nv",
                    "loss_l1",
                    "cs",
                    "time",
                ]
                tags_iter_fi = [
                    "batch",
                    "iter",
                    "loss",
                    "loss_fi",
                    "loss_tv",
                    "loss_gm",
                    "psnr",
                    "ssim",
                    "lpips",
                    "time",
                ]
                tags_batch = ["batch", "cs"]
                tags_batch_fi = ["batch", "psnr", "ssim", "lpips"]
            else:  # text
                tags_iter = [
                    "batch",
                    "iter",
                    "loss",
                    "loss_ne",
                    "loss_decor",
                    "cs",
                    "time",
                ]
                tags_iter_fi = [
                    "batch",
                    "iter",
                    "loss",
                    "time",
                ]
                tags_batch = ["batch", "cs"]
                tags_batch_fi = ["batch"]
        return tags_iter, tags_iter_fi, tags_batch, tags_batch_fi

    def update_rec(self, inp, emb, rec_emb, rec):
        self.rec_dict["inp"].append(inp)
        self.rec_dict["emb"].append(emb)
        self.rec_dict["rec_emb"].append(rec_emb)
        self.rec_dict["rec"].append(rec)
