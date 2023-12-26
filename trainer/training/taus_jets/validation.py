import os
import sys
import json

import torch
import time

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import corner

from scipy.stats import wasserstein_distance

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "postprocessing"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "utils"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "extractor"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "extractor", "taus_NanoAOD_v9"))

from postprocessing import postprocessing
from post_actions import target_dictionary_taus as target_dictionary
from corner_plots import make_corner

from nan_resampling import nan_resampling
from columns import tau_cond as tau_cond_M
from columns import tau_names


def validate(
    test_loader,
    model,
    epoch,
    writer,
    save_dir,
    args,
    device,
):
    if writer is not None:
        save_dir = os.path.join(save_dir, f"./figures/validation@epoch-{epoch}")
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
    else:
        save_dir = None

    times = []
    model.eval()
    # Generate samples
    with torch.no_grad():
        gen = []
        reco = []
        samples = []

        for bid, data in enumerate(test_loader):
            x, y = data[0], data[1]
            inputs_y = y.cuda(device)
            start = time.time()
            x_sampled = model.sample(
                num_samples=1, context=inputs_y.view(-1, args.y_dim)
            )
            t = time.time() - start
            print(f"Objects per second: {len(x_sampled) / t} [Hz]")
            times.append(t)

            x_sampled = x_sampled.cpu().detach().numpy()
            inputs_y = inputs_y.cpu().detach().numpy()
            x = x.cpu().detach().numpy()
            x_sampled = x_sampled.reshape(-1, args.x_dim)
            gen.append(inputs_y)
            reco.append(x)
            samples.append(x_sampled)

    print(f"Average objs/sec: {len(x_sampled)/np.mean(np.array(times))}")

    # Fix cols names to remove M at beginning
    reco_columns = ["Tau_" + x for x in tau_names]
    tau_cond = [var.replace("M", "", 1) for var in tau_cond_M]
    # Making DataFrames

    gen = np.array(gen).reshape((-1, args.y_dim))
    reco = np.array(reco).reshape((-1, args.x_dim))
    samples = np.array(samples).reshape((-1, args.x_dim))

    if np.isnan(samples).any():
        print("RESAMPLING")
    samples = nan_resampling(samples, gen, model, device)

    fullarray = np.concatenate((gen, reco, samples), axis=1)
    full_sim_cols = ["FullSTau_" + x for x in tau_names]
    full_df = pd.DataFrame(
        data=fullarray, columns=tau_cond + full_sim_cols + reco_columns
    )
    full_df.to_pickle(os.path.join(save_dir, "./full_tau_df.pkl"))

    gen = pd.DataFrame(data=full_df[tau_cond].values, columns=tau_cond)
    reco = pd.DataFrame(data=full_df[full_sim_cols].values, columns=reco_columns)
    samples = pd.DataFrame(data=full_df[reco_columns].values, columns=reco_columns)

    # Postprocessing
    # NOTE maybe add saturation here as done in nbd??
    reco = postprocessing(
        reco,
        gen,
        target_dictionary,
        "scale_factors_taus.json",
        saturate_ranges_path=None,
    )

    samples = postprocessing(
        samples,
        gen,
        target_dictionary,
        "scale_factors_taus.json",
        saturate_ranges_path=None,
    )

    # New DataFrame containing FullSim-range saturated samples

    saturated_samples = pd.DataFrame()

    # 1D FlashSim/FullSim comparison

    for column in reco_columns:
        ws = wasserstein_distance(reco[column], samples[column])

        fig, axs = plt.subplots(1, 2, figsize=(9, 4.5), tight_layout=False)
        fig.suptitle(f"{column} comparison")

        # RECO histogram
        _, rangeR, _ = axs[0].hist(
            reco[column], histtype="step", lw=1, bins=100, label="FullSim"
        )

        # Saturation based on FullSim range
        saturated_samples[column] = np.where(
            samples[column] < np.min(rangeR), np.min(rangeR), samples[column]
        )
        saturated_samples[column] = np.where(
            saturated_samples[column] > np.max(rangeR),
            np.max(rangeR),
            saturated_samples[column],
        )

        # Samples histogram
        axs[0].hist(
            saturated_samples[column],
            histtype="step",
            lw=1,
            range=[np.min(rangeR), np.max(rangeR)],
            bins=100,
            label=f"FlashSim, ws={round(ws, 4)}",
        )

        axs[0].legend(frameon=False, loc="upper right")

        # Log-scale comparison

        axs[1].set_yscale("log")
        axs[1].hist(reco[column], histtype="step", lw=1, bins=100)
        axs[1].hist(
            saturated_samples[column],
            histtype="step",
            lw=1,
            range=[np.min(rangeR), np.max(rangeR)],
            bins=100,
        )
        plt.savefig(os.path.join(save_dir, f"{column}.png"))
        writer.add_figure(f"1D_Distributions/{column}", fig, global_step=epoch)
        writer.add_scalar(f"ws/{column}_wasserstein_distance", ws, global_step=epoch)
        plt.close()

    # Return to physical kinematic variables

    for df in [reco, samples, saturated_samples]:
        df["Tau_pt"] = df["Tau_ptRatio"] * gen["GenTau_pt"]
        df["Tau_eta"] = df["Tau_etaMinusGen"] + gen["GenTau_eta"]
        df["Tau_phi"] = df["Tau_phiMinusGen"] + gen["GenTau_phi"]

    # Zoom-in for high ws distributions

    incriminated = [
        ["Tau_pt", [0, 100]],
        ["Tau_eta", [-3, 3]],
        ["Tau_phi", [-3.14, 3.14]],
    ]
    for elm in incriminated:
        column = elm[0]
        rangeR = elm[1]
        inf = rangeR[0]
        sup = rangeR[1]

        full = reco[column].values
        full = np.where(full > sup, sup, full)
        full = np.where(full < inf, inf, full)

        flash = samples[column].values
        flash = np.where(flash > sup, sup, flash)
        flash = np.where(flash < inf, inf, flash)

        ws = wasserstein_distance(full, flash)

        fig, axs = plt.subplots(1, 2, figsize=(9, 4.5), tight_layout=False)
        fig.suptitle(f"{column} comparison")

        axs[0].hist(
            full, histtype="step", lw=1, bins=100, range=rangeR, label="FullSim"
        )
        axs[0].hist(
            flash,
            histtype="step",
            lw=1,
            range=rangeR,
            bins=100,
            label=f"FlashSim, ws={round(ws, 4)}",
        )

        axs[0].legend(frameon=False, loc="upper right")

        axs[1].set_yscale("log")
        axs[1].hist(full, histtype="step", range=rangeR, lw=1, bins=100)
        axs[1].hist(
            flash,
            histtype="step",
            lw=1,
            range=rangeR,
            bins=100,
        )
        plt.savefig(f"{save_dir}/{column}_incriminated.png", format="png")
        writer.add_figure(f"Zoom_in_1D_Distributions/{column}", fig, global_step=epoch)
        plt.close()

    # # Return to physical kinematic variables

    physical = ["Tau_pt", "Tau_eta", "Tau_phi"]

    for column in physical:
        ws = wasserstein_distance(reco[column], samples[column])

        fig, axs = plt.subplots(1, 2, figsize=(9, 4.5), tight_layout=False)
        fig.suptitle(f"{column} comparison")

        # RECO histogram
        _, rangeR, _ = axs[0].hist(
            reco[column], histtype="step", lw=1, bins=100, label="FullSim"
        )

        # Saturation based on FullSim range
        saturated_samples[column] = np.where(
            samples[column] < np.min(rangeR), np.min(rangeR), samples[column]
        )
        saturated_samples[column] = np.where(
            saturated_samples[column] > np.max(rangeR),
            np.max(rangeR),
            saturated_samples[column],
        )

        # Samples histogram
        axs[0].hist(
            saturated_samples[column],
            histtype="step",
            lw=1,
            range=[np.min(rangeR), np.max(rangeR)],
            bins=100,
            label=f"FlashSim, ws={round(ws, 4)}",
        )

        axs[0].legend(frameon=False, loc="upper right")

        # Log-scale comparison

        axs[1].set_yscale("log")
        axs[1].hist(reco[column], histtype="step", lw=1, bins=100)
        axs[1].hist(
            saturated_samples[column],
            histtype="step",
            lw=1,
            range=[np.min(rangeR), np.max(rangeR)],
            bins=100,
        )
        plt.savefig(os.path.join(save_dir, f"{column}.png"), format="png")
        writer.add_figure(f"1D_Distributions/{column}", fig, global_step=epoch)
        writer.add_scalar(f"ws/{column}_wasserstein_distance", ws, global_step=epoch)
        plt.close()


    # Corner plots:

    # Kinematics

    labels = ["Tau_pt", "Tau_eta", "Tau_phi"]

    ranges = [(0, 200), (-4, 4), (-3.2, 3.2)]

    fig = make_corner(reco, saturated_samples, labels, "Kinematics", ranges=ranges)
    plt.savefig(f"{save_dir}/Kinematics_corner.png", format="png")
    writer.add_figure("Corner_plots/Kinematics", fig, global_step=epoch)
