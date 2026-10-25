#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 10:51:35 2024

@author: elinarosato
"""


import os

import numpy as np
import matplotlib.pyplot as plt
import flip_evaluator as flip
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

# input and output images are in the assets folder next to this script
ASSETS = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'assets')



def ransac_color_matrix(f, g, n_iter=200, threshold=0.1, seed=0):
    # f, g: (N,3) arrays, one RGB pixel per row, with g = A f for good pixels.
    # Returns the 3x3 matrix A and a boolean inlier mask of length N.
    rng = np.random.default_rng(seed)
    N = f.shape[0]
    best_A, best_count = None, 0

    for _ in range(n_iter):
        # 3 pixel pairs give 9 equations for the 9 unknowns in A
        idx = rng.choice(N, 3, replace=False)
        F = f[idx].T  # columns are the sampled f-colors
        G = g[idx].T  # columns are the sampled g-colors
        if np.linalg.cond(F) > 1e3:  # singular, e.g. a black pixel: skip
            continue
        A = G @ np.linalg.inv(F)

        # f @ A.T is A f for every pixel at once
        err = np.linalg.norm(g - f @ A.T, axis=1)
        count = np.sum(err < threshold)
        if count > best_count:
            best_A, best_count = A, count

    # refit with least squares on all inliers of the best candidate
    inliers = np.linalg.norm(g - f @ best_A.T, axis=1) < threshold
    At, _, _, _ = np.linalg.lstsq(f[inliers], g[inliers], rcond=None)
    A = At.T
    inliers = np.linalg.norm(g - f @ A.T, axis=1) < threshold
    return A, inliers


def compute_metrics(ref, test, mask):
    # ref: original f, test: image to evaluate, both (H,W,3) floats in [0,1]
    # mask: (H,W) boolean, True for uncorrupted pixels
    # Returns {metric: (mean over whole image, mean over mask only)}

    # PSNR: from the mean squared error
    mse_all = np.mean((ref - test) ** 2)
    mse_mask = np.mean((ref[mask] - test[mask]) ** 2)
    psnr_all = 10 * np.log10(1 / mse_all)
    psnr_mask = 10 * np.log10(1 / mse_mask)

    # SSIM: full=True also returns the per-pixel map, shape (H,W,3)
    ssim_all, ssim_map = ssim(ref, test, channel_axis=2, data_range=1.0, full=True)
    ssim_mask = ssim_map[mask].mean()

    # FLIP: reference first, then test image. Gives per-pixel error map
    flip_map, flip_all, _ = flip.evaluate(ref, test, "LDR")
    flip_mask = np.squeeze(flip_map)[mask].mean()

    return {"PSNR": (psnr_all, psnr_mask),
            "SSIM": (ssim_all, ssim_mask),
            "FLIP": (flip_all, flip_mask)}


if __name__ == "__main__":
    # load data (plt.imread already gives floats in [0,1] for png files)

    g_im = plt.imread(os.path.join(ASSETS, 'task1_color.png'))[..., :3].astype(np.float32)
    f_im = plt.imread(os.path.join(ASSETS, 'task1_org.png'))[..., :3].astype(np.float32)

    f = f_im.reshape(-1, 3).astype(np.float64)
    g = g_im.reshape(-1, 3).astype(np.float64)

    A, inliers = ransac_color_matrix(f, g)
    print("Estimated A:\n", A)
    print(f"Inliers: {inliers.sum()} of {len(inliers)} ({inliers.mean()*100:.1f}%)")

    # estimate the original image: f_hat = A^-1 g for every pixel
    print("det(A) =", np.linalg.det(A))
    A_inv = np.linalg.inv(A)
    f_hat = np.clip(g @ A_inv.T, 0, 1)
    f_hat_im = f_hat.reshape(g_im.shape).astype(np.float32)

    plt.imsave(os.path.join(ASSETS, 'task1_corrected.png'), f_hat_im)

    fig, ax = plt.subplots(1, 3, figsize=(18, 4))
    for a, im, title in zip(ax, [g_im, f_hat_im, f_im],
                            ['g (color cast)', 'f_hat (corrected)', 'f (original)']):
        a.imshow(im)
        a.set_title(title)
        a.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(ASSETS, 'task1_comparison.png'), dpi=100)
    plt.show()

    # error metrics. Uncorrupted pixels = where f is not black
    mask = f_im.sum(axis=2) > 0
    print(f"Uncorrupted pixels: {mask.mean()*100:.1f}%")

    res_hat = compute_metrics(f_im, f_hat_im, mask)
    res_g = compute_metrics(f_im, g_im, mask)

    print(f"{'':12s}{'whole image':>14s}{'uncorrupted':>14s}")
    for name, res in [("f_hat vs f", res_hat), ("g vs f", res_g)]:
        for metric, (a, m) in res.items():
            print(f"{name:12s}{metric:6s}{a:8.4f}{m:14.4f}")


    
  
    

