"""Evaluation helper (not part of the course code).

The course benchmark removes the last character of every ground-truth string, but the
.txt files have no trailing newline, so it only scores 4 of the 5 garments in each image.
This script pairs segments and labels in the same way as the benchmark (by order), but
scores all 5 garments. It also measures how good the segmentation is and what the hitrate
would be if the segmentation were perfect, to see which stage causes the errors.
"""
import glob
import os

import matplotlib.pyplot as plt
import numpy as np

DATASETS = ['short1', 'short2', 'home1', 'home2', 'home3']


def jaccard(a, b):
    a, b = a > 0, b > 0
    return np.sum(a & b) / np.sum(a | b)


def evaluate(im2segment, segment2feature, feature2class, classdata, datadir):
    ims = sorted(glob.glob(os.path.join(datadir, '*jpg')))
    segs = sorted(glob.glob(os.path.join(datadir, '*npy')))
    txts = sorted(glob.glob(os.path.join(datadir, '*txt')))
    ok_sys, ok_gt, jacc = [], [], []
    right_count = 0
    for imn, sn, tn in zip(ims, segs, txts):
        im = plt.imread(imn)
        Sgt = np.load(sn)
        gt = [int(c) for c in open(tn).read().strip()]
        Se = im2segment(im)
        right_count += len(Se) == len(Sgt)
        for g, Sg, Si in zip(gt, Sgt, Se):          # same pairing by order as the benchmark
            jacc.append(jaccard(Sg, Si))
            c = int(feature2class(segment2feature(Si), classdata))
            ok_sys.append(c == g)
        for g, Sg in zip(gt, Sgt):                  # perfect segmentation
            c = int(feature2class(segment2feature(Sg), classdata))
            ok_gt.append(c == g)
        ok_sys += [False] * max(0, len(gt) - len(Se))   # garments that were never found count as wrong
    n_img = len(ims)
    return dict(hit_all5=np.sum(ok_sys) / (5 * n_img),
                hit_gt_segments=np.mean(ok_gt),
                mean_jaccard=np.mean(jacc),
                images_with_right_count=right_count / n_img)
