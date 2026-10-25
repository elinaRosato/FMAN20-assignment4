#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Image Analysis, Assignment 4: version 2 of the garment classification system.

im2segment        threshold + cut the image at the 4 widest empty gaps between garments
segment2feature   the silhouette of the garment, scaled to 20x20 (aspect ratio kept)
feature2class     support vector machine (RBF kernel) trained on garmentsegmentdata.npy
"""

import os

import numpy as np
from scipy import ndimage
from skimage.transform import resize
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from A4.garment_recognition_project.benchmarking.benchmark_assignment3 import benchmark_assignment3

THISDIR = os.path.dirname(os.path.realpath(__file__))

N_GARMENTS = 5          # every image contains 5 garments in a row
THRESHOLD = 15          # pixels brighter than this belong to a garment
SIZE = 20               # the silhouette is scaled to SIZE x SIZE


def im2segment(im, threshold=THRESHOLD, n_garments=N_GARMENTS):
    # return a list of true/false images of the same size as im, ordered from left to right
    fg = im > threshold

    # remove tiny specks (jpeg noise), then fill holes inside the garments
    labels, n = ndimage.label(fg, structure=np.ones((3, 3)))
    sizes = ndimage.sum(fg, labels, range(1, n + 1))
    fg = np.isin(labels, 1 + np.flatnonzero(sizes >= 4))
    fg = ndimage.binary_fill_holes(fg)

    # columns that contain garment pixels, and the empty gaps between them
    used = np.flatnonzero(fg.any(axis=0))
    if used.size == 0:
        return []
    gaps = []                                   # (width, first empty column, next used column)
    for a, b in zip(used[:-1], used[1:]):
        if b - a > 1:
            gaps.append((b - a - 1, a + 1, b))

    if len(gaps) >= n_garments - 1:
        # the garments are separated by the widest gaps
        widest = sorted(gaps, key=lambda g: (-g[0], g[1]))[:n_garments - 1]
        widest.sort(key=lambda g: g[1])
        starts = [used[0]] + [g[2] for g in widest]
        ends = [g[1] for g in widest] + [used[-1] + 1]
        S = []
        for a, b in zip(starts, ends):
            m = np.zeros(fg.shape, dtype=bool)
            m[:, a:b] = fg[:, a:b]
            S.append(m)
        return S

    # fewer gaps than expected (garments that touch): use connected components instead
    labels, n = ndimage.label(fg, structure=np.ones((3, 3)))
    S = [labels == k for k in range(1, n + 1) if np.sum(labels == k) >= 15]
    return sorted(S, key=lambda m: np.flatnonzero(m.any(axis=0))[0])


def segment2feature(Si):
    # return a feature vector (array with size nx1)
    ys, xs = np.nonzero(np.asarray(Si) > 0)
    if ys.size == 0:
        return np.zeros((SIZE * SIZE + 1, 1))
    B = (np.asarray(Si)[ys.min():ys.max() + 1, xs.min():xs.max() + 1] > 0).astype(float)
    h, w = B.shape

    # put the garment in the middle of a square, so the shape is not stretched,
    # and scale it to SIZE x SIZE (this makes the feature independent of the garment size)
    m = max(h, w)
    P = np.zeros((m, m))
    P[(m - h) // 2:(m - h) // 2 + h, (m - w) // 2:(m - w) // 2 + w] = B
    small = resize(P, (SIZE, SIZE), anti_aliasing=True, mode='edge')

    features = np.concatenate([small.ravel(), [h / (h + w)]])
    return features.reshape(-1, 1)


def class_train(X, Y):
    # X: one feature vector per row, Y: the class of each row
    model = make_pipeline(StandardScaler(), SVC(C=10, gamma='scale'))
    model.fit(np.asarray(X, dtype=float), np.asarray(Y).ravel().astype(int))
    return model


def feature2class(x, classification_data):
    # return a classification 0..9 for the feature x
    model = classification_data
    return int(model.predict(np.asarray(x, dtype=float).reshape(1, -1))[0])


def train_classifier():
    # features of the 1000 training segments, then train
    segments = np.load(os.path.join(THISDIR, 'garmentsegmentdata.npy'))
    labels = np.load(os.path.join(THISDIR, 'garmentsegmentgt.npy')).ravel()
    X = np.array([segment2feature(s).ravel() for s in segments])
    return class_train(X, labels)


if __name__ == "__main__":

    # train your classifier or load classification data
    classification_data = train_classifier()

    # Run benchmark on all datasets

    datasets = ['short1', 'short2', 'home1', 'home2', 'home3']

    mode = 0  # debug modes
    # 0 with no plots
    # 1 with some plots

    for ds in datasets:
        datadir = os.path.join(THISDIR, 'datasets', ds)
        hitrate, confmat, allres, alljs, alljfg, allX, allY = benchmark_assignment3(
            im2segment, segment2feature, feature2class, classification_data, datadir, mode)
        print(ds + f', Hitrate = {hitrate*100:0.5}%')
