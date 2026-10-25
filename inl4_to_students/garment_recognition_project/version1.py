#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Image Analysis, Assignment 4: version 1 of the garment classification system (= the system from Assignment 3, unchanged).

im2segment        from Assignment 1 (brightness threshold + 8-connected components)
segment2feature   from Assignment 2 (10 translation- and scale-invariant shape features)
class_train       trains a k-nearest-neighbour classifier on garmentsegmentdata.npy
feature2class     classifies one feature vector (10 classes)
"""

import os
from collections import deque

import numpy as np
from scipy import ndimage

from benchmarking.benchmark_assignment3 import benchmark_assignment3

THISDIR = os.path.dirname(os.path.realpath(__file__))


# --------------------------------------------------------------- segmentation
def find_connected_components(mask):
    """Label the 8-connected components of the True pixels in a binary mask.

    Scans column by column so components come out ordered left to right,
    which is the order the benchmark expects.
    """
    n, m = mask.shape
    labels = np.zeros(mask.shape, dtype=int)
    current_label = 0
    offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1),
               (0, 1), (1, -1), (1, 0), (1, 1)]
    for j in range(m):
        for i in range(n):
            if mask[i, j] and labels[i, j] == 0:
                current_label += 1
                queue = deque([(i, j)])
                labels[i, j] = current_label
                while queue:
                    ci, cj = queue.popleft()
                    for di, dj in offsets:
                        ni, nj = ci + di, cj + dj
                        if (0 <= ni < n and 0 <= nj < m
                                and mask[ni, nj] and labels[ni, nj] == 0):
                            labels[ni, nj] = current_label
                            queue.append((ni, nj))
    return labels, current_label


def im2segment(im, background_threshold=20, min_area=15):
    """Segment garments laid out on a dark background (from Assignment 1)."""
    labels, num = find_connected_components(im > background_threshold)
    return [labels == k for k in range(1, num + 1)
            if np.sum(labels == k) >= min_area]


# ------------------------------------------------------------------- features
FEATURE_NAMES = ["aspect", "extent", "spread_x", "spread_y", "tilt",
                 "holes", "compactness", "v_centre", "width_25", "width_75"]


def segment2feature(Si):
    """Return a 10x1 shape feature vector for one segment (from Assignment 2).

    aspect, extent, spread_x, spread_y, tilt, holes, compactness, v_centre,
    width_25, width_75. All are computed from the cropped bounding box or from
    central moments, so all are translation invariant, and they are normalised
    so that they do not depend on the size of the garment.
    """
    S = (np.asarray(Si) > 0).astype(np.uint8)
    ys, xs = np.nonzero(S)
    if ys.size == 0:                       # empty segment: return zeros
        return np.zeros((len(FEATURE_NAMES), 1))

    B = S[ys.min():ys.max() + 1, xs.min():xs.max() + 1]   # crop to bounding box
    h, w = B.shape
    area = float(B.sum())

    aspect = h / (h + w)
    extent = area / (h * w)

    yy, xx = np.nonzero(B)
    cy, cx = yy.mean(), xx.mean()
    dy, dx = yy - cy, xx - cx
    mu20 = (dx ** 2).sum() / area
    mu02 = (dy ** 2).sum() / area
    mu11 = (dx * dy).sum() / area
    scale = np.sqrt(area)
    spread_x = np.sqrt(mu20) / scale
    spread_y = np.sqrt(mu02) / scale
    tilt = mu11 / np.sqrt(mu20 * mu02 + 1e-9)

    padded = np.pad(B, 1)
    _, n_background = ndimage.label(1 - padded)
    holes = min(n_background - 1, 2) / 2

    cross = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], bool)
    eroded = ndimage.binary_erosion(padded, cross)
    perimeter = float((padded & ~eroded).sum())
    compactness = 4 * np.pi * area / perimeter ** 2 if perimeter else 0.0

    v_centre = cy / (h - 1) if h > 1 else 0.5

    row_widths = B.sum(1).astype(float)
    widest = row_widths.max()
    width_25 = row_widths[int(0.25 * (h - 1))] / widest
    width_75 = row_widths[int(0.75 * (h - 1))] / widest

    return np.array([[aspect], [extent], [spread_x], [spread_y], [tilt],
                     [holes], [compactness], [v_centre],
                     [width_25], [width_75]], dtype=float)


# ----------------------------------------------------------------- classifier
K_CANDIDATES = [1, 3, 5, 7, 9, 11, 13, 15]     # odd k, as in Task 1


def vote(neighbour_labels):
    """Majority vote among labels sorted from nearest to farthest neighbour.

    With 10 classes a tie is possible even for odd k; a tie is broken by the
    tied class that has the nearest neighbour.
    """
    counts = np.bincount(neighbour_labels, minlength=10)
    tied = np.flatnonzero(counts == counts.max())
    for label in neighbour_labels:
        if label in tied:
            return int(label)


def class_train(X, Y):
    """k-NN training: store the training data and choose k by leave-one-out.

    X: n x d matrix with one feature vector per row, Y: the n class labels.
    Same method as Task 1, extended from two classes to ten.
    """
    X_train = np.asarray(X, dtype=float)
    Y_train = np.asarray(Y).ravel().astype(int)

    sq = np.sum(X_train ** 2, axis=1)
    D = sq[:, None] + sq[None, :] - 2 * X_train @ X_train.T   # squared distances
    np.fill_diagonal(D, np.inf)         # a point may not be its own neighbour
    order = np.argsort(D, axis=1)
    loo_errors = []
    for k in K_CANDIDATES:
        wrong = sum(vote(Y_train[order[i, :k]]) != Y_train[i]
                    for i in range(len(Y_train)))
        loo_errors.append(wrong / len(Y_train))
    k = K_CANDIDATES[int(np.argmin(loo_errors))]       # ties -> smallest k

    classification_data = (X_train, Y_train, k, loo_errors)
    return classification_data


def feature2class(x, classification_data):
    """Return the class 0..9 for the feature vector x."""
    X_train, Y_train, k, _ = classification_data
    x = np.asarray(x, dtype=float).ravel()
    d2 = np.sum((X_train - x) ** 2, axis=1)       # squared distances
    nearest = np.argsort(d2)[:k]
    return vote(Y_train[nearest])


def train_classifier():
    """Compute features of the 1000 training segments and train on them."""
    segments = np.load(os.path.join(THISDIR, 'garmentsegmentdata.npy'))
    labels = np.load(os.path.join(THISDIR, 'garmentsegmentgt.npy')).ravel()
    X = np.array([segment2feature(s).ravel() for s in segments])
    return class_train(X, labels)


if __name__ == "__main__":

    classification_data = train_classifier()
    print('chosen k =', classification_data[2],
          ' leave-one-out error per k =', np.round(classification_data[3], 3))

    # Debug modes for the benchmark:
    # 0 with no plots
    # 1 with some plots
    # 2 with the most plots (use if the results are bad)
    mode = 0

    for name in ['short1', 'short2', 'home1', 'home2', 'home3']:
        datadir = os.path.join(THISDIR, 'datasets', name)
        hitrate, confmat, allres, alljs, alljfg, allX, allY = benchmark_assignment3(
            im2segment, segment2feature, feature2class, classification_data,
            datadir, mode)
        print(name + ': Hitrate = ' + str(hitrate * 100) + '%')
