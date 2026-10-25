#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 14:06:01 2024

@author: magnuso
"""

import numpy as np
import scipy
import scipy.io
from scipy.sparse.csgraph import maximum_flow, breadth_first_order
from scipy import sparse
import matplotlib.pyplot as plt

def edges4connected(height,width,only_one_dir = 0):
    #
    # Generates a 4-connectivity structure for a height*width grid
    #
    # if only_one_dir==1, then there will only be one edge between node i and
    # node j. Otherwise, both i-->j and i<--j will be added.
    #
    
    N = height*width
    I = np.array([])
    J = np.array([])
    
    # connect vertically (down, then up)
    iis = np.delete(np.arange(N),np.arange(height-1,N,height))
    jjs = iis+1;
    if ~only_one_dir:
        I = np.hstack((I,iis,jjs))
        J = np.hstack((J,jjs,iis))
    else:
        I = np.hstack((I,iis))
        J = np.hstack((J,jjs))
    
    # connect horizontally (right, then left)
    
    iis = np.arange(0,N-height)
    jjs = iis+height
    if ~only_one_dir:
        I = np.hstack((I,iis,jjs))
        J = np.hstack((J,jjs,iis))
    else:
        I = np.hstack((I,iis))
        J = np.hstack((J,jjs))
    
    return I,J

def edges8connected(height,width,only_one_dir = 0):
    #
    # Generates a 8-connectivity structure for a height*width grid
    #
    # if only_one_dir==1, then there will only be one edge between node i and
    # node j. Otherwise, both i-->j and i<--j will be added.
    #
    
    N = height*width
    I = np.array([])
    J = np.array([])
    
    # connect vertically (down, then up)
    iis = np.delete(np.arange(N),np.arange(height-1,N,height))
    jjs = iis+1;
    if ~only_one_dir:
        I = np.hstack((I,iis,jjs))
        J = np.hstack((J,jjs,iis))
    else:
        I = np.hstack((I,iis))
        J = np.hstack((J,jjs))
    
    # Diagonal
    jjs = iis+1+height
    if ~only_one_dir:
        I = np.hstack((I,iis,jjs))
        J = np.hstack((J,jjs,iis))
    else:
        I = np.hstack((I,iis))
        J = np.hstack((J,jjs))

    jjs = iis+1-height;
    if ~only_one_dir:
        I = np.hstack((I,iis,jjs))
        J = np.hstack((J,jjs,iis))
    else:
        I = np.hstack((I,iis))
        J = np.hstack((J,jjs))


    ind = (I>=0) & (I<N)
    I = I[ind]
    J = J[ind]
    ind = (J>=0) & (J<N)
    I = I[ind]
    J = J[ind]
    

    # connect horizontally (right, then left)
    
    iis = np.arange(0,N-height)
    jjs = iis+height
    if ~only_one_dir:
        I = np.hstack((I,iis,jjs))
        J = np.hstack((J,jjs,iis))
    else:
        I = np.hstack((I,iis))
        J = np.hstack((J,jjs))
    
    return I,J



def graph_cut_segmentation(im, m_chamber, s_chamber, m_background, s_background, nu):
    # Segments im into chamber (True) and background (False) with a graph cut.
    # nu is the weight of the regularization (the length of the boundary).
    M, N = im.shape
    n = M*N

    # The edge functions number the pixels column by column (pixel (i,j) gets
    # index i + j*M), so the image is flattened with order='F'.
    x = im.flatten(order='F')

    # Data terms: negative log likelihood of a Gaussian without the constant
    cost_chamber = (x - m_chamber)**2 / (2*s_chamber**2)
    cost_background = (x - m_background)**2 / (2*s_background**2)

    # Edges between neighbouring pixels. Cutting one of them costs nu.
    Ie, Je = edges4connected(M, N)
    Ve = nu*np.ones(len(Ie))

    # Node n is the source (chamber side), node n+1 is the sink (background side).
    # If a pixel ends up on the sink side the edge source->pixel is cut, so that
    # edge must cost what it costs to call the pixel background. In the same way
    # the edge pixel->sink costs what it costs to call the pixel chamber.
    Is1 = np.arange(n)
    Js1 = n*np.ones((n,))
    Is2 = n*np.ones((n,))
    Js2 = np.arange(n)
    Vs = cost_background

    It1 = np.arange(n)
    Jt1 = (n+1)*np.ones((n,))
    It2 = (n+1)*np.ones((n,))
    Jt2 = np.arange(n)
    Vt = cost_chamber

    # setup graph, discretize since maxflow algorithm only works on int
    I = np.hstack((Ie, Is1, Is2, It1, It2)).astype(np.int32)
    J = np.hstack((Je, Js1, Js2, Jt1, Jt2)).astype(np.int32)
    V = np.hstack((Ve, Vs, Vs, Vt, Vt))

    # we have real values but multiply with large number and round
    sf = 10000
    V = np.round(V*sf).astype(np.int32)

    F = sparse.coo_array((V, (I, J)), shape=(n+2, n+2)).tocsr()

    # find the maximum flow (it is equal to the minimum cut)
    mf = maximum_flow(F, n, n+1)

    # After the max flow, the pixels that can still be reached from the source
    # through edges with free capacity are on the source side of the cut.
    residual = (F - mf.flow).tocsr()
    residual.data[residual.data < 0] = 0
    residual.eliminate_zeros()
    reached = breadth_first_order(residual, n, directed=True,
                                  return_predecessors=False)
    seg = np.zeros(n+2, dtype=bool)
    seg[reached] = True

    return seg[:n].reshape((M, N), order='F')


if __name__ == '__main__':

    # load data
    data = scipy.io.loadmat('heart_data.mat')
    data_chamber = data['chamber_values'].ravel()
    data_background = data['background_values'].ravel()
    im = data['im']

    # Task 2.1: mean and standard deviation of the two classes
    m_chamber = np.mean(data_chamber)
    s_chamber = np.std(data_chamber)
    m_background = np.mean(data_background)
    s_background = np.std(data_background)
    print(f"chamber:    mean = {m_chamber:.4f}, std = {s_chamber:.4f}")
    print(f"background: mean = {m_background:.4f}, std = {s_background:.4f}")

    # Task 2.2: try different weights nu on the boundary length
    nus = [0, 0.5, 1, 2, 3, 5, 8]
    fig, ax = plt.subplots(2, 4, figsize=(16, 8))
    ax = ax.ravel()
    ax[0].imshow(im, cmap='gray')
    ax[0].set_title('input image')
    for a, nu in zip(ax[1:], nus):
        imseg = graph_cut_segmentation(im, m_chamber, s_chamber,
                                       m_background, s_background, nu)
        print(f"nu = {nu}: {imseg.sum()} chamber pixels")
        a.imshow(im, cmap='gray')
        a.contour(imseg, levels=[0.5], colors='r', linewidths=1)
        a.set_title(f'nu = {nu}')
    for a in ax:
        a.axis('off')
    plt.tight_layout()
    plt.savefig('task2_nu_comparison.png', dpi=80)
    plt.show()

    # Final result with the chosen weight
    nu = 3
    imseg = graph_cut_segmentation(im, m_chamber, s_chamber,
                                   m_background, s_background, nu)
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].imshow(im, cmap='gray')
    ax[0].set_title('input image')
    ax[1].imshow(imseg, cmap='gray')
    ax[1].set_title(f'segmentation (nu = {nu})')
    ax[2].imshow(im, cmap='gray')
    ax[2].contour(imseg, levels=[0.5], colors='r', linewidths=1)
    ax[2].set_title('boundary on the image')
    for a in ax:
        a.axis('off')
    plt.tight_layout()
    plt.savefig('task2_result.png', dpi=100)
    plt.show()
