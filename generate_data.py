#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  1 08:56:06 2018

@author: hafizimtiaz
"""

import numpy as np

D = 10
K = 2
Ns = 100
S = 3

tmp = np.linspace(D, 1, K)
tmp = np.concatenate((tmp, 0.01 * np.random.random(D-K)))
cov = np.diag(tmp)
mu = np.zeros([D, ])

for s in range(S):
    Xs = np.random.multivariate_normal(mu, cov, Ns).T
    
    # normalize data samples
    nrms = np.sqrt(np.diag(np.dot(Xs.T, Xs)))
    max_nrm = nrms.max()
    mult = (1/max_nrm) * np.diag(np.ones([Ns, ]))
    Xs = np.dot(Xs, mult)
    
    filename = 'value' + str(s) + '.npz'
    np.savez(filename, Xs, mu, cov, K)