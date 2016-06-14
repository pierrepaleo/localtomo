#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2016, European Synchrotron Radiation Facility
# Main author: Pierre Paleo <pierre.paleo@esrf.fr>
#
from __future__ import division
import numpy as np
from math import sqrt
from utils import dot, norm2sq, norm1

def power_method(K, Kadj, data, n_it=10):
    '''
    Calculates the norm of operator K
    i.e the sqrt of the largest eigenvalue of K^T*K
        ||K|| = sqrt(lambda_max(K^T*K))

    K : forward operator
    Kadj : backward operator (adjoint of K)
    data : initial data
    '''
    x = np.copy(Kadj(data)) # Copy in case of Kadj = Id
    for k in range(0, n_it):
        x = Kadj(K(x))
        s = sqrt(norm2sq(x))
        x /= s
    return sqrt(s)




def proj_Omega(g, g0, mask):
    res = np.copy(g)
    res[mask] = np.copy(g0[mask]) # constraint

    #~ from utils import ims
    #~ ims([res, g, g0, mask])

    return res



def fista(data, K, Kadj, g0, mask, Lip=None, n_it=100, return_all=True):
    """
    Fast Iterative Shrinkage-Thresholding Algorithm
    (here Nesterov projected gradient) for minimizing
        argmin_g || K g - d ||_2^2  + i_Omega(g)

    data: data term ("d")
    K : forward operator
    Kadj: Adjoint of K
    g0, g0mask: arrays implementing the set Omega. The constraint is applied as g[mask] = g0[mask]
    Lip: Largest eigenvalue of Kadj*K
    n_it: number of iterations
    return_all: if True, both result and cost function decay are returned

    """

    if Lip is None:
        print("Warn: fista_l1(): Lipschitz constant not provided, computing it with 20 iterations")
        Lip = power_method(K, Kadj, data, 20)**2 * 1.2
        print("Lip = %e" % Lip)

    if return_all: en = np.zeros(n_it)
    g = np.zeros_like(Kadj(data))
    y = np.zeros_like(g)
    for k in range(0, n_it):

        #~ from utils import ims
        #~ if (k % 100) == 0: ims(g)

        grad_y = Kadj(K(y) - data)
        g_old = np.copy(g)
        g = y - (1.0/Lip)*grad_y
        g = proj_Omega(g, g0, mask)
        y = g + (k/(k + 2.1))*(g - g_old)
        # Calculate norms
        if (k % 10) == 0:
            if return_all:
                energy = 0.5*norm2sq(K(g)-data)
                en[k] = energy
                print("[%d] : energy %e" % (k, energy))
            else: print("Iteration %d" % k)

    if return_all: return en, g
    else: return g



















def _soft_thresh(x, beta):
    return np.maximum(np.abs(x)-beta, 0)*np.sign(x)


def fista_l1(data, K, Kadj, Lambda, Lip=None, n_it=100, return_all=True):
    '''
    Beck-Teboulle's forward-backward algorithm to minimize the objective function
        ||K*x - d||_2^2 + Lambda*||x||_1
    When K is a linear operators.

    K : forward operator
    Kadj : backward operator
    Lambda : weight of the regularization (the higher Lambda, the more sparse is the solution in the H domain)
    Lip : largest eigenvalue of Kadj*K
    n_it : number of iterations
    return_all: if True, an array containing the values of the objective function will be returned
    '''

    if Lip is None:
        print("Warn: fista_l1(): Lipschitz constant not provided, computing it with 20 iterations")
        Lip = power_method(K, Kadj, data, 20)**2 * 1.2
        print("Lip = %e" % Lip)

    if return_all: en = np.zeros(n_it)
    x = np.zeros_like(Kadj(data))
    y = np.zeros_like(x)
    for k in range(0, n_it):
        grad_y = Kadj(K(y) - data)
        x_old = x
        w = y - (1.0/Lip)*grad_y
        w = _soft_thresh(w, Lambda/Lip)
        x = w
        y = x + (k/(k+10.1))*(x - x_old)
        # Calculate norms
        if return_all:
            fidelity = 0.5*norm2sq(K(x)-data)
            l1 = norm1(w)
            energy = fidelity + Lambda*l1
            en[k] = energy
            if (k%10 == 0):
                print("[%d] : energy %e \t fidelity %e \t L1 %e" % (k, energy, fidelity, l1))
        #~ elif (k%10 == 0): print("Iteration %d" % k)
    if return_all: return en, x
    else: return x


