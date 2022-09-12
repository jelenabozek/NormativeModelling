#!/usr/bin/env python

#  This code is released to the public domain.

#  Author: Mark Jenkinson, University of Oxford and University of Adelaide
#  Date: August 2021

#  Neither the University of Oxford, the University of Adelaide, nor
#  any of their employees imply any warranty of usefulness of this software
#  for any purpose, and do not assume any liability for damages,
#  incidental or otherwise, caused by any use of this document.

"""
A set of functions to be used for normative modelling command-line tools
"""

import sys
import numpy as np
from scipy.stats import norm

# stop-pylint-annoying-warnings-for-functions-with--pylint: disable=unused-variable


def sigmoid(x) -> float or np.array:
    """Return sigmoid = 1/(1+exp(x))."""
    return 1 / (1 + np.exp(-x))


def rect(x) -> float or np.array:
    """Return rect/box function: 1 if |x|<=0.5 else 0."""
    return 1*(abs(x) <= 0.5)


def sort_by_second(x_1, x_2) -> tuple:
    """reorders x_1 according to the order obtained by sorting x_2

    Args:
        x_1 (1D array | list): of any numerical values
        x_2 (1D array | list): of any numerical values

    Returns:
        tuple of sorted np.arrays as (x_1_sorted, x_2_sorted)
    """
    idx2 = np.argsort(x_2, axis=0)
    x2s = x_2[idx2]
    x1s = x_1[idx2]
    return x1s, x2s


def sorted_rank(x):
    """
    For a 1D vector, get the ascending rank, such that x[i] is the rank[i]'th ordered value
    e.g. if x[3] is the minimum value then rank[3]=0, and if x[7] is the next biggest,
         rank[7]=1, etc

    Args:
        x (np.array): of numerical (float|int) values

    Returns:
        rank (np.array): of rank for each corresponding place in x
    """
    rank = x*0
    idx = np.argsort(x)
    # x_sorted[n] = x[idx[n]] and we want rank[idx[n]] = n
    for n in range(x.shape[0]):
        rank[idx[n]] = n
    return rank


def mapdata(z, dist_params):
    """
    Map standard Gaussian (mu=0, sigma=1 everywhere) to one with variable mu and sigma

    Args:
        z (np.array): 1D set of z values (typically a function of x)
        dist_params (sequence of np.arrays or float): [0] = mean; [1] = sigma
                both mean and sigma can be arrays, same size as z (i.e. fns of x)
                or they can be scalars

    Returns:
        y (np.array): values after mapping (same size as z)
    """
    sigma = dist_params[1]
    mu = dist_params[0]
    y = sigma*z + mu
    return y


def calc_perc_curves(dist_params, percvalues):
    """
    Calculate ground truth percentile curves (e.g. vols for each sample's age)

    Args:
        dist_params (sequence of np.arrays or float): [0] = mean; [1] = sigma
                both mean and sigma are 1D arrays, Nsubj x 1
        percvalues (list or np.array): Nperc percentile values for desired curves, each value [0,1]

    Returns:
        list of percentile curves: Nperc (list) of Nsubj x 1
    """
    p_curve = []
    for ptl in percvalues:
        g_pv = norm.ppf(ptl)
        p_curve += [ mapdata(g_pv, dist_params) ]
    return p_curve


def makesampledata(dist_params):
    """
    Generate random samples from a simple normal random variable value and pass them through a mapping function

    Args:
        dist_params (sequence of np.arrays or float): [0] = mean; [1] = sigma
                both mean and sigma are 1D arrays, Nsubj x 1

    Returns:
        y (np.array): Nsubj x 1 ; simulated set of values
    """
    zerovec = 0*dist_params[0]
    y_0 = np.random.normal(zerovec, zerovec+1.0)  # N(0,1)
    y = np.copy(y_0)
    y = mapdata(y, dist_params)
    return y


def conv2perc(vals, mu, sigma):
    """
    Convert raw values (e.g. volumes) to percentile values

    Args:
        vals (np.array or float): raw value or values (e.g. volumes)
        mu (np.array or float): mean value or values (latter must match size of argument 'vals')
        sigma (np.array or float): standard deviation value or values (latter must match size of argument 'vals')

    Returns:
        percentile values: same size as argument 'vals'
    """
    z = (vals-mu)/sigma
    percvalues = norm.cdf(z)
    return percvalues


def hist_age_data(allages, agebinwidth=1):
    """
    Create age histogram information

    Args:
        allages (np.array): Nsubj x 1 ; set of age values
        agebinwidth (float): width of age histogram bins

    Returns:
        binages (np.array): nagebins x 1 ; age values at centres of histogram bins
        binedges (np.array): nagebins+1 x 1 ; age values at edges of histogram bins
        agebinwidth (float): width of histogram bins (typically a copy of input parameter, but could be changed)
        nagebins (int): number of age histogram bins
        agevals (np.array): Nsubj x 1 ; set of age values
    """
    agevals = np.round(allages)  # for repeatability, round to nearest year (for paper there was little effect)
    minage = np.min(agevals)
    maxage = np.max(agevals)
    print(f'Min and max ages are: {minage} and {maxage}')

    # Setup age histogram bins
    binages = np.array(np.arange(minage, maxage+1, agebinwidth))  # centres of bins: nagebins x 1
    nagebins = binages.shape[0]
    binedges = np.array(np.arange(minage-1, maxage+1, agebinwidth)) + 0.5*agebinwidth  # nagebins+1 x 1
    print(f'Age bin centres are {binages}')

    return binages, binedges, agebinwidth, nagebins, agevals


# Modelling functions

def model(theta, x, modeltype=None):
    """
    Creates ground truth model

    Args:
        theta (np.array): is a vector of parameters used to define the ground truth
        x (np.array): a vector of values (e.g. age) that the ground truth is a function of
        modeltype (str): name of model type (linear, poly, nonlin) which specifies how to interpret theta

    Returns:
        returns a list of two arrays (mu and sigma; each Nsubj x 1) that effectively represent the distribution
        parameters for each separate value of x, since we want to model distributions changing with x
    """
    if modeltype == 'linear':
        mu = theta[0] + theta[1]*x
        sigma = theta[2]
        # if only 3 elements then constant variance
        if len(theta) > 3:
            sigma += theta[3]*x
        retval = [mu, sigma]
    elif modeltype == 'poly':
        nord = int(len(theta)/2)
        mu = theta[0]
        for nidx in range(nord-1):
            mu += theta[nidx+1]*x**(nidx+1)
        sigma = theta[nord]
        for nidx in range(nord-1):
            sigma += theta[nord+nidx+1]*x**(nidx+1)
        #print(('Pre-stack size is ',mu.shape))
        retval = [mu, sigma]
    elif modeltype == 'nonlin':
        # specialised simulation function, not used for estimation
        mu = sigmoid((x - theta[0])*theta[1])*((x-theta[0])*theta[3]) \
            + (x-theta[0])*theta[2] + theta[4]
        sigma = sigmoid((x - theta[5])*theta[6])*((x-theta[0])*theta[8]) \
            + (x-theta[0])*theta[7] + theta[9]
        #print(('Pre-stack size is ',mu.shape))
        retval = [mu, sigma]
    else:
        print(f'Cannot find model type = {modeltype}')
        sys.exit(1)
    return retval


# ============================================================================= #

# Shared settings

truetheta_nonlin = [ 65.0, 0.1, 20.0, -70.0, 6000.0, 65.0, 0.1, 1.0, 5.0, 200.0 ]  # nonlin
truetheta_poly = [6000.0, -4.0, -0.7, 400.0, 0.2, 0.04]   # poly
truetheta_linear = [5000, -7.0, 300, 2.8]  # linear, linear var
truetheta_linearconstvar = [5000, -7.0, 300, 0]  # linear, const var

percvals = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
