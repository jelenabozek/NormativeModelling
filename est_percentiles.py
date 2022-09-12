#!/usr/bin/env python

#  This code is released to the public domain.

#  Author: Mark Jenkinson, University of Oxford and University of Adelaide
#  Date: August 2021

#  Neither the University of Oxford, the University of Adelaide, nor
#  any of their employees imply any warranty of usefulness of this software
#  for any purpose, and do not assume any liability for damages,
#  incidental or otherwise, caused by any use of this document.

"""
Command-line tool (and module) used to estimate a normative model (percentile curves) from a set of
data values, or many datasets, using sliding window approaches.
"""

import sys
import argparse

import numpy as np
import pandas as pd
import pyreadr

import normodlib as nm


# ======================================================================== #

# SUPPORT FUNCTIONS

# ======================================================================== #

def pw_interp(x, y, x_c):
    """
    Piecewise linear interpolation

    Args:
        x (np.array): Nsubj x 1 array of x (e.g. age) values, which must be jointly ordered (sorted) based on *y* values
        y (np.array): Nsubj x 1 array of y (e.g. volume) values, sorted
        x_c (float): the critical (requested) percentile value [0,1]

    Returns:
        float: the requested interpolated percentile curve; Nsubj x 1 array
    """
    offset = 1
    if x_c < np.min(x):  # if requested x_c is to the left of all data
        # find a value x[offset] that is greater than x[0] to estimate a slope at left end for extrapolation
        while x[offset] - x[0] < 1e-8 * x[0] and offset < len(x) / 2:
            offset *= 2  # double each time to efficiently expand the range (as many values of x may be equal)
        if (x[offset] - x[0]) < 1e-6:  # If a different value can't be found then assume zero slope
            y = y[0]
        else:  # extrapolate
            y = (y[offset] - y[0]) * (x_c - x[0]) / (x[offset] - x[0]) + y[0]
    elif x_c > np.max(x):  # if requested x_c is to the right of all data
        while x[-1] - x[-1 - offset] < 1e-8 * x[-1] and offset < len(x) / 2:
            offset *= 2   # double range (see above)
        if (x[-1] - x[-1 - offset]) < 1e-6:  # If a different value can't be found then assume zero slope
            y = y[-1]
        else:   # extrapolate
            y = (y[-1] - y[-1 - offset]) * (x_c - x[-1]) / (x[-1] - x[-1 - offset]) + y[-1]
    else:  # requested x_c is inside the data (do standard linear interpolation)
        # find bounding points and then interpolate between them
        idxa = np.where(x <= x_c)[0][-1]
        idxb = np.where(x > x_c)[0][0]
        x_a = x[idxa]
        x_b = x[idxb]
        y_a = y[idxa]
        y_b = y[idxb]
        y = y_a + (y_b - y_a) * (x_c - x_a) / (x_b - x_a)
    return y


def convolve_with_nans(y, kern):
    """
    Perform a 1D convolution where NaNs are ignored (treated as mask=0) and do not create NaNs in the output.

    Args:
        y (np.array): 1D array of data values
        kern (np.array): 1D convolutional kernel (should always have an odd length to facilitate centre selection)

    Returns:
        fullconv (np.array): 1D array of convolved output (same size as y)
    """
    assert (kern.shape[0] % 2) == 1, \
        print(f'convolve_with_nans::kern should have odd length, not length {kern.shape}')
    valid_mask = 1.0 - np.isnan(y) * 1.0  # mult by 1.0 to make numbers not bools
    mode = 'same'
    resnum = np.convolve(np.nan_to_num(y), kern, mode)  # NaNs --> 0
    resden = np.convolve(valid_mask, kern, mode)
    resden = resden + 1.0 * (resden < 1e-10)  # avoid division by zeros
    fullconv = resnum / resden
    return fullconv


def weighted_percentile(ptl, y, weights):
    """
    Calculate weighted percentile.

    Algorithm is this:
    y_s = sort(y)
    w = different weight for each sample (e.g. could come from a Parzen window based on age)
        order of w samples must correspond with the order of samples in y_s (i.e. order based on sort(y))
    cdfw = cumsum(w) / sum(w)
    the pairs (cdfw, y_s) represent the percentile function - piecewise linear
    to calculate the appropriate percentile need to interpolate the piecewise linear function
    e.g. 95th percentile --> a=0.95 --> find cdfw0 and cdfw1  st  cdfw0 < a < cdfw1
    and then interpolate between the corresponding y0 and y1 values

    Args:
        ptl (float): required percentile in [0,1]
        y (np.array): data values that *must* be sorted (1D array; Nx1)
        weights (np.array): weighting/probability value for each value (1D array; Nx1)
            sum of weights should be > 1e-3

    Returns:
        float: value of the requested percentile from data y, using weights
    """
    # Note that y must be ordered (sorted)
    if y.shape[0] < 1:
        return np.nan
    if np.sum(weights) > 1e-3:
        cdfw = np.cumsum(weights) / np.sum(weights)
        y_0 = pw_interp(cdfw, y, ptl)
    else:
        y_0 = np.nan
    return y_0


def smooth_with_nans(x, y, sigma):
    """
    Perform a 1D smoothing (Gaussian convolution) where NaNs are ignored (treated as mask=0)
        and with no NaNs in the output.

    Args:
        x (np.array): 1D array (Nx1) of location values (used to determine the kernel values)
        y (np.array): 1D array (Nx1) of data values
        sigma (np.array): 1D convolutional kernel (should always have an odd length to facilitate centre selection)

    Returns:
        np.array: 1D array (Nx1) of smoothed output (same size as x and y)
    """
    mididx = int(x.shape[0] / 2)
    kern = np.exp(-0.5 * (x - x[mididx]) ** 2 / (sigma ** 2))
    if kern.shape[0] % 2 == 0:
        kern = kern[1:]
    if np.sum(kern) < 1e-6:
        kern = kern * 0.0 + 1.0
    kern /= np.sum(kern)
    y_2 = convolve_with_nans(y, kern)
    return y_2


# ======================================================================== #


def sliding_window_estimate(percvals, x_s, y_s, xrank, ages, parzen, width, postsigma):
    """
    Sliding window normative modelling estimate (curves) for different requested percentiles

    NB: In Nobis et al 2019, a sliding-window approach is used where
    each window includes 10% of the subjects, and then smoothed with "a
    gaussian kernel of 20".  In Supplementary material they
    compare with "windows of fixed age bins (5 years width)".  They
    report 2.5, 5, 10, 25, 50 [and symmetric] percentiles. The curves
    for 2.5th percentile are not very biologically plausible (unlike the
    others, though still smooth, which is by construction, but changing slope
    from shallow to steep to shallow).  Age range on plots is [53,72].
    Assume that specified kernel width is FWHM --> sigma = 8.5 years.

    Args:
        percvals (list or np.array): set of requested percentile values - Nperc x 1
        x_s (np.array): x-axis values (e.g. ages) - N x 1 - where order is based on the sorting of y values
            e.g. x_s, y_s = nm.sort_by_second(x, y)
        y_s (np.array): values to be estimated (e.g. volumes) - Nx1 - and these must be *sorted*
        xrank (np.array): ascending rank of x_s - N x 1 - i.e. in x_s there are xrank[i] values less than x_s[i]
        ages (np.array): age values at which to calculate the resulting curves (centres of histogram bins) - Nage x 1
        parzen (str): type of Parzen window; 'rect' or 'gaussian'
        width (float): the width of the Parzen window in units of x_s (= sigma for Gaussian case)
        postsigma (float): sigma of smoothing to be applied after the initial sliding window estimate

    Returns:
        list of np.arrays: contains the set of percentile curves (each curves is an array - Nage x 1)
    """
    allptls = []
    halfw = width
    if parzen == 'fixedP':
        halfw = int(x_s.shape[0] * width / 100 / 2)
    # loop over percentiles of interests
    for pcval in percvals:
        wptl = np.zeros(ages.shape)
        # loop over age bins
        for idx, x_c in enumerate(ages):
            # calculate weights corresponding to the age centre, x_c, using appropriate Parzen window
            if parzen == 'rect':
                weights = nm.rect((x_s - x_c) / width)
            elif parzen == 'gaussian':
                weights = np.exp(-0.5 * (x_s - x_c) ** 2 / (width ** 2))
            elif parzen == 'fixedP':  # fixed percentage of samples must fit inside rect window
                idxsel0 = (np.abs(x_s - x_c) < 0.51)
                if np.sum(idxsel0) < 0.5:
                    idxsel0[np.argmin(np.abs(x_s - x_c))] = 1
                mididx = int(np.mean(xrank[idxsel0]))
                h_w = halfw
                if mididx - h_w < 0:
                    h_w = mididx
                if mididx + h_w > x_s.shape[0]:
                    h_w = x_s.shape[0] - 1 - mididx
                idxsel = np.logical_and((xrank >= mididx - h_w), (xrank <= mididx + h_w))
                # also include subjects of equal age to the end points but
                # rescale all the subjects at extreme ages to account for the extras
                weights = x_s * 0.0
                weights[idxsel] = 1.0
            else:
                print(f'Unknown Parzen window type: {parzen}')
                sys.exit(1)
            wptl[idx] = weighted_percentile(pcval, y_s, weights)  # calculate requested percentile
        # post smoothing on percentile plots
        wptls = smooth_with_nans(ages, wptl, postsigma)
        # put curve into a list of curves
        allptls += [wptls]
    return allptls


# ======================================================================== #


def fit_normod(mode, x, y, minx, maxx, percvals=None, args=None):
    """
    Fit normative model (currently only sliding windows - 'movingav' - is supported)

    Args:
        mode (str): type of model to use (currently only 'movingav' is supported)
        x (np.array): values on x-axis (e.g. age) ; Nsubj x 1
        y (np.array): values to estimate (e.g. volume) ; Nsubj x 1
        minx (float): minimum x-axis value (e.g. age)
        maxx (float): maximum x-axis value (e.g. age)
        percvals (list or np.array): set of requested percentile values (Nperc x 1)
        args (argparse structure): contains all command-line argument settings

    Returns:
        list of np.arrays: list of (Nperc) percentile curves where each is an Nsubj x 1 array
    """
    percvals = [0.05, 0.95] if percvals is None else percvals
    if mode == "movingav":
        # smoothing params
        width = args.percbinsize
        parzen = args.bintype
        smsig = np.max([args.postsmooth, 1e-2])
        agebin = args.binwidth
        # setup appropriate age bin centres
        ages = np.array(np.arange(minx + agebin / 2, maxx + 0.9 * agebin / 2, agebin))
        # fit the normative model (start with necessary sorting of values, required by sliding_window_estimate)
        x_s, y_s = nm.sort_by_second(x, y)
        xrank = nm.sorted_rank(x_s)
        hwptls = sliding_window_estimate(percvals, x_s, y_s, xrank, ages, parzen, width, smsig)
        # resample (by interpolatation) the estimated percentile curves to get a value for each sample's age
        wptls = []
        for idx, _ in enumerate(hwptls):
            wptls += [np.interp(x, ages, hwptls[idx])]
    else:
        print(f'ERROR::Unrecognised mode of {mode}')
        sys.exit(4)

    return wptls


# ======================================================================== #


def do_work(args):
    """
    Executes the main script that does the high level work, except for arg parsing that is done before.
    Stages are: setup parameters --> load data --> make age histogram --> fit normative model and save outputs
        into a pandas dataframe --> save results with specific format

    Args:
        args (argparse structure): contains all command-line argument information.

    Returns:
        Nothing.
    """

    # Setup parameters

    if args.estpercs:
        percvals = args.estpercs / 100.0
    else:
        percvals = [0.01, 0.02, 0.05, 0.10, 0.50, 0.90, 0.95, 0.98, 0.99]

    # Load simulated data matrix: N rows x M cols
    #  e.g. N = # samples; 1st col = age, next 1000 cols are different simulated sets

    if 'csv' in args.inputfile:
        yall = pd.read_csv(args.inputfile, header=None).to_numpy()
    elif 'rds' in args.inputfile:
        resdf = pyreadr.read_r(args.inputfile)
        yall = resdf[None].to_numpy()
    else:
        print(f'Cannot find either csv or rds files in specified location: {args.inputfile}')
        sys.exit(2)

    # ======================================================================== #

    agevals = np.round(yall[:, 0])
    _, _, _, _, x = nm.hist_age_data(agevals, agebinwidth=args.binwidth)
    minage = np.min(agevals)
    maxage = np.max(agevals)

    # ======================================================================== #

    # create output as pandas dataframe
    #  each row is one sample  (N rows = # samples)
    #  columns names are "age","simdata_1","centile1_sim1","centile2_sim1", ...,
    #    "simdata_2","centile1_sim2","centile2_sim2", etc

    maxiter = yall.shape[1] - 1
    print(f'Maxiter = {maxiter}')

    resnp = np.zeros((agevals.shape[0], 1 + maxiter * (len(percvals) + 1)))
    colnames = []
    colnames += ['age']
    resnp[:, 0] = agevals

    # loop over datasets, fitting the normative model each time and saving outputs into dataframe
    n_c = 1
    for n_y in range(maxiter):
        y = yall[:, n_y + 1]
        colnames += [f'simdata_{n_y + 1}']
        resnp[:, n_c] = y
        n_c += 1
        wptls = fit_normod(args.estmodeltype, x, y, minx=minage, maxx=maxage,
                           percvals=percvals, args=args)
        for idx, w_0 in enumerate(wptls):
            p = percvals[idx]
            cname = f'centile{int(p * 100)}_sim{n_y + 1}'
            colnames += [cname]
            resnp[:, n_c] = w_0
            n_c += 1

    # noinspection PyTypeChecker
    np.savetxt(args.out, resnp, fmt='%9.3f', header=','.join(colnames),
               delimiter=',', comments='')


# ======================================================================== #

# ---- MAIN ---- #

if __name__ == "__main__":

    # ======================================================================== #

    # User parameters

    parser = argparse.ArgumentParser(
        description="Implementation of various normative modelling approaches "
                    + "(for estimating distributions that vary wrt a parameter such as age)")

    parser.add_argument("-i", "--inputfile", type=str, default="",
                        help="name of input data file (csv format, the output "
                             + "of a normative model run on simulated data)", required=True)
    parser.add_argument("-o", "--out", type=str, default="",
                        help="name of output data file (csv format)", required=True)
    parser.add_argument("-p", "--percentile", type=float, default=5.0,
                        help="percentile value (0 to 100) to calculate and report on")
    parser.add_argument("-w", "--binwidth", type=float, default=1.0,
                        help="bin width to use for plotting and percentile estimation")
    parser.add_argument("--percbinsize", type=float, default=5.0,
                        help="bin size for percentile estimator")
    parser.add_argument("--postsmooth", type=float, default=1.0,
                        help="post percentile smoothing in percentile estimator")
    parser.add_argument("--bintype", type=str, default="rect",
                        help="type of bin estimator to use in percentile estimator (rect or gaussian)")
    parser.add_argument("--estpercs", type=float, nargs='+',
                        help="list of percentile values (0 to 100) to calculate and report on")
    parser.add_argument("--nosave", action="store_true",
                        help="do not save simulated values")
    parser.add_argument("--noplots", action="store_true")

    main_args = parser.parse_args()

    main_args.estmodeltype = "movingav"  # currently the only supported type

    # Sanity checking

    if main_args.bintype not in ['rect', 'gaussian', 'fixedP']:
        print(f"Invalid choice for bin type ({main_args.bintype}): must be rect or gaussian")
        sys.exit(1)

    do_work(main_args)
