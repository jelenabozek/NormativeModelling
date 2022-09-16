#!/usr/bin/env python

# This code is released to the public domain.

# Author: Mark Jenkinson, University of Oxford and University of Adelaide
# Date: August 2021

# Neither the University of Oxford, the University of Adelaide, nor
# any of their employees imply any warranty of usefulness of this software
# for any purpose, and do not assume any liability for damages,
# incidental or otherwise, caused by any use of this document.

"""
Command-line tool for measuring the results of a normative model.
"""

import sys
import argparse

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import scipy.stats
import pandas as pd
import seaborn as sns
import pyreadr

import normodlib as nm

np.set_printoptions(linewidth=np.inf)

# define colours to use in background (use colorblind friendly ones)
# from website:
# http://mkweb.bcgsc.ca/colorblind/img/colorblindness.palettes.v11.pdf
# colornames = ["jazzberry jam", "jeepers creepers", "vivid opal", "aquamarine",
#              "french violet", "dodger blue", "capri", "plum", "carmine",
#              "alizarin crimson", "outrageous orange", "bright spark"]
fillcolors = ['#9f0162', '#009f81', '#00ebc1', '#00fccf', '#b400cd', '#008df9',
              '#00c2f9', '#ffb2fd', '#a40122', '#e20134', '#ff63ea', '#ffc33b']
# reorder the colours
fillcolors = [fillcolors[n] for n in [5, 2, 9, 9, 2, 5]] * 2

default_percvals = [0.01, 0.05, 0.10, 0.50, 0.90, 0.95, 0.99]
default_reportpercvals = [0.01, 0.05, 0.10]

SMALL_SIZE = 12
MEDIUM_SIZE = 16
BIGGER_SIZE = 22

plt.rc('font', size=MEDIUM_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)  # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


# ========================================================================== #

# SUPPORT FUNCTIONS

# ========================================================================== #


def extract_data(resdf, report_ptls):
    """
    Extract required data from dataframe (read in from csv) and do minimal processing.

    Args:
        resdf (pandas dataframe): Contains everything read in from csv or equivalent.
        report_ptls (list): Each value in the list is a percentile to report [0,1].

    Returns:
        list of arrays: estimated percentile curves from fitting (one per requested percentile)
            is a list (Nsims) ; of lists (Nperc) ; of arrays (Nsubj x 1).
        np.array: simulated data (e.g. hipp vols) - Nsubj x Nsims.
        list of floats: the set of simulation indices (e.g. simulation number 10) that had valid data.
    """

    colnames = resdf.columns
    maxrep = len(resdf.filter(regex="simdata").columns)

    # Variable naming convention: y = original datapoints; w = estimated curve values
    # Both of these are in the units of hippocampal volume

    # Each sim dataset is Nsubj (# subjects) and number of sims = Nsims = #reps = maxrep
    #   ytot is simulated data (hipp vols): Nsubj x Nsims
    nsubj = resdf.shape[0]
    ytot = np.zeros((nsubj, maxrep))
    # pcurves is list of estimated percentile curves from fitting (one per nominal percentile)
    # it is a list (#reps) ; of lists (#percentiles) ; of (#subjects by 1) arrays
    pcurves = []
    valididx = []
    mrep = 0
    for nrep in range(1, maxrep + 1):
        estptls = []
        dname = f'simdata_{nrep + 1}'
        if dname in colnames:
            for p in report_ptls:
                cname = f'centile{int(p * 100)}_sim{nrep + 1}'
                if cname in colnames:
                    rcol = resdf[cname]  # extract column data
                    rcol = rcol.apply(
                        pd.to_numeric, errors='coerce').to_numpy()
                    estptls += [rcol]
            # only include full, successful sims (no NaNs) in pcurves
            if len(estptls) == len(report_ptls):
                pcurves += [estptls]
                ytot[:, mrep] = resdf[dname]
                valididx += [nrep + 1]
                mrep += 1

    ytot = ytot[:, :len(pcurves)]  # restrict to valid subset
    print(f'SIZE of ytot is {ytot.shape}')

    return pcurves, ytot, valididx


def set_ground_truth(args):
    """
    Setup necessary variables that specify the ground truth model.

    Args:
        args (argparse object): object containing all the command-line user arguments.

    Returns:
        truetheta (list): parameter values (floats) used by the ground truth model.
        simmodeltype (str): name of the type of ground truth model (e.g. linear, poly, nonlin).
    """
    simmodeltype = args.simmodeltype
    truetheta = None
    if not args.simparams:
        if simmodeltype == 'nonlin':
            truetheta = nm.truetheta_nonlin
        elif simmodeltype == 'poly':
            truetheta = nm.truetheta_poly  # poly
        elif simmodeltype == 'linear':
            truetheta = nm.truetheta_linear  # linear, linear var
        elif simmodeltype == 'linearconstvar':
            truetheta = nm.truetheta_linearconstvar  # linear, const var
    else:
        truetheta = args.simparams

    if simmodeltype == 'linearconstvar':
        simmodeltype = 'linear'

    return truetheta, simmodeltype


# Modelling and evaluation functions


def plot_model(
        theta,
        x,
        alpha=1,
        fill=True,
        modeltype=None,
        percvals=None):
    """
    Creates a matplotlib plot of the ground truth (percentile lines and datapoints as circles).

    Args:
        theta (list): parameter values for ground truth.
        x (np.array): ages (or equivalent) of datapoints (one per point).
        alpha (float, optional): Transparency value (alpha) to use for datapoints. Defaults to 1.
        fill (bool, optional): Fill, with colours, between the percentile curves. Defaults to True.
        modeltype (string, optional): Name of ground truth model type. Defaults to None.
        percvals (list, optional): Set of percentile values to display as lines; each value in [0,1]  Defaults to None.

    Returns:
        list [mindata, maxdata]: as minimum and maximum value of ground truth in plotted range.
    """
    percvals = default_percvals if percvals is None else percvals
    x_sorted = np.sort(x, axis=0)
    g_pv0 = norm.ppf(percvals[0])
    dist_params = nm.model(theta, x_sorted, modeltype=modeltype)
    # print(f'Shape of dist_params = {dist_params[0].shape}')
    mindata = np.min(dist_params[0][:])
    maxdata = np.max(dist_params[0][:])
    for nptl, ptl in enumerate(percvals):
        g_pv = norm.ppf(ptl)
        mdata = nm.mapdata(g_pv, dist_params)
        mindata = np.min((mindata, np.min(mdata[:])))
        maxdata = np.max((maxdata, np.max(mdata[:])))
        plt.plot(x_sorted, mdata, 'k-', alpha=alpha)
        if fill:
            plt.fill_between(x_sorted, nm.mapdata(g_pv0, dist_params),
                             nm.mapdata(g_pv, dist_params),
                             color=fillcolors[nptl - 1], alpha=alpha)
            # color was np.array([1-ptl,np.abs(0.5-ptl),ptl]),
        g_pv0 = g_pv
    print(f'mindata = {mindata}, maxdata = {maxdata}')
    return [mindata, maxdata]


def calc_error_e1(
        perc_curves,
        x,
        truetheta,
        simmodeltype=None,
        percvals=None):
    """
    Calculate signed error between true and estimated percentile values
       in units of volume (response variable) - i.e. E1 in the paper.

    Args:
        perc_curves (list of arrays): estimated percentiles curves - a list of Nperc curves
            (i.e. one per percentile value) with each curve being a 1D array - Nsubj x 1.
        x (np.array): array of values (e.g. ages) of datapoints (one per subject) - Nsubj x 1.
        truetheta (list): parameter values (floats) used by the ground truth model - Nparams.
        simmodeltype (str): name of the type of ground truth model (e.g. linear, poly, nonlin).
        percvals (list, optional): Set of percentile values to display as lines
            each value is in [0,1].  Defaults to None.

    Returns:
        np.array: error values (using E1 formula in paper) with one value (float) per datapoint - Nsubj x Nperc.
    """
    percvals = default_percvals if percvals is None else percvals
    # evaluate model at points x - returns a set of mean and stddev values
    dist_params_true = nm.model(truetheta, x, modeltype=simmodeltype)
    err = np.zeros((len(x), len(percvals)))
    for idx, ptl in enumerate(percvals):
        d_sig = norm.ppf(ptl)
        err[:, idx] = (perc_curves[idx]) - (dist_params_true[0] + d_sig * dist_params_true[1])
    return err


def make_error_histogram(errtot, nagebins, binages, binedges, ages):
    """
    Make a histogram of the errors (bin the values separately for each sim).

    Calculate mean of percentile curve estimates for all samples within a bin for 1 simulation (giving 1 value per bin)
        - if age bins are 1 year wide and curves are relatively smooth (or all ages are only known by an integer age)
        then mean within a bin does little/nothing.

    Args:
        errtot (np.array): total set of error values (all datapoints, perc curves and sims) - Nsubj x Nperc x Nsims.
        nagebins (float): number of age bins to use (=Nbins).
        binages (np.array): values at centres of age bins - Nbins x 1.
        binedges (np.array): values at edge of age bins - (Nbins+1) x 1.
        ages (np.array): set of ages (or alternative x-axis values) - Nsubj x 1.

    Returns:
        np.array: average of error values within each histogram bin (per percentile and sim) - Nbins x Nperc x Nsims.
        np.array: Nbins x 2, where first column stores centre values of histogram bins
            and second columns stores number of datapoints per bin.
    """
    print(ages.shape)  # this is the set of ages for all sims
    binerrtot = np.zeros((nagebins, errtot.shape[1], errtot.shape[2]))
    for percval in range(errtot.shape[1]):  # percentile curve
        for simnum in range(errtot.shape[2]):  # sim dataset
            # bin wrt age (binedges from above are age bin edges: nagebins+1 x 1)
            # result of binning is res: nagebins x 1
            res = scipy.stats.binned_statistic(
                ages, errtot[:, percval, simnum], statistic='mean', bins=binedges)
            binerrtot[:, percval, simnum] = res[0]

    print(binerrtot.shape)

    # calculate stats from histogram (all summaries taken across
    # repetitions/sims)
    xbinnum = np.zeros((binages.shape[0], 2))
    xbinnum[:, 0] = binages
    xbinnum[:, 1], _ = np.histogram(ages, bins=binedges, density=False)

    return binerrtot, xbinnum


def calc_perc_below_thresh(pcurves, report_idx, gt_params, x, binedges):
    """
    Calculate percentage of samples that would fall beneath the estimate normative (percentile) curves.

    Args:
        pcurves (list of list of np.arrays): stored percentile curves (one float value per subject)
            - Nsims of Nperc of (Nsubj x 1)
        report_idx (int): index value of where desired percentile is within pcurves (second index)
        gt_params (list of np.arrays): values for ground truth mu and sigma - each array is Nsubj x 1.
        x (np.array): values (e.g. ages) of datapoints - Nsubj x 1.
        binedges (np.array): values respresenting edges of bins for histogram used to report results
            - Nbins+1 x 1.

    Returns:
        np.array: percentile values of points on estimated curves (where each value is the average within a bin)
            - Nsims x Nbins   (NB: transpose near end of function, so Nbins x Nsims for most of function)
    """
    allpres = None
    for nidx, perc_curves in enumerate(pcurves):
        # extract est vols on perc curves: list of Nperc of Nsubjx1
        estptls = perc_curves
        estvols = estptls[report_idx]  # a vector: Nsubj x 1
        spres = nm.conv2perc(estvols, mu=gt_params[0], sigma=gt_params[1])
        # calculate mean of perc curve values over all samples in a bin for 1 sim (1 value per bin)
        #   if age bins are 1 year wide and curves are relatively smooth (or all ages
        # are only known by an integer age) then first pass of mean in a bin
        # does little/nothing
        bpres = scipy.stats.binned_statistic(
            x, spres, statistic='mean', bins=binedges)
        pres = bpres[0].reshape(-1, 1)

        # concatenate results over all estimated percentiles (numreps)
        if nidx == 0:
            allpres = np.copy(pres)
        allpres = np.concatenate((allpres, pres), axis=1)
        # size of allpres should be nbins by numreps (at this point)

    allpres = allpres.T
    return allpres


# ========================================================================== #

# FUNCTIONS FOR STORING AND PLOTTING RESULTS

# ========================================================================== #

def dfcolumns():
    """
    Function just used to return names of columns.

    Returns:
        list: names of columns.
    """
    return [
        'Simulation Model',
        'Estimation Method',
        'Error Type',
        'Summary Type',
        'Nominal Percentile',
        'Age Bin',
        'Number in Age Bin',
        'Simulation Run',
        'Value']


# Store results for dataframe creation


def make_dictlist_e1(report_ptls, binages, xbinnum, medianerr, iqrerr, ci95, args):
    """
    Form a dictionary that contains all the summary results related to E1 errors (for all percentile curves).

    Args:
        report_ptls (list): list of requested percentiles to report - each in [0,1].
        binages (np.array): age values at centre of histogram bins - Nbins x 1.
        xbinnum (np.array): centre values of histogram bins (1st col) and num of samples per bin (2nd col) - Nbins x 2.
        medianerr (np.array): median of E1 error values in each bin (for each curve separately) - Nbins x Nperc.
        iqrerr (TYPE): IQR of E1 error values in each bin (for each curve separately) - Nbins x Nperc.
        ci95 (TYPE): 95% range of E1 error values in each bin (for each curve separately) - Nbins x Nperc.
        args (argparse object): object containing all the command-line user arguments.

    Returns:
        dict: dictionary of results with many different types of keys (including summaries) for multiple percentiles.
    """
    dflist = []
    newdict = {}
    for pidx, ptl in enumerate(report_ptls):
        for aidx in range(binages.shape[0]):
            newdict = {'Simulation Model': args.simmodeltype,
                       'Estimation Method': args.estname,
                       'Error Type': 'E1',
                       'Summary Type': 'Median',
                       'Nominal Percentile': ptl * 100,
                       'Age Bin': binages[aidx],
                       'Number in Age Bin': xbinnum[aidx, 1],
                       'Simulation Run': 'All',
                       'Value': medianerr[aidx, pidx]}
            dflist += [newdict.copy()]
            newdict['Summary Type'] = 'IQR'
            newdict['Value'] = iqrerr[aidx, pidx]
            dflist += [newdict.copy()]
            newdict['Summary Type'] = 'ci95'
            newdict['Value'] = ci95[aidx, pidx]
            dflist += [newdict.copy()]
            newdict['Summary Type'] = 'Total Error'
            newdict['Value'] = np.abs(
                medianerr[aidx, pidx]) + iqrerr[aidx, pidx]
            dflist += [newdict.copy()]
        newdict['Age Bin'] = 'All'
        newdict['Number in Age Bin'] = np.sum(xbinnum[:, 1])
        newdict['Summary Type'] = 'Median ci95'
        newdict['Value'] = np.median(ci95[:, pidx])
        dflist += [newdict.copy()]
        newdict['Summary Type'] = 'Mean ci95'
        newdict['Value'] = np.mean(ci95[:, pidx])
        dflist += [newdict.copy()]
    return dflist


# Store results for dataframe creation
def make_dictlist_e2(
        report_ptl,
        binages,
        xbinnum,
        meanperc,
        stdperc,
        medianperc,
        iqrperc,
        maeperc,
        ci95,
        args):
    """
    Form a dictionary that contains all the summary results related to E2 errors (for one percentile curve).

    Args:
        report_ptl (float): requested percentile to report - in [0,1].
        binages (np.array): age values at centre of histogram bins - Nbins x 1.
        xbinnum (np.array): centre values of histogram bins (1st col) and num of samples per bin (2nd col) - Nbins x 2.
        meanperc (np.array): mean of E2 error values in each bin (for each curve separately) - Nbins x 1.
        stdperc (np.array): std dev of E2 error values in each bin (for each curve separately) - Nbins x 1.
        medianperc (np.array): median of E2 error values in each bin (for each curve separately) - Nbins x 1.
        iqrperc (np.array): IQR of E2 error values in each bin (for each curve separately) - Nbins x 1.
        maeperc (np.array): mean abs value of E2 error values in each bin (for each curve separately) - Nbins x 1.
        ci95 (np.array): 95% range of E2 error values in each bin (for each curve separately) - Nbins x 1.
        args (argparse object): object containing all the command-line user arguments.

    Returns:
        dict: dictionary of results with many different types of keys (including summaries) for a single percentile.
    """
    dflist = []
    newdict = {}
    for aidx in range(binages.shape[0]):
        newdict = {'Simulation Model': args.simmodeltype,
                   'Estimation Method': args.estname,
                   'Error Type': 'E2',
                   'Summary Type': 'Mean',
                   'Nominal Percentile': report_ptl * 100,
                   'Age Bin': binages[aidx],
                   'Number in Age Bin': xbinnum[aidx, 1],
                   'Simulation Run': 'All',
                   'Value': meanperc[aidx]}
        dflist += [newdict.copy()]
        newdict['Summary Type'] = 'IQR'
        newdict['Value'] = iqrperc[aidx]
        dflist += [newdict.copy()]
        newdict['Summary Type'] = 'Median'
        newdict['Value'] = medianperc[aidx]
        dflist += [newdict.copy()]
        newdict['Summary Type'] = 'Std'
        newdict['Value'] = stdperc[aidx]
        dflist += [newdict.copy()]
        newdict['Summary Type'] = 'ci95'
        newdict['Value'] = ci95[aidx]
        dflist += [newdict.copy()]
    newdict['Age Bin'] = 'All'
    newdict['Number in Age Bin'] = np.sum(xbinnum[:, 1])
    newdict['Summary Type'] = 'MAE'
    newdict['Value'] = maeperc
    dflist += [newdict.copy()]
    return dflist


# ========================================================================== #

def plot_histogram(x, binages, xbinnum):
    """
    Plot histogram of ages and save to file.

    Args:
        x (np.array): values (e.g. ages) of datapoints - Nsubj x 1.
        binages (np.array): age values at centre of histogram bins - Nbins x 1.
        xbinnum (np.array): centre values of histogram bins (1st col) and num of samples per bin (2nd col) - Nbins x 2.

    Returns:
        Nothing.
    """
    # Histogram of ages
    darkcolors = sns.color_palette("dark").as_hex()
    plt.figure()
    sns.histplot(data=x, bins=binages, color=darkcolors[0])
    plt.title('Histogram of age')
    plt.xlabel('Age')
    plt.savefig(f'Figure_AgeHist_N{len(x)}.pdf')
    plt.savefig(f'Figure_AgeHist_N{len(x)}.pdf')
    np.savetxt(f'AgeHist_N{len(x)}.txt', xbinnum)


def plot_gt_examples(x, truetheta, simmodeltype, y_exemplar, pcurves):
    """
    Plot ground truth curves, sample points and estimated percentile curves; saves to a file.

    Args:
        x (np.array): values (e.g. ages) of datapoints - Nsubj x 1.
        truetheta (list): parameter values (floats) used by the ground truth model - Nparams.
        simmodeltype (str): name of the type of ground truth model (e.g. linear, poly, nonlin).
        y_exemplar (np.array): one set of simulated values (e.g. volumes) to be plotted - Nsubj x 1
        pcurves (list of list of np.arrays): stored percentile curves (one float value per subject)
            - Nsims of Nperc of (Nsubj x 1)
    Returns:
        Nothing.
    """
    # Single set of estimated percentiles + ground truth + data
    plt.figure()
    [mind, maxd] = plot_model(truetheta, x, alpha=0.1, modeltype=simmodeltype)
    plt.ylim(mind - 0.1 * (maxd - mind), maxd + 0.1 * (maxd - mind))
    plt.xlabel('Age')
    plt.ylabel('Volume')
    plt.title('Ground truth')
    plt.savefig('Figure_GT.pdf')
    alphaval = 0.1 if len(x) < 10000 else 0.02
    plt.scatter(x.astype(int), y_exemplar, color='k', alpha=alphaval)
    plt.title('Ground truth and samples')
    plt.savefig(f'Figure_GT_Samples_N{len(x)}.pdf')
    for perc_curves in pcurves[-1]:
        plt.plot(x, perc_curves, 'b--', linewidth=2.5)
    plt.title('Single example of estimated percentiles')
    plt.savefig(f'Figure_1Pcurve_N{len(x)}.pdf')


def plot_est_pcurves(x, truetheta, simmodeltype, pcurves, linecolors):
    """
    Plots whole set of estimated percentile curves on top of each other to show spread; saves to a file (3Pcurves).

    Args:
        x (np.array): values (e.g. ages) of datapoints - Nsubj x 1.
        truetheta (list): parameter values (floats) used by the ground truth model - Nparams.
        simmodeltype (str): name of the type of ground truth model (e.g. linear, poly, nonlin).
        pcurves (list of list of np.arrays): stored percentile curves (one float value per subject)
            - Nsims of Nperc of (Nsubj x 1)
        linecolors (list): colour codes, one per percentile curve to be displayed (currently 3)

    Returns:
        Nothing.
    """
    # Full set of estimated percentiles (across all simulated datasets)
    plt.figure()
    [mind, maxd] = plot_model(truetheta, x, alpha=0.1, modeltype=simmodeltype)
    for perc_curves in pcurves:
        # was [0,3,6] as pctls [1,50,99]
        plt.plot(x, perc_curves[0], linecolors[0], alpha=0.02)
        plt.plot(x, perc_curves[1], linecolors[1], alpha=0.02)
        plt.plot(x, perc_curves[2], linecolors[2], alpha=0.02)
    plt.ylim(mind - 0.1 * (maxd - mind), maxd + 0.1 * (maxd - mind))
    plt.title('Set of estimated percentiles')
    plt.xlabel('Age')
    plt.ylabel('Volume')
    plt.savefig(f'Figure_3Pcurves_N{len(x)}.pdf')


def plot_mean_err(x, errtot, maxerr, legend_perc):
    """
    Plots the mean E1 error as a function of age.

    Args:
        x (np.array): values (e.g. ages) of datapoints - Nsubj x 1.
        errtot (np.array): total set of error values (all datapoints, perc curves and sims) - Nsubj x Nperc x Nsims.
        maxerr (float): maximum error to use for scaling the plots
        legend_perc (tuple of strings): tuple to be passed to the legend call in matplotlib

    Returns:
        Nothing.
    """
    # Plot the mean error of estimated percentiles (mean over sims)
    plt.figure()
    plt.plot(x, np.mean(errtot[:, :3, :], axis=2))
    plt.gca().legend(legend_perc)
    plt.xlabel('Age')
    plt.ylabel('Mean Error')
    plt.ylim(-maxerr, maxerr)
    plt.title(f'Mean error values: N = {len(x)}')
    plt.savefig(f'Figure_MeanErr_N{len(x)}.pdf')


def plot_median_err(x, medianerr, binages, agebinwidth, linecolors,
                    plotrangesim, legend_perc):
    """
    Bar plots of the median E1 error for different percentiles as a function of age.

    Args:
        x (np.array): values (e.g. ages) of datapoints - Nsubj x 1.
        medianerr (np.array): median error values per bin and per percentile curve - Nbins x Nperc
        binages (np.array): values at centres of age bins - Nbins x 1.
        agebinwidth (float): width of the age histogram bins.
        linecolors (list): colour codes, one per percentile curve to be displayed (currently 3).
        plotrangesim (list or np.array): values for setting y-axis plot range ([0] is min, [1] is max).
        legend_perc (tuple of strings): tuple to be passed to the legend call in matplotlib.

    Returns:
        Nothing.
    """
    # Bar plots of median error for 1, 5, 10-th percentiles (or whatever are in report_ptls[0:3])
    plt.figure()
    plt.bar(binages - 0.3 * agebinwidth,
            medianerr[:, 0], 0.3, color=linecolors[0])
    plt.bar(binages, medianerr[:, 1], 0.3, color=linecolors[1])
    plt.bar(binages + 0.3 * agebinwidth,
            medianerr[:, 2], 0.3, color=linecolors[2])
    plt.ylim(plotrangesim[0], plotrangesim[1])
    plt.gca().legend(legend_perc)
    plt.xlabel('Age')
    plt.ylabel('Median Error')
    plt.title(f'Median error values: N = {len(x)}')
    plt.savefig(f'Figure_MedianErr_N{len(x)}.pdf')


def plot_iqr_err(x, iqrerr, binages, agebinwidth, linecolors,
                 plotrangesim, legend_perc):
    """
    Bar plots of the IQR of E1 error for different percentiles as a function of age.

    Args:
        x (np.array): values (e.g. ages) of datapoints - Nsubj x 1.
        iqrerr (np.array): IQR of error values per bin and per percentile curve - Nbins x Nperc
        binages (np.array): values at centres of age bins - Nbins x 1.
        agebinwidth (float): width of the age histogram bins.
        linecolors (list): colour codes, one per percentile curve to be displayed (currently 3)
        plotrangesim (list or np.array): values for setting y-axis plot range ([0] is min, [1] is max)
        legend_perc (tuple of strings): tuple to be passed to the legend call in matplotlib

    Returns:
        Nothing.
    """
    plt.figure()
    plt.bar(binages - 0.3 * agebinwidth,
            iqrerr[:, 0], 0.3, color=linecolors[0])
    plt.bar(binages, iqrerr[:, 1], 0.3, color=linecolors[1])
    plt.bar(binages + 0.3 * agebinwidth,
            iqrerr[:, 2], 0.3, color=linecolors[2])
    plt.ylim(plotrangesim[0], plotrangesim[1])
    plt.gca().legend(legend_perc)
    plt.xlabel('Age')
    plt.ylabel('IQR of Errors')
    plt.title(f'IQR error values: N = {len(x)}')
    plt.savefig(f'Figure_IQRErr_N{len(x)}.pdf')


def create_e1_plots(
        x,
        pcurves,
        report_ptls,
        truetheta,
        simmodeltype,
        y_exemplar,
        errtot,
        medianerr,
        iqrerr,
        maxerr,
        binages,
        xbinnum,
        agebinwidth,
        simname):
    """
    Creates numerous plots all based on E1 error and saves them as files.

    Args:
        x (np.array): values (e.g. ages) of datapoints - Nsubj x 1.
        pcurves (list of list of np.arrays): stored percentile curves (one float value per subject)
            - Nsims of Nperc of (Nsubj x 1)
        report_ptls (list): Each value in the list is a percentile to report [0,1].
        truetheta (list): parameter values (floats) used by the ground truth model - Nparams.
        simmodeltype (str): name of the type of ground truth model (e.g. linear, poly, nonlin).
        y_exemplar (np.array): one set of simulated values (e.g. volumes) to be plotted - Nsubj x 1
        errtot (np.array): total set of error values (all datapoints, perc curves and sims) - Nsubj x Nperc x Nsims.
        medianerr (np.array): median error values per bin and per percentile curve - Nbins x Nperc
        iqrerr (np.array): IQR of error values per bin and per percentile curve - Nbins x Nperc
        maxerr (float): maximum error to use for scaling the plots
        binages (np.array): values at centres of age bins - Nbins x 1.
        xbinnum (np.array): centre values of histogram bins (1st col) and num of samples per bin (2nd col) - Nbins x 2.
        agebinwidth (float): width of the age histogram bins.
        simname (str): key used for dictionary plotranges (used to fix y-scaling if needed).

    Returns:
        Nothing.
    """
    linecolors = fillcolors  # sns.color_palette("colorblind").as_hex()

    legend_perc = (f'{report_ptls[0] * 100}',
                   f'{report_ptls[1] * 100}', f'{report_ptls[2] * 100}')

    plot_histogram(x, binages, xbinnum)
    plot_gt_examples(x, truetheta, simmodeltype, y_exemplar, pcurves)
    plot_est_pcurves(x, truetheta, simmodeltype, pcurves, linecolors)
    plot_mean_err(x, errtot, maxerr, legend_perc)

    # Fixed plot ranges just for figures in paper
    plotranges = {'NonLinMean_NonConstVar': [-50, 100],
                  '_LinMean_ConstVar': [-40, 40],
                  'NonLinMean_ConstVar': [-70, 70],
                  'Unknown': [-maxerr, maxerr]}
    plot_median_err(x, medianerr, binages, agebinwidth, linecolors,
                    plotranges[simname], legend_perc)

    # Redfine plot ranges for IQR plots
    plotranges = {'NonLinMean_NonConstVar': [0, 200],
                  '_LinMean_ConstVar': [0, 200],
                  'NonLinMean_ConstVar': [0, 200],
                  'Unknown': [0, 2 * maxerr]}
    plot_iqr_err(x, iqrerr, binages, agebinwidth, linecolors,
                 plotranges[simname], legend_perc)


# ========================================================================== #

def create_perc_below_thr_plots(
        x,
        allpres,
        report_ptl,
        report_idx,
        binages,
        meanperc,
        medianperc,
        iqrperc,
        simname):
    """
    Creates numerous plots all based on percentage-below-threshold and E2 error, and saves them as files.

    Note: E2 error = percentage below estimated percentile (used as a threshold) - desired percentile.

    Args:
        x (np.array): values (e.g. ages) of datapoints - Nsubj x 1.
        allpres (np.array): percentile values of points on estimated curves for all sims (for BoxPlots) - Nsims x Nbins.
        report_ptl (float): requested percentile to report - in [0,1].
        report_idx (int): index value of where report_ptl value is in report_ptls array.
        binages (np.array): values at centres of age bins - Nbins x 1.
        meanperc (np.array): mean of percentage below threshold values, one per bin - Nbins x 1.
        medianperc (np.array): median of percentage below threshold values, one per bin - Nbins x 1.
        iqrperc (np.array): IQR of percentage below threshold values, one per bin - Nbins x 1.
        simname (str): key used for dictionary plotranges (used to fix y-scaling if needed).

    Returns:
        Nothing.
    """

    plt.figure()
    sns.set(style="whitegrid")
    ax = sns.barplot(x=binages, y=meanperc * 100)
    sns.lineplot(x=[-0.5, len(binages) - 0.5], y=report_ptl * 100, color="red")
    ax.lines[-1].set_linestyle("--")
    ymax = np.ceil(np.nanmax(iqrperc * 100) / 5.0) * 5  # round to nearest 5
    ymin = -5 if ymax >= 7.5 else -ymax / 5.0
    ax.set(ylim=(ymin, ymax))
    plt.xlabel('Age', fontsize=MEDIUM_SIZE)
    plt.ylabel('Mean of percentage below threshold', fontsize=MEDIUM_SIZE)
    plt.title(f'Percentile = {int(report_ptl * 100)}, Data size = {len(x)}',
              fontsize=BIGGER_SIZE)
    plt.xticks(ticks=np.arange(45, 81, 5) - binages[0], labels=np.arange(45, 81, 5))  # hard-coded x-range
    plt.savefig(f'Figure_MeanEstPct{int(report_ptl * 100)}_N{len(x)}.pdf')

    plt.figure()
    sns.set(style="whitegrid")
    ax = sns.barplot(x=binages, y=medianperc * 100)
    sns.lineplot(x=[-0.5, len(binages) - 0.5], y=report_ptl * 100, color="red")
    ax.lines[-1].set_linestyle("--")
    ax.set(ylim=(ymin, ymax))
    plt.xlabel('Age', fontsize=MEDIUM_SIZE)
    plt.ylabel('Median of percentage below threshold', fontsize=MEDIUM_SIZE)
    plt.title(f'Percentile = {int(report_ptl * 100)}, Data size = {len(x)}',
              fontsize=BIGGER_SIZE)
    plt.xticks(ticks=np.arange(45, 81, 5) - binages[0], labels=np.arange(45, 81, 5))  # hard-coded x-range
    plt.savefig(f'Figure_MedianEstPct{int(report_ptl * 100)}_N{len(x)}.pdf')

    plt.figure()
    sns.set(style="whitegrid")
    ax = sns.barplot(x=binages, y=iqrperc * 100)
    ax.set(ylim=(ymin, ymax))
    plt.xlabel('Age', fontsize=MEDIUM_SIZE)
    plt.ylabel('IQR of percentages below threshold', fontsize=MEDIUM_SIZE)
    plt.title(f'Percentile = {int(report_ptl * 100)}, Data size = {len(x)}',
              fontsize=BIGGER_SIZE)
    plt.xticks(ticks=np.arange(45, 81, 5) - binages[0], labels=np.arange(45, 81, 5))  # hard-coded x-range
    plt.savefig(f'Figure_IQREstPct{int(report_ptl * 100)}_N{len(x)}.pdf')

    plt.figure()
    plotranges = {'NonLinMean_NonConstVar': [0, 5, 25, 50],
                  '_LinMean_ConstVar': [0, 5, 25, 50],
                  'NonLinMean_ConstVar': [0, 5, 25, 50],
                  'Unknown': [0, ymax, ymax, ymax]}
    sns.set(style="whitegrid")
    plotdata = pd.DataFrame(data=allpres * 100, columns=binages,
                            index=list(range(allpres.shape[0])))
    ax = sns.boxplot(data=plotdata)
    sns.lineplot(x=[-0.5, len(binages) - 0.5], y=report_ptl * 100, color="red")
    ax.lines[-1].set_linestyle("--")
    ax.set(ylim=(plotranges[simname][0], plotranges[simname][min(1 + report_idx, 3)]))
    plt.xlabel('Age', fontsize=MEDIUM_SIZE)
    plt.ylabel('Actual percentage below threshold', fontsize=MEDIUM_SIZE)
    plt.xticks(ticks=np.arange(45, 81, 5) - binages[0], labels=np.arange(45, 81, 5))  # hard-coded x-range
    plt.title(f'Percentile = {int(report_ptl * 100)}, Data size = {len(x)}',
              fontsize=BIGGER_SIZE)
    plt.savefig(f'Figure_BoxplotEstPct{int(report_ptl * 100)}_N{len(x)}.pdf')


# ========================================================================== #

def do_work(args):
    """
    Main flow of functionality that happens after arg parsing: setup ground truth --> load data --> setup histograms
        --> extract data --> calc E1 --> plot results to file and put in dict
        --> calc E2 --> plot results to file and add to dict --> save dict results to file.

    Args:
      args (argparse object): object containing all the command-line user arguments.

    Returns:
      Nothing.
    """
    # Set up ground truth
    truetheta, simmodeltype = set_ground_truth(args)

    simname = 'Unknown'
    for simn in [
            'NonLinMean_NonConstVar',
            '_LinMean_ConstVar',
            'NonLinMean_ConstVar']:
        if simn in args.inputfile:
            simname = simn

    # Percentiles of interest (to be processed below)
    if args.reportpercs:
        report_ptls = np.array(args.reportpercs) / 100.0
    else:
        report_ptls = np.array(default_reportpercvals)

    # ========================================================================== #

    # Load data (results from running some normative model)

    if 'csv' in args.inputfile:
        resdf = pd.read_csv(args.inputfile)
    elif 'rds' in args.inputfile:
        resdf0 = pyreadr.read_r(args.inputfile)
        resdf = resdf0[None]
    else:
        print(
            f'Cannot find either csv or rds files in specified location: {args.inputfile}')
        sys.exit(2)

    print(resdf.columns)

    # ========================================================================== #

    # Setup age range and histogram
    binages, binedges, agebinwidth, nagebins, x = nm.hist_age_data(resdf['age'].to_numpy())

    # Setup true distribution info
    theta = truetheta  # from input args/model-spec
    dist_params0 = nm.model(theta, x, modeltype=simmodeltype)

    # ========================================================================== #

    # Extract required data and get it into the desired format

    pcurves, ytot, _ = extract_data(resdf, report_ptls)  # main extraction
    nsubj = ytot.shape[0]
    numreps = len(pcurves)
    print(f'Number of valid simulation runs = {numreps}')
    y_exemplar = resdf['simdata_1']

    # ========================================================================== #
    # ========================================================================== #

    # Calculate errors (E1)
    #    errtot: N samples * N percentile curves * N simulated datasets
    errtot = np.zeros((nsubj, len(report_ptls), numreps))
    for idx, perc_curves in enumerate(pcurves):
        errtot[:, :, idx] = calc_error_e1(perc_curves, x, truetheta, simmodeltype=simmodeltype, percvals=report_ptls)

    # Make a histogram of the errors (bin the values separately for each sim)
    #    binerrtot is: Nbins x Nperc_curves x Nsims
    binerrtot, xbinnum = make_error_histogram(errtot, nagebins, binages, binedges, x)

    # Calculate summary stats and plot/print results for E1 errors

    # used for plotting
    maxerr = 1.1 * np.ma.masked_invalid(np.nanmean(np.abs(binerrtot), axis=2)[:]).max()
    medianerr = np.nanmedian(binerrtot, axis=2)  # median over sims
    # scipy.stats.iqr(binerrtot, axis=2)  # iqr over sims
    iqrerr = np.nanpercentile(binerrtot, 75, axis=2) - np.nanpercentile(binerrtot, 25, axis=2)
    ci95 = np.nanpercentile(binerrtot, 97.5, axis=2) - np.nanpercentile(binerrtot, 2.5, axis=2)

    # Store results in dataframe
    dflist = []
    dflist += make_dictlist_e1(report_ptls, binages,
                               xbinnum, medianerr, iqrerr, ci95, args)

    # ========================================================================== #

    # Print and plot results for E1 errors and summaries

    for n, ptl in enumerate(report_ptls):
        # Text outputs
        print(
            f'   **************** Percentile = {ptl * 100} *******************')
        print(f'Bias (median) for P={ptl * 100} is {medianerr[:, n]}')
        print(f'Variance (IQR) for P={ptl * 100} is {iqrerr[:, n]}')
        print(
            f'Total error for P={ptl * 100} is {np.abs(medianerr[:, n]) + iqrerr[:, n]}')
        print(f'ci95 for P={ptl * 100} is {ci95[:, n]}')
        print(f'Median of ci95 for P={ptl * 100} is {np.nanmedian(ci95[:, n])}')
        print(f'Mean of ci95 for P={ptl * 100} is {np.nanmean(ci95[:, n])}')

    if not args.noplots:
        create_e1_plots(x, pcurves, report_ptls, truetheta, simmodeltype,
                        y_exemplar, errtot, medianerr, iqrerr, maxerr,
                        binages, xbinnum, agebinwidth, simname)

    # ========================================================================== #
    # ========================================================================== #

    # Calculate percentile errors (E2)
    #   errors are in terms of percentile units (differences of true to nominal percentile)
    #  allres : true percentile values of points on estimated curves
    #    - Nsims x nbins array

    # loop over report_ptls (e.g. 1,5,10 percentiles)
    for report_idx, report_ptl in enumerate(report_ptls):
        print(f'PERCENTILE = {report_ptl}')
        allpres = calc_perc_below_thresh(pcurves, report_idx, dist_params0, x, binedges)

        # ========================================================================== #

        # Calculate summary results

        meanperc = np.nanmean(allpres, axis=0)
        stdperc = np.nanstd(allpres, axis=0)
        medianperc = np.nanmedian(allpres, axis=0)
        iqrperc = np.nanpercentile(
            allpres, 75, axis=0) - np.nanpercentile(allpres, 25, axis=0)
        maeperc = np.nanmean(
            np.mean(np.abs(allpres - report_ptl), axis=0))  # scalar summary
        ci95 = np.nanpercentile(allpres, 97.5, axis=0) - np.nanpercentile(allpres, 2.5, axis=0)

        dflist += make_dictlist_e2(report_ptl,
                                   binages,
                                   xbinnum,
                                   meanperc,
                                   stdperc,
                                   medianperc,
                                   iqrperc,
                                   maeperc,
                                   ci95,
                                   args)

        ###########################

        # Report summary results

        print(f'SHAPE of allpres is (P = {report_ptl * 100}) {allpres.shape}')
        print(f'Mean percentile rates are (P = {report_ptl * 100}): {meanperc}')
        print(f'IQR of percentile rates are (P = {report_ptl * 100}): {iqrperc}')
        print(
            f'STDDEV of percentile rates are (P = {report_ptl * 100}): {stdperc}')
        print(f'MAE of percentile rate (P = {report_ptl * 100}) = {maeperc}')

        # Plot results
        if not args.noplots:
            create_perc_below_thr_plots(x, allpres, report_ptl, report_idx, binages,
                                        meanperc, medianperc, iqrperc, simname)

    # ========================================================================== #
    # ========================================================================== #

    # create overall dataframe and save results

    df = pd.DataFrame(dflist, columns=dfcolumns())
    df.to_csv(args.outname + ".csv", index=False)


# ========================================================================== #
# ========================================================================== #
# ========================================================================== #


# ====== MAIN ====== #

if __name__ == "__main__":

    # ========================================================================== #

    # User parameters

    parser = argparse.ArgumentParser(
        description="Measure performance of normative models using simulated data")

    parser.add_argument("-i", "--inputfile", type=str,
                        help="name of input data file (csv format, the output of a " +
                             "normative model run on simulated data", required=True)
    parser.add_argument("-e", "--estname", type=str,
                        help="name of estimation method (used for naming results output)", required=True)
    parser.add_argument("-s", "--simmodeltype", type=str, default="unknown",
                        help="type of model for simulation (linear, linearconstvar, nonlin, poly)")
    parser.add_argument("-o", "--outname", type=str, default="normodres",
                        help="basename for output files")
    parser.add_argument("--simparams", type=float,  nargs='+',
                        help="initial parameters used for simulation model")
    parser.add_argument("--reportpercs", type=float, nargs='+',
                        help="percentile values to report (each percentile between 0 and 100)", default=None)
    parser.add_argument("--noplots", action="store_true")

    args_main = parser.parse_args()

    # Sanity checking
    if args_main.simmodeltype not in ['nonlin', 'linear', 'linearconstvar', 'poly']:
        print(
            f"Invalid choice for simulation type ({args_main.simmodeltype}): "
            + "must be linear, linearconstvar or nonlin")
        sys.exit(1)

    # Continue with main processing
    do_work(args_main)
