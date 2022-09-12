#!/usr/bin/env python


#  This code is released to the public domain.

#  Author: Mark Jenkinson, University of Oxford and University of Adelaide
#  Date: August 2021

#  Neither the University of Oxford, the University of Adelaide, nor
#  any of their employees imply any warranty of usefulness of this software
#  for any purpose, and do not assume any liability for damages,
#  incidental or otherwise, caused by any use of this document.

"""
This command-line tool (and module) is used to create simulated data to test
normative modelling approaches.
"""

import sys
import time
import argparse

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pyreadr

import normodlib as nm


# ========================================================================== #

def makeage(mode, arg, maxnum=None):
    """
    Make a set of simulated age values.

    Args:
        mode (str): how to interpret arg ('file' or other).
        arg (str or list): filename if mode=='file' otherwise
            arg[0] is number of subjects; arg[1] is minimum age; arg[2] is maximum age.
        maxnum (int, optional): number of subjects to take in 'file' mode.
            Defaults to None, meaning that all values from the file are taken.

    Returns:
        x (np.array): set of ages.
        minage (float): minimum age value in set.
        maxage (float): maximum age value in set.

    """

    if mode == 'file':
        #   Read in age distribution from file
        xall = np.loadtxt(arg)
        xall = xall[:, 2]
        if maxnum:
            if maxnum > xall.shape[0]:
                xall = np.tile(xall, int(maxnum / xall.shape[0]) + 1)  # extend by replication
                xall = xall + (np.random.rand(xall.shape[0]) - 0.5)  # small jitter
            x = xall[0:maxnum]
        else:
            x = xall
        nsubj = x.shape[0]
        x = np.reshape(x, (nsubj,))
    else:
        #   Generate simulated age distribution: mixture of truncated Gaussian and uniform
        nsubj = arg[0]  # e.g. 5000
        minx = arg[1]  # e.g. 40 years
        maxx = arg[2]  # e.g. 80 years

        # simulate a set of subjects ith (uses some hardcoded rules for mu & sigma)
        x = np.random.normal(0.5 * (maxx + minx), 0.175 * (maxx - minx), size=(int(nsubj * 0.8), 1))
        # exclude subjects outside the allowed age range
        x = x[x > minx]
        x = x[x < maxx]
        # add a proportion that is distributed with a uniform dist (mixture model)
        x = np.hstack((x, np.random.uniform(low=minx, high=maxx, size=(nsubj - x.shape[0],))))

    print(x.shape)
    # sort so that the age values are in order
    x = np.sort(x, axis=0)
    minage = np.min(x)
    maxage = np.max(x)
    return x, minage, maxage


# ========================================================================== #

def simulate_data(x_age, dist_params, plotres=False, args=None):
    """
    Create a set of simulated data values (e.g. volumes) given the age (or any x) values.

    Args:
        x_age (np.array): array of x (e.g. age) values
        dist_params (list of np.array): [0] = mu , [1] = sigma ; parameters of the ground truth model, each Nsubj x 1.
        plotres (bool): flag to control whether to generate plots or not.
        args (argparse structure): set of command-line arguments.

    Returns:
        y_all (list of np.array): Nsims (list) of Nsubj x 1 arrays, each contains y (e.g. volume) values.
        wptls (list of np.array): Nperc (list) of Nsubj x 1 arrays, each array being a ground truth percentile curve.
    """
    # Create a set of simulated samples from the same ground truth
    y_all = []
    wptls = nm.calc_perc_curves(dist_params, args.percvals)  # x_age not needed as dist_params has one value per subj
    for _ in range(args.nsim):
        y = nm.makesampledata(dist_params)
        y_all += [y]

    # Plot results if explicitly requested
    if plotres:
        plt.figure()
        # noinspection PyUnboundLocalVariable
        plt.plot(x_age, y, 'o')
        plt.plot(x_age, dist_params[0], 'rx')
        print(f'Dist_params are {len(dist_params)} by {dist_params[0].shape} '
              + f'and {dist_params[1].shape} ; wptls are {len(wptls)} by {wptls[0].shape}')
        plt.plot(x_age, wptls[1], 'g.')  # 5th percentile
        plt.plot(x_age, wptls[4], 'b.')  # 50th percentile
        plt.plot(x_age, wptls[7], 'g.')  # 95th percentile
        plt.show(block=False)
        input("Hit Enter To Close")
        plt.close()
    return y_all, wptls


# ========================================================================== #

def do_work(args):
    """
    Main set of steps in the algorithm: create ages ; generate ground truth ; simulate data ; save results.

    Args:
        args (argparse structure): set of command-line arguments.

    Returns:
        Nothing.
    """
    # Create age distribution
    if main_args.agedist:
        x, minage, maxage = makeage('file', args.agedist, args.numsamples)
    else:
        x, minage, maxage = makeage('rand', [args.numsamples, args.minage, args.maxage])

    print(f'Age range is {minage} to {maxage}')

    # ========================================================================== #

    # Generate ground truth parameters (dist_params contain mean, stddev, etc, for each x value)
    np.random.seed(args.seed)  # set to be repeatable (may not be consistent across machines)
    dist_params = nm.model(truetheta, x, modeltype=args.simmodeltype)

    # ========================================================================== #

    # Simulate data
    y_all, gt_wptls = simulate_data(x, dist_params, plotres=args.plotres, args=args)
    print(f'Size of data: y_all = {len(y_all)} by {y_all[0].shape} and '
          + 'x = {len(x)} by {x[0].shape}')
    mergedata = np.concatenate((np.array(x).reshape(-1, 1), np.array(y_all).T), axis=1)

    # ========================================================================== #

    # Save data
    if not args.nosave:
        # save in either csv, R or numpy formats
        if args.save_format == 'csv':
            # noinspection PyTypeChecker
            np.savetxt(f'simdata_{mergedata.shape[0]}.csv', mergedata, delimiter=',')
            if args.save_gt:
                np.savetxt('percentiles_gt.csv', gt_wptls, delimiter=',')
        else:
            f32array = np.array(mergedata, dtype=np.float32)
            if args.save_format == 'rds':
                df = pd.DataFrame(f32array)
                pyreadr.write_rds(f'simdata_{mergedata.shape[0]}.rds', df, compress="gzip")
                if args.save_gt:
                    gt_wptls_df = pd.DataFrame(np.array(gt_wptls, dtype=np.float32))
                    pyreadr.write_rds('percentiles_gt.rds', gt_wptls_df, compress="gzip")
            elif args.save_format == 'npy':
                np.save(f'simdata_{mergedata.shape[0]}.npy', f32array)
                if args.save_gt:
                    np.save('percentiles_gt.npy', np.array(gt_wptls, dtype=np.float32))
            else:
                print(f'Save format ({args.save_format}) is not csv, rds or npy')
                sys.exit(2)

    if args.save_gt:
        np.savetxt('percentiles_gt.csv', gt_wptls, delimiter=',')


# ========================================================================== #
# ========================================================================== #
# ========================================================================== #


# ====== MAIN  ====== #

if __name__ == "__main__":

    # ========================================================================== #

    # User parameters

    parser = argparse.ArgumentParser(description="Simulation of ground truth "
                                                 + "data for normative modelling evaluation")

    parser.add_argument("-n", "--numsamples", type=int, default=5000,
                        help="number of samples", required=True)
    parser.add_argument("-s", "--simmodeltype", type=str, default="nonlin",
                        help="type of model for simulation (linear, linearconstvar, nonlin, poly)")
    parser.add_argument("--simparams", type=float, nargs='+',
                        help="initial parameters used for simulation model")
    parser.add_argument("--nsim", type=int, default=100,
                        help="number of simulated datasets")
    parser.add_argument("--agedist", type=str, default=None,
                        help="filename for age distribution")
    parser.add_argument("--minage", type=int, default=40,
                        help="minimum age for random age distribution")
    parser.add_argument("--maxage", type=int, default=80,
                        help="maximum age for random age distribution")
    parser.add_argument("--save_gt", action="store_true",
                        help="save ground truth percentiles")
    parser.add_argument("--nosave", action="store_true",
                        help="do not save simulated values")
    parser.add_argument("--save_format", type=str, default='csv',
                        help="type of save file format (csv, rds or npy)")
    parser.add_argument("--plotres", action="store_true",
                        help="plot results")
    parser.add_argument("--seed", type=int, default=42,
                        help="set random seed (use -1 to have it set from current time)")

    main_args = parser.parse_args()

    # Sanity checking
    if main_args.simmodeltype not in ['nonlin', 'linear', 'linearconstvar', 'poly']:
        print(f"Invalid choice for simulation type ({main_args.simmodeltype}): "
              + "must be linear, linearconstvar or nonlin")
        sys.exit(1)
    if main_args.numsamples < 10:
        print("Minimal number of samples is 10")
        sys.exit(1)
    if main_args.seed < 0:
        main_args.seed = time.time()
        main_args.seed = int((main_args.seed - int(main_args.seed)) * 1000)  # extract milliseconds

    # ========================================================================== #

    # Setup parameters
    if not main_args.simparams:
        if main_args.simmodeltype == 'nonlin':
            truetheta = nm.truetheta_nonlin
        elif main_args.simmodeltype == 'poly':
            truetheta = nm.truetheta_poly
        elif main_args.simmodeltype == 'linear':
            truetheta = nm.truetheta_linear
        elif main_args.simmodeltype == 'linearconstvar':
            truetheta = nm.truetheta_linearconstvar
    else:
        truetheta = main_args.simparams

    main_args.percvals = nm.percvals

    do_work(main_args)
