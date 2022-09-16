# NormativeModelling

This is a collection of command line utilities for testing and evaluation of normative modelling methods.

The code is written in python and R can also be accessed programmatically, but the recommended method is to use the command line interface.

## Authors

Code:
  - Mark Jenkinson and Jelena Bozek

Main citation:
  - Scripts included in this repository are described in detail in the paper "Normative models for neuroimaging markers: impact of model selection, sample size and evaluation criteria" by Jelena Bozek, Ludovica Griffanti, Stephan Lau and Mark Jenkinson.



## General Code

There are four main commands:
 - sim_data.py
    - This creates simulated datasets
 - normative_modelling_gamlss.R
    - This R script fits dataset using GAMLSS. The implementation we used for the gamlss function came from package GAMLSS (version 5.1-6). The script uses several GAMLSS models, implementing linear fitting or cubic spline smoothing across age, together with a Box Cox T (BCT) or a SinhArcsinh (SHASH) transformation, both of which create four parameter continuous distributions.
 - est_percentiles.py
    - This estimates normative models using sliding windows methods
 - measure_results.py
    - This measures a range of performance metrics using the normative model outputs

Detailed help and usage for each command can be found by running it with the argument `-h`


## Example Usage and Explanatory Notes

### Simulate Data

  `sim_data.py --numsamples 1000 --nsim 5000 --agedist simages.txt --simmodeltype nonlin --simparams 65.0 0.1 20.0 -70.0 6000.0 65.0 0.1 1.0 5.0 200.0`
  
  - This creates a set of 5000 simulations, where each simulation contains 1000 datapoints (i.e. simulated subjects)
  - The ages are read from the file simages.txt (the first number on each line is taken as a separate age)
  - The output is written into a file called simdata_1000.csv (the format can be changed using the --save_format option)
    - The structure of the output file is that each row contains one age value followed by one value for each of the Nsim simulations (5000 here)
    - There are as many rows as simulated subjects (1000 here)
    - The first column is the set of ages of all subjects (common to all simulations) and every other column is one simulated dataset
  - The ground truth distribution is specified by a set of parameters (simparams) and the class (nonlin), as the different classes interpret the parameters differently.  The main classes and the corresponding mean and standard deviation functions are:
    - *linear*: &mu;(x) = &theta;<sub>0</sub> + &theta;<sub>1</sub> * x ; &sigma;(x) = &theta;<sub>2</sub> + &theta;<sub>3</sub> * x 
      - where &theta;<sub>n</sub> represents the n'th simparam value
    - *poly*: &mu;(x) = &theta;<sub>0</sub> + &theta;<sub>1</sub> * x + &theta;<sub>2</sub> * x<sup>2</sup> + ... ; &sigma;(x) = &theta;<sub>m</sub> + &theta;<sub>m+1</sub> * x + &theta;<sub>m+2</sub> * x<sup>2</sup> + ...
      - where the polynomial has degree m, specified by 2m values in simparam
    - *nonlin*: &mu;(x) = sigmoid((x - &theta;<sub>0</sub>) * &theta;<sub>1</sub>) * ((x-&theta;<sub>0</sub>) * &theta;<sub>3</sub>) + (x-&theta;<sub>0</sub>) * &theta;<sub>2</sub> + &theta;<sub>4</sub> ;   &sigma;(x) = sigmoid((x - &theta;<sub>5</sub>) * &theta;<sub>6</sub>) * ((x-&theta;<sub>5</sub>) * &theta;<sub>8</sub>)  + (x-&theta;<sub>5</sub>) * &theta;<sub>7</sub> + &theta;<sub>9</sub>
  - No noise is added to the simulated values, they are drawn directly from the simulated ground truth probability distribution

### Estimate Normative Model (percentile curves)

For GAMLSS models:

  `normative_modelling_gamlss.R ????`
  
  - Some notes go here ..........

For sliding window models:

  `est_percentiles.py --inputfile simdata_1000.csv --out fit_MovingAvW5_1000.csv --postsmooth 2.12 --percbinsize 5.0 --binwidth 1 --bintype rect`
  
  - This takes a set of simulated datasets (as output by sim_data.py) and estimates percentile curves using normative model based on sliding windows
  - The output is written to the file fit_MovingAvW5_1000.csv in this case and the format contains one row per datapoint with a header row like this:
    - *age, simdata_1, centile1_sim1, centile2_sim1, centile5_sim1, centile10_sim1, centile50_sim1, centile90_sim1, centile95_sim1, centile98_sim1, centile99_sim1, simdata_2, centile1_sim2, centile2_sim2, centile5_sim2, centile10_sim2, centile50_sim2, centile90_sim2, centile95_sim2, centile98_sim2, centile99_sim2, simdata_3, ...*
    - where centileP_simN represents the estimated value of percentile curve "P" at the Nth simulated dataset, while "simdata_N" represents the value (e.g. volume) of the this datapoint for the Nth simulated dataset
    - for example, in this case there will be one header row followed by 1000 rows (one per datapoint) where in each row will start with "age", "simdata_1" and go to "simdata_5000", "centile1_sim5000", ... "centile99"sim5000"
    - this is a redundant representation, which is helpful for matching files if the filenames do not contain enough information
  - the set of estimated percentiles can be specified using the `--estpercs` option
  - settings for the sliding window algorithm are specified via the options:
    - `--bintype` specifies the type of window function - it can be `rect` of `guassian` for fixed width windows of different weighting profiles or `fixedP` for a rectangular window that varies in width to fit a given percentage of the samples
    - `--percbinsize` specifies the width of the window (the value is interpretted as the percentage for the `fixedP` bintype and as an age range for the fixed width bintypes)
    - `--binwidth` specifies the sampling of the age values throughout the code (typically use a value of 1.0 unless some coarser sampling is desired)
    - `--postsmooth` specifies the amount of smoothing the estimated curves after the initial sliding window calculation (as a sigma value)
  - In the example here the window is a fixed size, of width 5 years, and is smoothed by &sigma; of 2.12 years (FWHM of 5 years). 
  
  ### Measure Performance/Errors
  
  `measure_results.py --inputfile fit_MovingAvW5_1000.csv --outname normodres_MovingAvW5 --estname MovingAvW5 --simmodeltype nonlin --simparams 65.0 0.1 20.0 -70.0 6000.0 65.0 0.1 1.0 5.0 200.0`
  
  - This takes as input the output of est_percentiles.py and measures different types of errors, saving these in an output file (normodres_MovingAvW5.csv) and creating plots of the results (saved as files)
  - The output of any normative modelling method can be used as long as its output is written using the format specified above, although both csv and rds file formats can be read
  - The plots are saved with fixed filenames, into the present working directory
  - The type of ground truth and the associated simulation parameters are specified with the `--simmodeltype` and `--simparams` options, as described above for sim_data.py (it should match the ground truth used to generate the initial simulated data)
  - The estimation method name specified by `--simmodeltype` is used in the output file, as a way of identifying the output when comparing methods
  - The output basename, specified by `--outname`, is used to form the name of the csv file that saves all the error values
    - Columns in the csv file include:
     - *Simulation Model* is the name of the ground truth model (e.g. nonlin)
     - *Estimation Method* is the name given in the `-e` option
     - *Error Type* is E1 or E2; Summary Type is Median, IQR, CI95, etc
     - *Age Bin* is the centre value of the bin in the age histogram (used for sampling in the age-specific summaries) or 'All' if it summaries over ages
     - *Simulation Run* is 'All' for all summaries (currently the only outputs)
     - *Value* is the numerical value of the performance/error summary
  - Note that there is a lot of verbose output to standard output, and if this is unwanted then it can be redirected into a file by appending ` > filename` to the end of the command (this is not special for this command but a general unix/linux way of saving output into a file)


## Installation instructions

The code requires R version ???? and python version 3 with the following python packages:
 - pyreadr
 - scipy
 - numpy
 - matplotlib
 - pandas
 
All of these packages are very standard except for `pyreadr`, and this can be installed with either of the following methods:
 - `pip install pyreadr`
 - `conda install -c conda-forge pyreadr`
