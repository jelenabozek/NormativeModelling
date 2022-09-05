#!/usr/bin/env Rscript


# Script for running normative modelling using GAMLSS
# The script can be run from the command line or in RStudio.
# The script uses parallel processing to speed things up (SHASH takes longer time, especially with larger datasets).

# Author: Jelena Bozek, University of Zagreb, Faculty of Electrical Engineering and Computing

# August 2022

####################################

# load libraries
library(gamlss)
library(lme4)

# library for parsing inputs and python-like style for options
library(optparse)

# libraries for parallelisation
library(foreach)
library(doParallel)


######## HELPER FUNCTIONS ########
# function for getting percentiles from gamlss object; based on the centiles() function (but centiles() only plots, doesn't output values)
get_percentiles <- function(obj, xvar, cent=c(1,2,5,10,50,90,95,98,99)) {
  if (!is.gamlss(obj))  stop(paste("This is not a gamlss object", "\n", ""))
  fname <- obj$family[1]
  qfun <- paste("q",fname,sep="")
  lpar <- length(obj$parameters)
  ii <- 0
  ll <- list()
  for (var in cent)   { 
    if(lpar==1)     {
      newcall <-call(qfun,var/100,
                     mu=fitted(obj,"mu")[order(xvar)]) 
    }
    else if(lpar==2)    {
      newcall <-call(qfun,var/100,
                     mu=fitted(obj,"mu")[order(xvar)],
                     sigma=fitted(obj,"sigma")[order(xvar)]) 
    }
    else if(lpar==3)    {
      newcall <-call(qfun,var/100,
                     mu=fitted(obj,"mu")[order(xvar)],
                     sigma=fitted(obj,"sigma")[order(xvar)],
                     nu=fitted(obj,"nu")[order(xvar)])
    }
    else     {
      newcall <-call(qfun,var/100,
                     mu=fitted(obj,"mu")[order(xvar)],
                     sigma=fitted(obj,"sigma")[order(xvar)],
                     nu=fitted(obj,"nu")[order(xvar)],
                     tau=fitted(obj,"tau")[order(xvar)]) 
    }    
    ii <- ii+1
    ll[[ii]] <- eval(newcall)
  }
  centiles = do.call(cbind, ll)
  return (centiles)
}

############ PARSING INPUTS ############ 

option_list <- list(
  make_option(c("-f", "--file"), type="character", default=NULL, 
              help="dataset file name in .rds format; firts column is age, followed by columns with a measure to model (e.g. hippocampal volume)", metavar="character"),
  make_option(c("-n", "--numsubjects"), type="character", default="50", 
              help="number of subjects in the dataset [default= %default]", metavar="character"),
  make_option(c("--func"), type="character", default="splineMuSigma", 
              help="formula function to be used for fitting, can be one of the following: 'linear', 'polynomial', 'spline', 'splineMuSigma', 'splineMuSigmaNuTau'. [default= %default]", metavar="character"),
  make_option(c("-d", "--distribution"), type="character", default="SHASH", 
              help="family distribution to use for modelling, can be 'BCT' or 'SHASH'. [default= %default]", metavar="character"),
  make_option(c("-o", "--out"), type="character", default="fitdata_gamlss.rds", 
              help="output file name with extension .rds [default= %default]", metavar="character")
); 

opt_parser <- OptionParser(option_list=option_list);
opt <- parse_args(opt_parser);


# test if there is at least one argument: if not, return an error
if (is.null(opt$file)){
  print_help(opt_parser)
  stop("At least one argument must be supplied (input file).", call.=FALSE)
}


###### assigning inputs to variables ######
input_file <- opt$file
num_of_subj <- opt$numsubjects
formula <- opt$func
fam <- opt$distribution
output_file <- opt$out

  

############ MAIN PART ############ 

# detect number of CPUs for parallel computing
nC <- detectCores()
registerDoParallel(nC)  # use multicore, set to the number of available cores


# load simulated data
my_data <- readRDS(input_file)

# convert all columns to numeric
my_data[] <- lapply(my_data, function(x) {
  as.numeric(as.character(x))
})
sapply(my_data, class)

age <- my_data[,1] # ages
## round ages to closest integer value - only for N=50000; 
if (num_of_subj==50000) { age <- round(age) }

# get number of columns in simulated sets
ncolumns <- ncol(my_data)
y_vol <- my_data[,2:ncolumns] # volumes; columns 2:ncolumns have different simulation sets
numcol <- ncol(y_vol)

# initilise output data frame with input ages
# output file will have the following structure: 1st column ages, 2nd column input simdata, followed by columns with percentiles (output of fitted.values), then again input simdata, percentiles etc.
out_data <- data.frame(age)

start_time <- Sys.time()
my_centiles <- foreach (i=1:numcol, .combine= rbind) %dopar%    # start parallel processing
  {
    if (formula == 'linear') {
      my_fit <- gamlss(y_vol[,i] ~ age, family=fam, data = my_data)
    }
    else if (formula == 'polynomial'){
      my_fit <- gamlss(y_vol[,i] ~ poly(age,3), family=fam, data=my_data)
    }
    else if (formula == 'spline') {
      my_fit <- gamlss(y_vol[,i]~cs(age,df=3), family=fam, data=my_data)
    }
    else if (formula == 'splineMuSigma') {
      my_fit <- gamlss(y_vol[,i]~cs(age,df=3), sigma.formula = ~cs(age,df=3), family=fam, data=my_data)
    }
    else if (formula == 'splineMuSigmaNuTau') {
      my_fit <- gamlss(y_vol[,i]~cs(age,df=3), sigma.formula = ~cs(age,df=3), nu.formula = ~cs(age,df=3), tau.formula = ~cs(age,df=3), family=fam, data=my_data)
    }
    return (list(get_percentiles (my_fit,age)))
  }

for (i in 1:numcol) {
  #  placing all percentiles into single file
  out_data[,paste0("simdata_",i)] <- y_vol [,i]
  out_data[,paste0("centile1_sim",i)] <- my_centiles[[i]][,1]
  out_data[,paste0("centile2_sim",i)] <- my_centiles[[i]][,2]
  out_data[,paste0("centile5_sim",i)] <- my_centiles[[i]][,3]
  out_data[,paste0("centile10_sim",i)] <- my_centiles[[i]][,4]
  out_data[,paste0("centile50_sim",i)] <- my_centiles[[i]][,5]
  out_data[,paste0("centile90_sim",i)] <- my_centiles[[i]][,6]
  out_data[,paste0("centile95_sim",i)] <- my_centiles[[i]][,7]
  out_data[,paste0("centile98_sim",i)] <- my_centiles[[i]][,8]
  out_data[,paste0("centile99_sim",i)] <- my_centiles[[i]][,9]
  
}
end_time <- Sys.time()

# save data as binary .rds file
saveRDS(out_data, file = output_file)
