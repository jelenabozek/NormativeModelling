# created on 11 Feb 2020 by jelena
# updated in August 2021 - reduce the number of methods tested

# load libraries
library(gamlss)
library(lme4)

# libraries for parallelisation
library(foreach)
library(doParallel)


#### HELPER FUNCTIONS ####
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

############ MAIN PART ############ 


## SETTING INPUTS ##

# setting working directory
setwd("~/projekti/Normative_modelling")

# folder name in working directory with simulated data
folderName <- c('LinMean_ConstVar', 'NonLinMean_NonConstVar') # << this was run in Aug 2021 ; previously >> 'LinMean_LinVar', 'LinMean_ConstVar', 'NonLinMean_ConstVar')
#folderName <- c('NonLinMean_ConstVar') # additional run in Nov 2021

#  simulated data sets have different number of subjects
num_of_subjects <- c(50, 100, 200, 500, 1000, 2000, 5000, 50000)  
#num_of_subjects <- c(1000, 5000) # testing limited number, Feb 2022


# set different family functions to fit
#formula_function <- c('linear', 'spline') 
#formula_function <- c('spline')
#formula_function <- c('linear')
formula_function <- c('splineMuSigma') # this was set up in Aug 2021, as in Dinga et al.
#formula_function <- c('splineMuSigmaNuTau') # this was set up in Feb 2022, while writing up the paper; just for testing

# set family distribution to use in folder name and in modelling of gamlss
families <- c('BCT', 'SHASH') # 'BCPE' used in Lisa Nobis' paper does not work (she uses lms() function, not gamlss())
#families <- c('SHASH') 

### DONE WITH INPUTS ###

# detect number of CPUs for parallel computing
nC <- detectCores()
registerDoParallel(nC)  # use multicore, set to the number of our cores

for (fam in families) {
  for (fold in folderName) {
    
    # create output directory if it doesn't exist already; set path to the directory
    outdir<- sprintf("fit_gamlss_%s/fit_%s",fam, fold)
    dir.create(file.path('~/projekti/Normative_modelling/', outdir), recursive=TRUE)
    print (fold)
    
    for (formula in formula_function ) {
      print (formula)
      
      for (num_of_subj in num_of_subjects) {
        print (num_of_subj)
        
        # load simulated data
        #my_data <- read.csv(sprintf("sim_%s/simdata_%s.csv",fold,num_of_subj),sep = ",", header=FALSE)
        my_data <- readRDS(sprintf("sim_%s/simdata_%s.rds",fold,num_of_subj)) #ß,sep = ",", header=FALSE)

        # convert all columns to numeric (for some reason they are loaded as character)
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
        
        # initilise output data frame with only ages
        # output file should have the following structure: 1st column ages, 2nd column simdata, followed by columns with percentiles (output of fitted.values), then again sim data etc
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
        
        # save out_data in a file for that formula and number of subjects
        #write.csv(out_data, sprintf("%s/fitdata_gamlss_%s_%s.csv",outdir,formula,num_of_subj), row.names = F)
        
        # save data as binary .rds file
        saveRDS(out_data, file = sprintf("%s/fitdata_gamlss_%s_%s.rds",outdir,formula,num_of_subj))
      }
    }
  }
}



## code for plotting percentiles (checking if fitting runs well) ##

# set margins for plot, otherwise getting an error that figure margins are too large
#par(mar=c(1,1,1,1))

#my_centiles <- get_percentiles (my_fit,age) 
# plotTitle <- sprintf("%s %s, iter $s",fold, fam, i)
# matplot (age, my_centiles[], type='l', main = plotTitle)
# matpoints (age, y_vol[,i], pch = "*")
# cent_lin <- get_percentiles (my_fit_linear,age) 
#cent_poly <- get_percentiles(my_fit_poly, age)
#cent_spline <- get_percentiles(my_fit_spline, age)
# create plots to see if function output is the same as output from centiles() -> it is
#matplot (age, cent_lin[], type='l', xlim = range(age), ylim = range(my_fit_linear$y),) #,cent_lin[,-1],type='l')
#matpoints (age, y_vol[,i], pch = ".")
#centiles(my_fit_linear, age, cent = c(1,2,5,10,50,90,95,98,99), legend=FALSE)
#matplot (age, cent_poly[],type='l', xlim = range(age), ylim = range(my_fit_poly$y),) #cent_poly[,-1], type = 'l')
#centiles(my_fit_poly, age, cent = c(1,2,5,10,50,90,95,98,99))
#matplot (age, cent_spline[], type='l', xlim = range(age), ylim = range(my_fit_spline$y),) #cent_spline[,-1], type = 'l')
#centiles(my_fit_spline,age, cent = c(1,2,5,10,50,90,95,98,99))
