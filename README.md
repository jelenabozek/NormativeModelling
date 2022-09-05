# NormativeModelling

Scripts included in this repository are described in detial in the paper "Normative models for neuroimaging markers: impact of model selection, sample size and evaluation criteria" by Jelena Bozek, Ludovica Griffanti, Stephan Lau and Mark Jenkinson.

The R script normative_modelling_gamlss.R fits dataset using GAMLSS. The implementation we used for the gamlss function came from package GAMLSS (version 5.1-6). The scripts can use several GAMLSS models, implementing linear fitting or cubic spline smoothing across age, together with a Box Cox T (BCT) or a SinhArcsinh (SHASH) transformation, both of which create four parameter continuous distributions.
