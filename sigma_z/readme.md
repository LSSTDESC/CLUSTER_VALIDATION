
**Code written by Matt Kwiecien**

This directory contains the code and data necessary to generate a plot that compares the sigma-z distribution for cosmoDC2 and DC2 to DES Y1 and buzzard sims.

The `sigma_z_cosmoDC2_fit.ipynb` notebook is annotated and explains how to generate the plots. It's all wired to be run on NERSC but isn't necessarily required. You would just need the truth catalogs locally set up with GCRCatalogs.

The structure of that code is as follows :

- DataLoader is just a simple abstract class which is implemented by GCRDataloader that wraps around GCRCatalogs. This seems unnecessary because we are only using GCRCatalogs, but is nice if we load data from FITS as well (which we are doing in my other repo where this code is from)
- PlotHelper houses all the matplotlib code
- LambdaModel contains the lambda(z) curve fit from Matteo Costanzi's paper.
- SigmaZ is a base class that contains the logic to fit the lambda(z) curves. This uses parallel processing so it's nice and fast on NERSC
- CosmoDC2 and DC2 both calculate sigma-z in a slightly different way, so each have their own implementation of SigmaZ class, e.g. DC2SigmaZ and CosmoDC2SigmaZ

