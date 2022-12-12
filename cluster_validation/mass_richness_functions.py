import numpy as np
import matplotlib.pyplot as plt
import emcee
import corner
#####################################################
# ----------------------------------------------------
# ####################################################



#####################################################
# functions to fit and plot mass richness relations
# ####################################################

#mean of the relation
def mu_logM_lambda(redshift, logrichness, mu0, G_z_mu, G_lambda_mu):
    richness_0 = 40
    z_0 = 0.4
    return mu0 + G_z_mu*np.log10((1+redshift)/(1 + z_0)) + G_lambda_mu*(logrichness-np.log10(richness_0))


#chi2
def sum_chi2(data, model, error=None):
    if error==None:
        y =  (data - model)**2
    else :
        y = (data - model)**2/error**2
    return np.sum(y)

#likelihood
def lnL(data, theta):
    mu0, G_z_mu, G_lambda_mu = theta
    return -0.5*sum_chi2(data[0],  mu_logM_lambda(data[1], data[2], mu0, G_z_mu, G_lambda_mu) )

def sampler_prep(init_positions, data, likelihood=lnL):
    nwalkers, ndim = init_positions.shape
    return emcee.EnsembleSampler(nwalkers, ndim,  lambda y: likelihood(data, y))

def plot_chains(labels, sampler):
    fig, axes = plt.subplots(3, figsize=(10, 7), sharex=True)
    samples = sampler.get_chain()
    for i in range(3):
        ax = axes[i]
        ax.plot(samples[:, :, i], "k", alpha=0.3)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)
        
def plot_corner(fig, flat_samples, labels, bins=20, color='magenta'):
    
    corner.corner(
    flat_samples,
    bins=bins, levels=1. - np.exp(-(np.array([1,2,3])/1.)**2/2.),
    fig = fig,
        smooth1d=False,smooth=False,plot_datapoints=True,
                  fill_contours=True, labels = labels,
                         
    color=color,
    label_kwargs={"fontsize": 20},
                  use_math_text=True,
                        plot_density=False,
                        max_n_ticks = 5,
    );

#####################################################
# functions to add richness-mass relations from litterature
# ####################################################

#____________________________________________________
#parametrization of the mass richness relation 


def mass_richness_parametrization(l, z, Omega_m_z0, M0, l0, z0, F, G, mass_def="crit"):
    """Parametrization of the mass-richness-redshift relation used in several DES papers """
    if mass_def == "mean":
        M = M0
    elif mass_def == "crit":
        M = M0 * Omega_m_z0*(1.+z)**3
#    elif mass_def == "FoF":
#        M = M0 * Omega_m_z0*(1.+z)**3 / 0.92
        
    return M*(l/l0)**F*((1+z)/(1+z0))**G

#____________________________________________________
#parameters of mass richness relations from litterature
#partly adapted from https://github.com/LSSTDESC/DC2-analysis/blob/u/shenmingfu/cosmoDC2-cluster-mass-richness/contributed/cosmoDC2-cluster-mass-richness.ipynb

#------------------------------
def M_Saro(l, z, Omega_m_z0):
    """Mass richness parameters from McClintock et al. (2018) [https://arxiv.org/pdf/1805.00039.pdf]"""
    
    M0 = 2.754e14   #(+-0.075 +-0.133)e14
    # log10(M0) = 14.489 #+-0.011 +-0.019
    l0 = 40.
    z0 = 0.35
    F = 0.91   #+-0.051 +-0.008
    G = 0   #+-0.30 +-0.06
    return mass_richness_parametrization(l, z, Omega_m_z0, M0, l0, z0, F, G, mass_def="crit")


#------------------------------
def M_DES_SV(l, z, Omega_m_z0):
    """Mass richness parameters from Melchior et al. (2017) [https://arxiv.org/pdf/1610.06890.pdf], the richness is updated to match that of DES Y1, following the equations given in McClintock  et al. 2018"""
        
    l_new = l/(1.08) #1.08+-0.16
    M0 = 2.3496e14   #
    # log10(M0) = 14.371
    l0 = 30.
    z0 = 0.5
    F = 1.12
    G = 0.18   

    return mass_richness_parametrization(l_new, z, Omega_m_z0, M0, l0, z0, F, G, mass_def="crit")


#------------------------------
def M_SDSS(l, z, Omega_m_z0):
    """Mass richness parameters from Simet et al. (2017) [https://arxiv.org/pdf/1603.06953.pdf], slightly revised by McClintock et al. (2018), the richness in updated to match that of DES Y1, following the equations given in McClintock  et al. 2018"""
    
    l_new = l/(0.93) #0.93+-0.14
    M0 = 3.020e14
    # log10(M0) = 14.48 +- 0.03
    l0 = 40.
    z0 = 0.2
    F = 1.30   #+- 0.09

    return mass_richness_parametrization(l_new, z, Omega_m_z0, M0, l0, z0, F, 0, mass_def="crit")

#------------------------------
def M_DES_Y1(l, z, Omega_m_z0):
    """Mass richness parameters from McClintock et al. (2018) [https://arxiv.org/pdf/1805.00039.pdf]"""
    
    M0 = 3.081e14   #(+-0.075 +-0.133)e14
    # log10(M0) = 14.489 #+-0.011 +-0.019
    l0 = 40.
    z0 = 0.35
    F = 1.356   #+-0.051 +-0.008
    G = -0.30   #+-0.30 +-0.06

    return mass_richness_parametrization(l, z, Omega_m_z0, M0, l0, z0, F, G, mass_def="crit")


#------------------------------
def M_DES_Y1_lim(l, z, lim, Omega_m_z0):
    # Assume l is an array but z is a number
    M0_min = (3.081-(0.075**2+0.133**2)**0.5)*1.0e14
    M0_max = (3.081+(0.075**2+0.133**2)**0.5)*1.0e14
    F_min = 1.356-(0.051**2+0.008**2)**0.5
    F_max = 1.356+(0.051**2+0.008**2)**0.5
    G_min = -0.30-(0.30**2+0.06**2)**0.5
    G_max = -0.30+(0.30**2+0.06**2)**0.5
    
    l0 = 40.
    z0 = 0.35
    
    arr = np.zeros_like(l)
    
    idx1 = l<=l0
    idx2 = l>l0

    if lim=='min':
        if z<=z0:
            arr[idx1] = M0_min*(l[idx1]/l0)**F_max*((1+z)/(1+z0))**G_max
            arr[idx2] = M0_min*(l[idx2]/l0)**F_min*((1+z)/(1+z0))**G_max
            #print(arr)
            return arr*Omega_m_z0*(1.+z)**3
        else:
            arr[idx1] = M0_min*(l[idx1]/l0)**F_max*((1+z)/(1+z0))**G_min
            arr[idx2] = M0_min*(l[idx2]/l0)**F_min*((1+z)/(1+z0))**G_min
            return arr*Omega_m_z0*(1.+z)**3

    elif lim=='max':
        if z<=z0:
            arr[idx1] = M0_max*(l[idx1]/l0)**F_min*((1+z)/(1+z0))**G_min
            arr[idx2] = M0_max*(l[idx2]/l0)**F_max*((1+z)/(1+z0))**G_min
            return arr*Omega_m_z0*(1.+z)**3
        else:
            arr[idx1] = M0_max*(l[idx1]/l0)**F_min*((1+z)/(1+z0))**G_max
            arr[idx2] = M0_max*(l[idx2]/l0)**F_max*((1+z)/(1+z0))**G_max
            return arr*Omega_m_z0*(1.+z)**3

    else: 
        print("Error: Limit string should be \'min\' or \'max\'.")
        return arr

    
def MOR_from_sample(flat_sample, z0=0.4):
    xval = np.logspace(np.log10(5), np.log10(200))
    mor_mean = 10**(mu_logM_lambda(z0, np.log10(xval), flat_sample[:,0].mean(), flat_sample[:,1].mean(), flat_sample[:,2].mean()))
    mor_samples = np.vstack([10**(mu_logM_lambda(z0, np.log10(xval[i]), flat_sample[:,0], flat_sample[:,1], flat_sample[:,2])) for i in range(xval.size)])
    
    return mor_mean, mor_samples