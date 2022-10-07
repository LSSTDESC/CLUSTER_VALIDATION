import numpy as np
import matplotlib.pyplot as plt

#####################################################
#----------------------------------------------------
#####################################################


#####################################################
#functions to produce association validation plots
#####################################################

#____________________________________________________
#detection vs true redshift plots

def plot_redshift_comparison(truth_data, cluster_data, ind_bij):
    """ind1, ind2 = ind_truth_data, ind_cluster_data"""
    
    plt.subplot(2,1,1)
    plt.plot(truth_data['redshift'][ind_bij[:,1]],cluster_data['redshift'][ind_bij[:,0]],'k.',alpha=0.2)
    plt.ylabel('$z_{RM}$')
    plt.xlabel('$z_{DC2}$');
    plt.grid()

    plt.subplot(2,1,2)
    plt.plot(truth_data['halo_mass'][ind_bij[:,1]],(cluster_data['redshift'][ind_bij[:,0]]-truth_data['redshift'][ind_bij[:,1]])/(1+truth_data['redshift'][ind_bij[:,1]]),'k.',alpha=0.2)
    plt.xscale('log')
    plt.xlabel('$M_{fof}$')
    plt.ylabel('$\\frac{(z_{RM}-z_{DC2})}{(1+z_{DC2})}$');
    plt.grid()

    plt.subplots_adjust(hspace = 0.6)
    
    return 

#____________________________________________________
#position of associated clusters

def plot_cluster_and_halo_position(truth_data, cluster_data, ind_1w, ind_2w, ind_bij):
    """ind_1w, ind_2w = ind_truth_data, ind_cluster_data, bijective_ind"""
    
    fig, (ax1, ax2) = plt.subplots(1,2, figsize = (10,8))
    ax1.set_title('truth')
    ax1.plot(truth_data['ra'], truth_data['dec'], 'k.', label='all')
    ax1.plot(truth_data['ra'][ind_1w>0], truth_data['dec'][ind_1w>0], 'ro',alpha=0.5, label ='one way')
    ax1.plot(truth_data['ra'][ind_bij[:,1]], truth_data['dec'][ind_bij[:,1]], 'bx', label ='bijective')
    ax1.set_xlabel("ra")
    ax1.set_ylabel("dec")
    ax1.legend(fontsize='small', borderpad = -0.1, handletextpad =0.1) ;

    ax2.set_title('detection')
    ax2.plot(cluster_data['ra'], cluster_data['dec'], 'k.')
    ax2.plot(cluster_data['ra'][ind_2w>0], cluster_data['dec'][ind_2w>0], 'ro',alpha=0.5)
    ax2.plot(cluster_data['ra'][ind_bij[:,0]], cluster_data['dec'][ind_bij[:,0]], 'bx')   
    ax2.set_xlabel("ra")
    ax2.set_ylabel("dec");
    
    fig.subplots_adjust(hspace = 0.6)
    
    return fig, ax1, ax2

#####################################################
#functions to plot richness mass relations
#####################################################

#____________________________________________________
#plot richness a a function of mass

def plot_richness_mass(halo_data, cluster_data, ind_bij, zmin = 0, zmax = np.inf, min_richness=None, min_halo_mass=None, fig=None, ax=None, fmt='k.'):
    
    alpha_scale = 500/len(halo_data) #arbitrary
    
    if ax!=None and fig!=None:
        fig, ax = fig, ax
    
    else:
        fig, ax = plt.subplots(1,1, sharex=True, sharey=True)
    
    z_truth = halo_data['redshift'][ind_bij[:,1]]
    cond = (z_truth>zmin) & (z_truth<zmax)
    
    ax.loglog(halo_data['halo_mass'][ind_bij[:,1]][cond],cluster_data['richness'][ind_bij[:,0]][cond],fmt,alpha=alpha_scale, label = '_');
    
    ax.set_xlabel('$M_{fof}$');
    ax.set_ylabel('$\lambda_{RM}$')
    
    if min_richness!=None:
        ax.axhline(min_richness,linestyle='dotted',color='black')
    if min_halo_mass!=None:
        ax.axvline(min_halo_mass,linestyle='dotted',color='black')
        
    return fig, ax


#____________________________________________________
#plot mass a a function of richness

def plot_mass_richness(halo_data, cluster_data, ind_bij, zmin = 0, zmax = np.inf, min_richness=None, min_halo_mass=None, fig=None, ax=None, fmt='k.'):
    
    alpha_scale = 500/len(halo_data) #arbitrary
    
    if ax!=None and fig!=None:
        fig, ax = fig, ax
    
    else:
        fig, ax = plt.subplots(1,1, sharex=True, sharey=True)
    
    z_truth = halo_data['redshift'][ind_bij[:,1]]
    cond = (z_truth>zmin) & (z_truth<zmax)
    
    ax.loglog(cluster_data['richness'][ind_bij[:,0]][cond], halo_data['halo_mass'][ind_bij[:,1]][cond],fmt, alpha=alpha_scale);
    
    ax.set_ylabel('$M_{fof}$');
    ax.set_xlabel('$\lambda_{RM}$')
    
    if min_richness!=None:
        ax.axvline(min_richness,linestyle='dotted',color='black')
    if min_halo_mass!=None:
        ax.axhline(min_halo_mass,linestyle='dotted',color='black')
        
    return fig, ax


#####################################################
#functions to add richness-mass relations from litterature
#####################################################

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
    return mass_richness_parametrization(l, z,  F, G, Omega_m_z0, M0, l0, z0, mass_def="FoF")


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
