import numpy as np
import matplotlib.pyplot as plt


#functions to produce association validation plots

#detection vs true redshift plots

def plot_redshift_comparison(truth_data, cluster_data, ind1, ind2):
    #ind1, ind2 = ind_truth_data, ind_cluster_data
    plt.subplot(2,1,1)
    plt.plot(truth_data['redshift'][ind1],cluster_data['redshift'][ind2],'k.',alpha=0.2)
    plt.ylabel('$z_{RM}$')
    plt.xlabel('$z_{DC2}$');
    plt.grid()

    plt.subplot(2,1,2)
    plt.plot(truth_data['halo_mass'][ind1],(cluster_data['redshift'][ind2]-truth_data['redshift'][ind1])/(1+truth_data['redshift'][ind1]),'k.',alpha=0.2)
    plt.xscale('log')
    plt.xlabel('$M_{fof}$')
    plt.ylabel('$\\frac{(z_{RM}-z_{DC2})}{(1+z_{DC2})}$');
    plt.grid()

    plt.subplots_adjust(hspace = 0.6)
    
    return 