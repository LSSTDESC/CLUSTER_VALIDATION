import numpy as np
import GCRCatalogs
from astropy.table import Table

#####################################################
#----------------------------------------------------
#####################################################

#TODO 
#- Add docstring to RM_DC2_cat_open


#####################################################
#function to open, read and select info in DC2 and redmapper catalogs
#####################################################


def RM_DC2_cat_open(RM_cat_name, DC2_cat_name, min_richness=20, min_halo_mass=1e14, redshift_max = 1.3, cluster_only=True, mag_query=None, RM_only=False):

    # Get the redMaPPer catalog
    gc = GCRCatalogs.load_catalog(RM_cat_name)
    # Select out the cluster and member quantities into different lists
    quantities = gc.list_all_quantities()
    cluster_quantities = [q for q in quantities if 'member' not in q]
    member_quantities = [q for q in quantities if 'member' in q]
    
    # Read in the cluster and member data
    query = GCRCatalogs.GCRQuery('(richness > ' + str(min_richness) +')')
    cluster_data = Table(gc.get_quantities(cluster_quantities, [query]))
    member_data = Table(gc.get_quantities(member_quantities))
    
    if RM_only:
        return cluster_data, member_data, gc
    else :
        #read in the "truth" catalog as a comparison (can take a while...)
        gc_truth = GCRCatalogs.load_catalog(DC2_cat_name)  
        quantities_wanted = ['redshift','halo_mass','halo_id','galaxy_id','ra','dec', 'is_central']
        if mag_query:
            quantities_wanted = ['redshift','halo_mass','halo_id','galaxy_id','ra','dec', 'is_central', 'mag_true_u_lsst', 'mag_true_g_lsst', 'mag_true_r_lsst', 'mag_true_i_lsst', 'mag_true_z_lsst', 'mag_true_y_lsst']
        if cluster_only :
            query = GCRCatalogs.GCRQuery('(is_central == True) & (halo_mass > ' + str(min_halo_mass) +')& (redshift < ' + str(redshift_max) +')')
        else :
            query = GCRCatalogs.GCRQuery('(halo_mass > ' + str(min_halo_mass) +') & (redshift < ' + str(redshift_max) +')')
        
        truth_data = Table(gc_truth.get_quantities(quantities_wanted, [query]))
    
        return cluster_data, member_data, truth_data, gc, gc_truth

