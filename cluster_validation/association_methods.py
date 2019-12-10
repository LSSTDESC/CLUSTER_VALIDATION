import numpy as np
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.cosmology import FlatLambdaCDM


#association method based on volume within a cylinder

def volume_association(truth_data, cluster_data, delta_zmax, r_max, method, cosmo):
    
    """
        Perform a geometrical association between DC2 and RM catalogs. 
 
        The association is computed in two ways (DC2=>RM and RM=>DC2), the intersection of which is bijective.
         #parameters:
        :param truth_data : the truth catalog
        :param cluster_data : the detection catalog
        :param delta_zmax: The depth of the cylinder considered. Such as depth = 2 x delta_zmax (1+z)
        :param r_max: The radius considered, a constant in Mpc if method="fixed" and a fraction of R_fof of method="scaled"
        :param method: The choice of treatment for r_max, either "fixed" or "scaled"        
        :param cosmo: an astropy cosmology object
        :type delta_zmax: float
        :type r_max: float
        :type method: string
        
         #results:        
        :return: two vectors containing the number of associations for each entry in the two ways, and one array containing the indices of nearest associated objects in both catalogs (in a bijective way) following: [ind_RM,ind_DC2]
        :rtype: tuple containing two vectors of floats and one array of integers
        
    """
    
    delta_c = 200 # overdensity at which the enclosed mass is supposed to be close to the FoF mass definition
    #load DC2 and RM clusters coordinates
    coo_DC2 = SkyCoord(truth_data['ra']*u.deg,truth_data['dec']*u.deg)
    coo_RM  = SkyCoord(cluster_data['ra']*u.deg, cluster_data['dec']*u.deg)

    ########################################################################################
    #one way association : DC2 => RM
    ########################################################################################
    match_pos_num_1w = np.zeros(truth_data['redshift'].size)
    match_num_1w = np.zeros(truth_data['redshift'].size)
    ind_DC2_match_1w = []
    ind_RM_match_1w = []
    ind_RM = np.linspace(0,cluster_data['redshift'].size-1,cluster_data['redshift'].size)
    r_max_fof = np.zeros(truth_data['redshift'].size)    
    
    
    for i in range(truth_data['redshift'].size):

        if method=="fixed":
            theta_max = cosmo.arcsec_per_kpc_proper(truth_data['redshift'][i]).to(u.arcmin/u.Mpc)*r_max*u.Mpc
        
        elif method=="scaled":       
            r_max_fof[i] = ((3/4.*truth_data['halo_mass'][i]*u.solMass/(np.pi*delta_c*cosmo.critical_density(truth_data['redshift'][i]).to(u.solMass/u.Mpc**3)))**(1/3.)).value
            theta_max = cosmo.arcsec_per_kpc_proper(truth_data['redshift'][i]).to(u.arcmin/u.Mpc)*r_max*r_max_fof[i]*u.Mpc
            
        else:
            print("error: wrong method name")
            break
            
        candidates_pos = coo_DC2[i].separation(coo_RM).arcmin*u.arcmin<theta_max
        match_pos_num_1w[i] = np.sum(candidates_pos)
        candidates = (abs(truth_data['redshift'][i]-cluster_data['redshift'][candidates_pos])<delta_zmax*(1+truth_data['redshift'][i]))
        match_num_1w[i] = np.sum(candidates)
        
        if (np.sum(candidates)==1):
            ind_RM_match_1w.append(int(ind_RM[candidates_pos][candidates]))
            ind_DC2_match_1w.append(i)
            
        if np.sum(candidates)>1:
            nearest = np.where(coo_DC2[i].separation(coo_RM[candidates_pos][candidates]) == np.min(coo_DC2[i].separation(coo_RM[candidates_pos][candidates])))[0][0]
            ind_RM_match_1w.append(int(ind_RM[candidates_pos][candidates][nearest]))
            ind_DC2_match_1w.append(i)

        
    ########################################################################################
    #one way association : RM => DC2
    ########################################################################################   
    match_pos_num_2w = np.zeros(cluster_data['redshift'].size)
    match_num_2w = np.zeros(cluster_data['redshift'].size)
    ind_DC2_match_2w = []
    ind_RM_match_2w = []
    ind_DC2 = np.linspace(0,truth_data['redshift'].size-1,truth_data['redshift'].size)
    theta_max_fof = cosmo.arcsec_per_kpc_proper(truth_data['redshift']).to(u.arcmin/u.Mpc)*r_max_fof*u.Mpc
 
    for i in range(cluster_data['redshift'].size):
        
        if method=="fixed":
            theta_max = cosmo.arcsec_per_kpc_proper(cluster_data['redshift'][i]).to(u.arcmin/u.Mpc)*r_max*u.Mpc     
            candidates_pos = coo_RM[i].separation(coo_DC2).arcmin*u.arcmin<theta_max
       
        elif method=="scaled":          
            theta_max = cosmo.arcsec_per_kpc_proper(cluster_data['redshift'][i]).to(u.arcmin/u.Mpc)*np.max(r_max_fof)*u.Mpc   
            candidates_pos_rmax = coo_RM[i].separation(coo_DC2).arcmin*u.arcmin<theta_max        
            candidates_pos = candidates_pos_rmax
            if (np.sum(candidates_pos_rmax)>0):
                for j in ind_DC2[candidates_pos_rmax]:
                    candidates_pos[int(j)] = candidates_pos_rmax[int(j)]*(coo_RM[i].separation(coo_DC2[int(j)]).arcmin*u.arcmin<theta_max_fof[int(j)])   
      
        else:
            print("error: wrong method name")
            break
        
        match_pos_num_2w[i] = np.sum(candidates_pos)
        candidates = (abs(cluster_data['redshift'][i]-truth_data['redshift'][candidates_pos])<delta_zmax*(1+cluster_data['redshift'][i]))
        match_num_2w[i] = np.sum(candidates)
        
        if (np.sum(candidates)==1):
            ind_DC2_match_2w.append(int(ind_DC2[candidates_pos][candidates]))
            ind_RM_match_2w.append(i)
            
        if np.sum(candidates)>1:
            nearest = np.where(coo_RM[i].separation(coo_DC2[candidates_pos][candidates]) == np.min(coo_RM[i].separation(coo_DC2[candidates_pos][candidates])))[0][0]
            ind_DC2_match_2w.append(int(ind_DC2[candidates_pos][candidates][nearest]))
            ind_RM_match_2w.append(i)


    ########################################################################################       
    #bijective associations
    ########################################################################################
    
    #find the indices of nearest associations in the two catalogs for the two association ways.
    ind_w1 = list(zip(ind_RM_match_1w,ind_DC2_match_1w))
    ind_w2 = list(zip(ind_RM_match_2w,ind_DC2_match_2w))
    
    #take intersection of ind_w1 and ind_w2 (bijective associations)
    ind_bij = np.array(list(set(ind_w1) & set(ind_w2)))

    return  match_num_1w, match_num_2w, ind_bij