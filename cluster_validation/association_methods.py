import numpy as np
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.cosmology import FlatLambdaCDM

#####################################################
#----------------------------------------------------
#####################################################

#####################################################
#association method based on volume within a cylinder
#####################################################

#_____________________________________________________
def r_scaled(mass, z, delta_ovd, cosmo, density_type="critical"):
    """mass in Msun, z = redshift, delta_ovd = critical overdensity, cosmo = astropy object, density_type = 'critical' or 'mean', default to 'critical'"""
    
    if density_type=="critical":
        cosmo_density = cosmo.critical_density(z)
    elif density_type=="mean":
        cosmo_density = cosmo.mean_density(z)
        
    r_scaled = (3/4.* mass * u.solMass/(np.pi*delta_ovd*cosmo_density.to(u.solMass/u.Mpc**3)))**(1/3.)
    return r_scaled.value


#_____________________________________________________
def r_richness_scaled(richness):
    """Richness scaled radius defined as in Rozo et al. 2015b (https://arxiv.org/pdf/1410.1193.pdf)"""
    
    r_scaled = 1. * u.Mpc * ((richness/100)**0.2)
    return r_scaled.value

#_____________________________________________________
def search_distance(truth_data, cluster_data, r_max, r_max_type, cosmo, association_way, delta_ovd=200):
    
    if association_way=="1w":
        redshift = truth_data['redshift']
    elif association_way=="2w":
        redshift = cluster_data['redshift'] 
    else:
        print("error: wrong association way")
    
    arcmin_to_Mpc_conv = cosmo.arcsec_per_kpc_proper(redshift).to(u.arcmin/u.Mpc)
    
    if r_max_type=="fixed_angle":
        theta_max = r_max * u.arcmin * np.ones_like(redshift)
        
    elif r_max_type=="fixed_dist":
        theta_max = r_max * u.Mpc * arcmin_to_Mpc_conv
        
    elif r_max_type=="scaled": 
        if association_way=="1w":
            r_max_scaled = r_scaled(truth_data['halo_mass'], truth_data['redshift'], delta_ovd, cosmo, density_type="critical")
        elif association_way=="2w":
            r_max_scaled = r_richness_scaled(cluster_data['richness'])                       
        theta_max = r_max * r_max_scaled * u.Mpc * arcmin_to_Mpc_conv
   
    else:
        print("error: wrong r_max_type name")
        
    return theta_max
    

#_____________________________________________________
def volume_associations(truth_data, cluster_data, theta_max, delta_zmax, association_way, coo_DC2=None, coo_RM=None):
    
    """Perform volume association based on theta_max and delta_zmax and return number and list of associations"""
    
    if coo_DC2 == None:
        coo_DC2 = SkyCoord(truth_data['ra']*u.deg,truth_data['dec']*u.deg)
        
    if coo_RM == None:
        coo_RM  = SkyCoord(cluster_data['ra']*u.deg, cluster_data['dec']*u.deg)
        
    
    if association_way=="1w": 
        redshift_ref, redshift_cat = truth_data['redshift'], cluster_data['redshift']
        coo_ref, coo_cat = coo_DC2, coo_RM
        
    elif association_way=="2w": 
        redshift_ref, redshift_cat = cluster_data['redshift'], truth_data['redshift']
        coo_ref, coo_cat = coo_RM, coo_DC2
        
    number_of_volume_match  = np.zeros(redshift_ref.size)
    list_of_volume_match    = np.empty(redshift_ref.size, dtype=object)   
    ind = np.linspace(0,redshift_cat.size-1,redshift_cat.size, dtype=int)

    for i in range(redshift_ref.size):
        
        spatial_candidates = coo_ref[i].separation(coo_cat).arcmin*u.arcmin<theta_max[i]        
        volume_candidates = (abs(redshift_ref[i]-redshift_cat[spatial_candidates])<delta_zmax*(1+redshift_ref[i]))
        number_of_volume_match[i] = np.sum(volume_candidates)
        list_of_volume_match[i]   = ind[spatial_candidates][volume_candidates]
    
    return number_of_volume_match, list_of_volume_match

#_____________________________________________________
def get_angular_distances(cat_ref, list_of_volume_match, coo_ref, coo_cat):
    
    angular_distances_of_associations = []
    
    for i in range(len(cat_ref)): 
        angular_distances_of_associations.append(coo_ref[i].separation(coo_cat[list_of_volume_match[i]]))
                                          
    return angular_distances_of_associations

#_____________________________________________________
def get_membership(cat_ref, cat, list_of_volume_match, truth_member_data, cluster_member_data, association_way):

    if association_way=="1w":
        cat_member_ref = truth_member_data
        cat_member = cluster_member_data

        label_cluster, label_member_cluster, label_member_galaxy = 'cluster_id', 'cluster_id_member', 'id_member'
        label_cluster_ref, label_member_cluster_ref, label_member_galaxy_ref = 'halo_id', 'halo_id', 'galaxy_id'
        
    elif association_way=="2w":
        cat_member_ref = cluster_member_data
        cat_member = truth_member_data

        label_cluster_ref, label_member_cluster_ref, label_member_galaxy_ref = 'cluster_id', 'cluster_id_member', 'id_member'
        label_cluster, label_member_cluster, label_member_galaxy = 'halo_id', 'halo_id', 'galaxy_id'
        
    else:
        print("error: wrong association way")
        
        

    membership_of_associations = list_of_volume_match.copy()

    for i in range(len(cat_ref)): 
        
        indices1 = np.argwhere(cat_member_ref[label_member_cluster_ref]==(cat_ref[label_cluster_ref][i]))[:,0];
        ref_member_id_list = cat_member_ref[label_member_galaxy_ref][indices1].data

        cluster_id = cat[label_cluster][list_of_volume_match[i]]
        membership = []
        for k in cluster_id:
            indices2 = np.argwhere(cat_member[label_member_cluster]== k)[:,0]
            member_id_list = cat_member[label_member_galaxy][indices2].data

            gal_in_common = set(ref_member_id_list).intersection(member_id_list)
            membership.append(len([*gal_in_common]))

        membership_of_associations[i] = membership 

    return membership_of_associations


#_____________________________________________________
def select_one_association(cat_ref, number_of_volume_match, list_of_volume_match, method, coo_ref=None, coo_cat=None, cat=None, truth_member_data=None, cluster_member_data=None, association_way=None):

    ind_cat_ref_match = []
    ind_cat_match     = [] 
    
    if method=='nearest':
        angular_distances_of_associations = get_angular_distances(cat_ref, list_of_volume_match, coo_ref, coo_cat)
    elif method=='membership':   
        membership_of_associations = get_membership(cat_ref, cat, list_of_volume_match, truth_member_data, cluster_member_data, association_way)
    
    for i in range(len(cat_ref)): 
                     
        if number_of_volume_match[i]==1:
            ind_cat_match.append(int(list_of_volume_match[i]))
            ind_cat_ref_match.append(i)    
        
        if number_of_volume_match[i]>1:
                                          
            if method=='nearest':
                select = np.argmin(angular_distances_of_associations[i])
            elif method=='membership':   
                select = np.argmax(membership_of_associations[i])
            else:
                print("error: wrong method name")
                                                                                    
            ind_cat_match.append(int(list_of_volume_match[i][select]))
            ind_cat_ref_match.append(i)
                  
    return ind_cat_ref_match, ind_cat_match

            
#_____________________________________________________
def bijective_associations(ind_RM_match_1w, ind_DC2_match_1w, ind_RM_match_2w, ind_DC2_match_2w):
    
    #find the indices of the associations in the two catalogs for the two association ways.
    ind_w1 = list(zip(ind_RM_match_1w,ind_DC2_match_1w))
    ind_w2 = list(zip(ind_RM_match_2w,ind_DC2_match_2w))
    
    #take intersection of ind_w1 and ind_w2 (bijective associations)
    ind_bij = np.array(list(set(ind_w1) & set(ind_w2)))    
    
    return ind_bij


#_____________________________________________________
def volume_match(truth_data, cluster_data, delta_zmax, r_max, r_max_type, method, cosmo, truth_member_data=None, cluster_member_data=None, delta_ovd=200, density_type="critical"):
    
    """
        Perform a geometrical association between DC2 and RM catalogs. 
 
        The association is computed in two ways (DC2=>RM and RM=>DC2), the intersection of which is bijective.
         #parameters:
        :param truth_data : the truth catalog
        :param cluster_data : the detection catalog
        :param delta_zmax: The depth of the cylinder considered. Such as depth = 2 x delta_zmax (1+z)
        :param r_max: The radius considered, a constant in arcmin if r_max_type = "fixed_angle", Mpc if r_max_type="fixed_dist" and a fraction of R_fof of r_max_type="scaled"
        :param r_max_type: The choice of treatment for r_max, either "fixed_angle", "fixed_dist" or "scaled"    
        :param method: the choice of method to select among possible volume candidates, either "nearest" or "membership"
        :param cosmo: an astropy cosmology object
        :param truth_member_data: (optional, to be used if 'method' = 'membership') , catalog containing the membership info of the truth catalog, default to None
        :param cluster_member_data: (optional, to be used if 'method' = 'membership') , catalog containing the membership info of the detection catalog, default to None
        :param delta_ovd: (optional, to be used if 'r_max_type' = 'scaled') overdensity constrast to be used to compute the scaled radius, default to 200         
        :param density_type: (optional, to be used if 'r_max_type' = 'scaled') type of overdensity  to be used to compute the scaled radius ('mean' or 'critical'), default to 'critical'
        
        
        :type delta_zmax: float or np.inf
        :type r_max: float
        :type r_max_type: string
        :type method: string
        :type delta_ovd: float
        :type density_type: string
        
         #results:        
        :return: two vectors containing the number of associations for each entry in the two ways, and one array containing the indices of nearest associated objects in both catalogs (in a bijective way) following: [ind_RM,ind_DC2]
        :rtype: tuple containing two vectors of floats and one array of integers
        
    """
 

    if method=="membership" and (truth_member_data is None or cluster_member_data is None):
        print ("error: with membership method you should provide member catlogs, see function documentation")
        
    ###load DC2 and RM clusters coordinates
    coo_DC2 = SkyCoord(truth_data['ra']*u.deg,truth_data['dec']*u.deg)
    coo_RM  = SkyCoord(cluster_data['ra']*u.deg, cluster_data['dec']*u.deg)

    ########################################################################################
    #one way association : DC2 => RM (1w)

    ###compute searching distances-------------------------------------------------------------
    theta_max = search_distance(truth_data, cluster_data,r_max, r_max_type, cosmo, "1w", delta_ovd=delta_ovd)

    ###search and store volume candidate------------------------------------------------------
    number_of_volume_match_1w, list_of_volume_match_1w = \
    volume_associations(truth_data, cluster_data, theta_max, delta_zmax, "1w", coo_DC2, coo_RM)
    
    ###select only one match---------------------------------------------------------------------
    ind_DC2_match_1w, ind_RM_match_1w = \
    select_one_association(truth_data, number_of_volume_match_1w, list_of_volume_match_1w, method, coo_DC2, coo_RM, cluster_data, truth_member_data, cluster_member_data, "1w")
        
    ########################################################################################
    #one way association : RM => DC2 (2w)

    ###compute searching distances-------------------------------------------------------------
    theta_max = search_distance(truth_data, cluster_data,r_max, r_max_type, cosmo, "2w")
    
    ###search and store volume candidate------------------------------------------------------
    number_of_volume_match_2w, list_of_volume_match_2w = \
    volume_associations(truth_data, cluster_data, theta_max, delta_zmax, "2w", coo_DC2, coo_RM)
    
    ###select only one match---------------------------------------------------------------------
    ind_RM_match_2w, ind_DC2_match_2w = \
    select_one_association(cluster_data, number_of_volume_match_2w, list_of_volume_match_2w, method, coo_RM, coo_DC2, truth_data, truth_member_data, cluster_member_data, "2w")

    ########################################################################################       
    #bijective associations
    
    ind_bij = bijective_associations(ind_RM_match_1w, ind_DC2_match_1w, ind_RM_match_2w, ind_DC2_match_2w)

    return  number_of_volume_match_1w, number_of_volume_match_2w, ind_bij

