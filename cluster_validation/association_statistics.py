import numpy as np

#####################################################
#----------------------------------------------------
#####################################################
#TODO 
# - include error on fraction

#####################################################
#functions returning different statistics about the associations
#####################################################


def number_of_associations(truth_data, bij_ind):
    """ Compute the number of bijective associatons.
    
         #parameters: the truth catalog
         :param truth_data : the truth catalog
         :param bij_ind : array containing the indices of nearest associated objects in both catalogs
 
         #results:
         :return: bij_number : the total number of bijective associatons
        
    """

    bij_number = truth_data['redshift'][bij_ind[:,1]].size

    return bij_number
    
#____________________________________________________   
def overmerging(det_to_truth_match, ind_bij, method="bij"): 
    """ Compute the number of bijective associatons.
    
         #parameters: 
         :param det_to_truth_match : vector containing the number of associations going from the detection to the true catalog.
         :param ind_bij : array containing the indices of nearest associated objects in both catalogs
         :param method : the association method with respect to which is computed the fraction. Can be "one way" or "bij" (for bijective). Default to "bij".
         
         #results:
         :return: ovmg_number : the total number of overmerging cases
         :return: ovmg_fraction : the fraction of overmerging cases
    """
    ovmg_number = det_to_truth_match[det_to_truth_match>1].size
    ovmg_fraction = ovmg_number/number_of_associations(ind_bij)
    if method == "one way":
        ovmg_fraction = ovmg_number/det_to_truth_match[det_to_truth_match>0].size
    elif method != "bij":
        print("error: wrong method name")
        ovmg_fraction = np.nan
    return ovmg_number, ovmg_fraction



#____________________________________________________
def fragmentation(truth_to_det_match, ind_bij, method="bij"):
    """ Compute the number of bijective associatons.
    
         #parameters: 
         :param truth_to_det_match : vector containing the number of associations going from the true catalog to the detection.
         :param ind_bij : array containing the indices of nearest associated objects in both catalogs
         :param method : the association method with respect to which is computed the fraction. Can be "one way" or "bij" (for bijective). Default to "bij".
         
         #results:
         :return: frag_number : the total number of fragmentation cases
         :return: frag_fraction : the fraction of fragmentation cases
    """
    frag_number = truth_to_det_match[truth_to_det_match>1].size
    frag_fraction = frag_number/number_of_associations(ind_bij)
    if method == "one way":
        frag_fraction = frag_number/truth_to_det_match[truth_to_det_match>0].size
    elif method != "bij":
        print("error: wrong method name")
        frag_fraction = np.nan
    return frag_number, frag_fraction

#____________________________________________________
def completeness(truth_data, ind_bij, gc, gc_truth):
    
    number_of_match = number_of_associations(ind_bij)
    number_of_halo = len(truth_data)
    area_ratio = np.min([gc_truth.sky_area, gc.sky_area])/gc_truth.sky_area
    
    compl = 1/ area_ratio * number_of_match/number_of_halo
    
    return compl

#____________________________________________________
def purity(cluster_data, ind_bij, gc, gc_truth):
    
    number_of_match = number_of_associations(ind_bij)
    number_of_detection = len(cluster_data)
    area_ratio = np.min([gc_truth.sky_area, gc.sky_area])/gc.sky_area
    
    pure = 1/ area_ratio * number_of_match/number_of_detection
    
    return pure

#____________________________________________________
def completeness_2d(halo_data, ind_bij, gc, gc_truth, bin_range = None, bins = None, nmin=10) :
    
    hist_num_match = np.histogram2d(halo_data['redshift'][ind_bij[:,1]],np.log10(halo_data['halo_mass'][ind_bij[:,1]]),range=bin_range, bins= bins)
    number_of_match = hist_num_match[0]
    
    hist_num_halo = np.histogram2d(halo_data['redshift'],np.log10(halo_data['halo_mass']),range=bin_range, bins= bins)
    number_of_halo = hist_num_halo[0]
    
    area_ratio = np.min([gc_truth.sky_area, gc.sky_area])/gc_truth.sky_area
    
    compl_2d = 1/ area_ratio * number_of_match/number_of_halo
    
    compl_2d_masked = np.ma.masked_where(number_of_halo<nmin, compl_2d)
    
    return compl_2d, compl_2d_masked

#____________________________________________________
def purity_2d(cluster_data, ind_bij, gc, gc_truth, bin_range = None, bins = None, nmin=10) :
    
    hist_num_match = np.histogram2d(cluster_data['redshift'][ind_bij[:,0]],np.log10(cluster_data['richness'][ind_bij[:,0]]),range=bin_range, bins= bins)
    number_of_match = hist_num_match[0]
    
    hist_num_halo = np.histogram2d(cluster_data['redshift'],np.log10(cluster_data['richness']),range=bin_range, bins= bins)
    number_of_detection = hist_num_halo[0]
    
    area_ratio = np.min([gc_truth.sky_area, gc.sky_area])/gc.sky_area
    
    pure_2d = 1/ area_ratio * number_of_match/number_of_detection
    
    pure_2d_masked = np.ma.masked_where(number_of_detection<nmin, pure_2d)
    
    return pure_2d, pure_2d_masked

#____________________________________________________
def centering_2d(data, ind_bij, ind_centered,bin_range = None, bins = None, nmin=10, ref_cat = "DC2") :
    
    if ref_cat=="DC2":
        hist_num_match = np.histogram2d(data['redshift'][ind_bij[:,1]],np.log10(data['halo_mass'][ind_bij[:,1]]),range=bin_range, bins= bins)    
        hist_num_match_well_centred = np.histogram2d(data['redshift'][ind_bij[:,1]][ind_centered],np.log10(data['halo_mass'][ind_bij[:,1]][ind_centered]),range=bin_range, bins= bins)
            
    elif ref_cat=="RM":
        hist_num_match = np.histogram2d(data['redshift'][ind_bij[:,0]],np.log10(data['richness'][ind_bij[:,0]]),range=bin_range, bins= bins)
        hist_num_match_well_centred = np.histogram2d(data['redshift'][ind_bij[:,0]][ind_centered],np.log10(data['richness'][ind_bij[:,0]][ind_centered]),range=bin_range, bins= bins)
        
    number_of_match = hist_num_match[0]
    number_of_well_centred_match = hist_num_match_well_centred[0]
    
    cent_2d = number_of_well_centred_match/number_of_match
    
    cent_2d_masked = np.ma.masked_where(number_of_match<nmin, cent_2d)
    
    return cent_2d, cent_2d_masked