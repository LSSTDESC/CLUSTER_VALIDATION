import numpy as np


#functions returning different statistics about the associations

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
    
    
def overmerging(det_to_truth_match, truth_data, bij_ind, method="bij"):
    """ Compute the number of bijective associatons.
    
         #parameters: 
         :param det_to_truth_match : vector containing the number of associations going from the detection to the true catalog.
         :param truth_data : the truth catalog
         :param bij_ind : array containing the indices of nearest associated objects in both catalogs
         :param method : the association method with respect to which is computed the fraction. Can be "one way" or "bij" (for bijective). Default to "bij".
         
         #results:
         :return: ovmg_number : the total number of overmerging cases
         :return: ovmg_fraction : the fraction of overmerging cases
    """
    ovmg_number = det_to_truth_match[det_to_truth_match>1].size
    ovmg_fraction = ovmg_number/number_of_associations(truth_data, bij_ind)
    if method == "one way":
        ovmg_fraction = ovmg_number/det_to_truth_match[det_to_truth_match>0].size
    elif method != "bij":
        print("error: wrong method name")
        ovmg_fraction = np.nan
    return ovmg_number, ovmg_fraction


def fragmentation(truth_to_det_match, truth_data, bij_ind, method="bij"):
    """ Compute the number of bijective associatons.
    
         #parameters: 
         :param truth_to_det_match : vector containing the number of associations going from the true catalog to the detection.
         :param truth_data : the truth catalog
         :param bij_ind : array containing the indices of nearest associated objects in both catalogs
         :param method : the association method with respect to which is computed the fraction. Can be "one way" or "bij" (for bijective). Default to "bij".
         
         #results:
         :return: frag_number : the total number of fragmentation cases
         :return: frag_fraction : the fraction of fragmentation cases
    """
    frag_number = truth_to_det_match[truth_to_det_match>1].size
    frag_fraction = frag_number/number_of_associations(truth_data, bij_ind)
    if method == "one way":
        frag_fraction = frag_number/truth_to_det_match[truth_to_det_match>0].size
    elif method != "bij":
        print("error: wrong method name")
        frag_fraction = np.nan
    return frag_number, frag_fraction