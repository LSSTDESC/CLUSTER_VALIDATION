import numpy as np


#functions returning different statistics about the associations

def number_of_associations(truth_data, bij_ind):

    bij_number = truth_data['redshift'][bij_ind[:,1]].size
    return bij_number
    
    
def overmerging(det_to_truth_match, truth_data, bij_ind, method="bij"):

    ovmg_number = det_to_truth_match[det_to_truth_match>1].size
    ovmg_fraction = ovmg_number/number_of_associations(truth_data, bij_ind)
    if method == "one way":
        ovmg_fraction = ovmg_number/det_to_truth_match[det_to_truth_match>0].size
    elif method != "bij":
        print("error: wrong method name")
        ovmg_fraction = np.nan
    return ovmg_number, ovmg_fraction


def fragmentation(truth_to_det_match, truth_data, bij_ind, method="bij"):
    
    frag_number = truth_to_det_match[truth_to_det_match>1].size
    frag_fraction = frag_number/number_of_associations(truth_data, bij_ind)
    if method == "one way":
        frag_fraction = frag_number/truth_to_det_match[truth_to_det_match>0].size
    elif method != "bij":
        print("error: wrong method name")
        frag_fraction = np.nan
    return frag_number, frag_fraction