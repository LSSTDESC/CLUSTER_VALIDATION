from abc import ABC, abstractstaticmethod

import numpy as np


class ModelBase(ABC):

    @abstractstaticmethod
    def Model(params):
        pass

    @abstractstaticmethod
    def Comparison(params, z_domain, lambda_values, z_true, delta_z, delta_lambda):
        pass


class LambdaModel(ModelBase):

    def Model(params):
        z, sigmaz, z_peak, lambda_peak = params
        l_mod = lambda_peak - (sigmaz*1.0e4)*(z-z_peak)**2.
        l_mod[l_mod < 0.] = 0.
        return l_mod

    def Comparison(params, z_domain, lambda_values, z_true, delta_z, delta_lambda):
        # Pull out the parameters
        z_init, sigmaz_init, lambda_init = params

        # Some priors
        if any(params < 0.0):
            return np.inf
        if z_init > 4.0:
            # Way too high redshift
            return np.inf
        if lambda_init > 1000:
            # Way too big
            return np.inf
        if np.abs(z_init-z_true) > delta_z:
            # Too far away from the true redshift
            return np.inf

        model_params = z_domain, sigmaz_init, z_init, lambda_init
        lam_model = LambdaModel.Model(model_params)
        X = (lambda_values - lam_model)**2

        # Indices of z_domain where z < z_t + dz AND z > z_t - dz
        z_cond = (z_domain < z_true+delta_z) * (z_domain > z_true-delta_z)

        # this is to be sure that it considers the max lambda in the relevant z-range
        max_lambda_in_z_cond = max(lambda_values[z_cond])

        indices = (lambda_values > max_lambda_in_z_cond*delta_lambda) * z_cond

        return sum(X[indices])
