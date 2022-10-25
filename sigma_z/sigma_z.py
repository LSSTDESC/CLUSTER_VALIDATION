from abc import ABC, abstractmethod
from collections import namedtuple
from multiprocessing import Pool

import numpy as np
import yaml
from scipy import stats
from scipy.optimize import minimize

from sigma_z.data_loader import DataLoader, GCRDataLoader
from sigma_z.lambda_model import LambdaModel


class SigmaZ(ABC):
    """
    Base class which holds all of the shared logic to calculate sigma_z for a galaxy catalog.
    """

    def __init__(
        self, config_file_path, data_model, z_start, z_end, z_step
    ) -> None:
        self.config = self.load_config(config_file_path)
        self.z_domain = np.arange(z_start, z_end, z_step)
        self.data_model = data_model

    def load_config(self, config_file_path) -> None:
        with open(config_file_path, "r") as stream:
            try:
                return yaml.safe_load(stream)
            except yaml.YAMLError as ex:
                print(ex)

    def fit_model_all(
        self, z_in, lambda_in, lambda_arrays, delta_z, delta_lambda
    ):
        """
        The main method that fits the provided model to all of the galaxy clusters in the input arrays.  Will return best fits for all galaxy clusters matched by index.
        """
        N = len(lambda_arrays)

        # Increase number of processes to speed this up, but be careful about resources
        with Pool() as pool:
            cluster_fits = [
                pool.apply_async(
                    self.fit_model_by_id,
                    (
                        k,
                        z_in[k],
                        lambda_in[k],
                        lambda_arrays[k],
                        delta_z,
                        delta_lambda,
                    ),
                )
                for k in range(0, N)
            ]
            results = [fit.get() for fit in cluster_fits]

        # sigmaz, z_peak, lambda_peak for all clusters
        # chis is to estimate the goodness of the fit (it is not exactly the chisq value)
        sz_all, zp_all, lp_all, chis = (
            np.zeros((N)),
            np.zeros((N)),
            np.zeros((N)),
            np.zeros((N)),
        )

        for result in results:
            idx = result["idx"]
            zp_all[idx] = result["z_best"]
            sz_all[idx] = result["sigmaz_best"]
            lp_all[idx] = result["lambda_best"]
            chis[idx] = result["chis"]

        # Define sig_z_kernel according to Eq. 9 of the Projection effect paper (https://arxiv.org/pdf/1807.07072.pdf)
        # With this transformation sig_z_kernel represents the width of the parabola used to fit lambda(z)
        sig_z_kernel = (sz_all / lp_all) ** (-0.5) / 100.0

        return {
            "sigmaz_all": sz_all,
            "zpeak_all": zp_all,
            "lambdapeak_all": lp_all,
            "sig_z_kernel": sig_z_kernel,
            "chis_all": chis,
        }

    def fit_model_by_id(
        self,
        idx,
        z_in,
        lambda_in,
        lambda_arrays,
        delta_z,
        delta_lambda,
        method="Nelder-Mead",
    ):
        """
        Provided an index, will fit the galaxy cluster at that index according to the provided model.  Will return the best fit.
        """
        x0 = [z_in, 1.0, lambda_in]

        result = minimize(
            self.data_model.Comparison,
            x0=x0,
            args=(
                self.z_domain,
                lambda_arrays,
                z_in,
                delta_z,
                delta_lambda,
            ),
            method=method,
        )
        zbest, szbest, lambest = result["x"]

        # for the goodness of fit just use the data point actaully used to fit
        z_cond = (self.z_domain < z_in + delta_z) * (
            self.z_domain > z_in - delta_z
        )
        # this is to be sure that it consider the max in the relevant z-range
        max_lambda_in_z_cond = max(lambda_arrays[z_cond])
        indices = (
            lambda_arrays > max_lambda_in_z_cond * delta_lambda
        ) * z_cond
        my_lambda_arrays = lambda_arrays[indices]

        if len(my_lambda_arrays) == 0:
            return {
                "idx": idx,
                "z_best": -1,
                "sigmaz_best": -1,
                "lambda_best": -1,
                "chis": 100000,
            }

        model_params = self.z_domain[indices], szbest, zbest, lambest
        tosum = (
            np.abs(self.data_model.Model(model_params) - my_lambda_arrays)
            / my_lambda_arrays
        )

        chis = np.sum(tosum) / len(tosum)

        if idx % 500 == 0:
            print("At cluster %d" % idx)

        return {
            "idx": idx,
            "z_best": zbest,
            "sigmaz_best": szbest,
            "lambda_best": lambest,
            "chis": chis,
        }

    def average_by_bin(self, z_peak, sigma_z, z_start, z_end, bin_size):

        z_bins = np.arange(z_start, z_end, bin_size)
        avg_sig_z = []
        sig_z_sem = []

        for i in range(0, len(z_bins)):
            # Condition for bins +- 1/2 bin size
            this_bin = z_bins[i]
            upper_lim = z_bins[i] + bin_size / 2
            lower_lim = z_bins[i] - bin_size / 2

            # Filter out all the indices where z_peak satisfies this condition
            indices = np.where(
                (np.array(z_peak) > lower_lim)
                & (np.array(z_peak) <= upper_lim)
            )[0]

            # Filter down our sigma_z
            this_bin = sigma_z[indices]

            avg_sig_z.append(np.mean(this_bin))
            sig_z_sem.append(stats.sem(this_bin))

        return (avg_sig_z, sig_z_sem, z_bins)

    @abstractmethod
    def get_data(self):
        """
        This is the method that is responsible for loading the data required by your galaxy catalog.  This may be different according to different galaxy catalog (e.g. this requires GCRCatalogs AND fits files for cosmoDC2)

        This method should return return a dictionary of the form {
            'id_cluster': ,
            'lambda_arrays': ,
            'z_in': ,
            'lambda_in':
        }
        """
        pass


class DESY3SigmaZ(SigmaZ):
    def __init__(self, config_file_path, data_model, z_start, z_end, z_steps):
        super().__init__(config_file_path, data_model, z_start, z_end, z_steps)

    def get_data(self):
        zscan_config = self.config["zscan_catalog"]
        truth_config = self.config["truth_catalog"]

        loader = DataLoader.from_gcr(zscan_config["filenm"])
        values_nm = [
            zscan_config["cluster_id"],
            zscan_config["richness_steps"],
            truth_config["redshift"],
            truth_config["richness"],
        ]
        values = loader.get_values(values_nm)

        return {
            "id_cluster": values[zscan_config["cluster_id"]],
            "lambda_arrays": values[zscan_config["richness_steps"]],
            "z_in": values[truth_config["redshift"]],
            "lambda_in": values[truth_config["richness"]],
        }


class DC2SigmaZ(SigmaZ):
    def __init__(self, config_file_path, data_model, z_start, z_end, z_steps):

        super().__init__(config_file_path, data_model, z_start, z_end, z_steps)

    def get_data(self):

        truth_config = self.config["truth_catalog"]

        truth_catalog = truth_config["filenm"]
        truth_values_nm = [
            truth_config["cluster_id"],
            truth_config["redshift"],
            truth_config["richness"],
        ]

        # Pull our true values from GCRCatalogs
        loader = DataLoader.from_gcr(truth_catalog)
        true_data = loader.get_values(truth_values_nm)

        zscan_config = self.config["zscan_catalog"]
        zscan_catalog = zscan_config["filenm"]
        zscan_values_nm = [
            zscan_config["cluster_id"],
            zscan_config["richness_steps"],
        ]
        # Pull our zscan values from FITS
        zscan_loader = DataLoader.from_fits(zscan_catalog)
        zscan_data = zscan_loader.get_values(zscan_values_nm)

        # Get cluster IDs
        cluster_ids = true_data[truth_config["cluster_id"]]
        lambda_in = []
        z_in = []
        id_cluster = []
        lambda_arrays = []

        for i in range(len(cluster_ids)):
            this_id = cluster_ids[i]
            lambda_in.append(true_data[truth_config["richness"]][i])
            z_in.append(true_data[truth_config["redshift"]][i])

            zscan_idx = np.where(
                zscan_data[zscan_config["cluster_id"]] == this_id
            )[0][0]
            lambda_arrays.append(
                zscan_data[zscan_config["richness_steps"]][zscan_idx]
            )
            id_cluster.append(
                zscan_data[zscan_config["cluster_id"]][zscan_idx]
            )

        return {
            "id_cluster": id_cluster,
            "lambda_arrays": lambda_arrays,
            "z_in": z_in,
            "lambda_in": lambda_in,
        }


class CosmoDC2SigmaZ(SigmaZ):
    def __init__(self, config_file_path, data_model, z_start, z_end, z_steps):

        super().__init__(config_file_path, data_model, z_start, z_end, z_steps)

    def get_data(self):

        truth_config = self.config["truth_catalog"]

        truth_catalog = truth_config["filenm"]
        truth_values_nm = [
            truth_config["cluster_id"],
            truth_config["redshift"],
            truth_config["richness"],
        ]

        # Pull our true values from GCRCatalogs
        loader = DataLoader.from_gcr(truth_catalog)
        true_data = loader.get_values(truth_values_nm)

        zscan_config = self.config["zscan_catalog"]
        zscan_catalog = zscan_config["filenm"]
        zscan_values_nm = [
            zscan_config["cluster_id"],
            zscan_config["richness_steps"],
        ]
        # Pull our zscan values from FITS
        zscan_loader = DataLoader.from_fits(zscan_catalog)
        zscan_data = zscan_loader.get_values(zscan_values_nm)

        # Get cluster IDs
        cluster_ids = zscan_data[zscan_config["cluster_id"]]
        lambda_in = []
        z_in = []

        # Find the assoc. true values based on cluster ID.
        for i in range(len(cluster_ids)):
            this_id = cluster_ids[i]
            true_idx = np.where(
                true_data[truth_config["cluster_id"]] == this_id
            )[0][0]
            lambda_in.append(true_data[truth_config["richness"]][true_idx])
            z_in.append(true_data[truth_config["redshift"]][true_idx])

        return {
            "id_cluster": zscan_data[zscan_config["cluster_id"]],
            "lambda_arrays": zscan_data[zscan_config["richness_steps"]],
            "z_in": z_in,
            "lambda_in": lambda_in,
        }
