import numpy as np
import pandas as pd
from matplotlib.ticker import MultipleLocator, StrMethodFormatter


class PlotHelper(object):
    """
    Helper class that houses most of the boiler-plate code to make matplotlib plots look good.
    """

    @staticmethod
    def set_defaults(mpl):
        """
        Set up the default styles used in all matplotlib plots for this notebook
        """
        # Not working at NERSC
        # matplotlib.rc('text', usetex=True)
        mpl.rc("font", family="serif")
        mpl.rc("figure", figsize=(9.6, 6.8))
        mpl.rc("xtick", labelsize=20)
        mpl.rc("ytick", labelsize=20)
        mpl.rc("legend", fontsize="x-large")
        mpl.rc("xtick", direction="in")
        mpl.rc("xtick", labelsize=24)
        mpl.rc("xtick.major", size=7)
        mpl.rc("xtick.minor", size=4)
        mpl.rc("xtick.major", width=1.25)
        mpl.rc("xtick.minor", width=1.25)
        mpl.rc("ytick", direction="in")
        mpl.rc("ytick", labelsize=24)
        mpl.rc("ytick.major", size=7)
        mpl.rc("ytick.minor", size=4)
        mpl.rc("ytick.major", width=1.25)
        mpl.rc("ytick.minor", width=1.25)
        mpl.rc("patch", edgecolor="k")
        mpl.rc("legend", markerscale=1.5)
        mpl.rc("errorbar", capsize=3)
        mpl.rc("axes", labelsize="x-large")
        mpl.rc("axes", linewidth=1.25)

    @staticmethod
    def plot_model_vs_data(
        plt,
        data_model,
        z_peak,
        sigmaz,
        lambda_peak,
        z_in,
        lambda_in,
        z_domain,
        lambda_array,
        delta_z,
        delta_lambda,
    ):
        """
        Provided a data model, best fit values, and true values, this function will plot the model compared to the true values.
        """

        plt.plot(z_domain, lambda_array, label="z-scan")
        plt.scatter(
            z_in, lambda_in, marker="o", s=200, color="r", label="true value"
        )
        plt.plot(
            z_domain,
            data_model.Model((z_domain, sigmaz, z_peak, lambda_peak)),
            label="model fit",
        )
        plt.ylabel("$\\lambda(z)$", fontsize=24)
        plt.xlabel("$z$", fontsize=24)

        z_cond = (z_domain < z_in + delta_z) * (z_domain > z_in - delta_z)
        # this is to be sure that it consider the max in the relevant z-range
        max_lambda_in_z_cond = max(lambda_array[z_cond])
        indices = (lambda_array > max_lambda_in_z_cond * delta_lambda) * z_cond

        plt.plot(
            z_domain[indices],
            lambda_array[indices],
            "^",
            label="data used in the fit",
        )
        plt.legend()

    @staticmethod
    def plot_sample(
        plt,
        cond,
        z_in,
        z_domain,
        lambda_arrays,
        lambda_in,
        data_model,
        sz_all,
        zp_all,
        lp_all,
        chis,
    ):
        idx = np.arange(len(z_in))
        idx_to_keep = idx[cond]

        print(len(idx_to_keep))

        f, ax = plt.subplots(
            min(10, len(idx_to_keep)), 1, sharex=True, figsize=(5, 20)
        )

        for i, idx in enumerate(idx_to_keep[: min(10, len(idx_to_keep))]):
            z_cond = (z_domain < z_in[idx] + 0.1) * (
                z_domain > z_in[idx] - 0.1
            )
            # this is to be sure that it consider the max in the relevant z-range
            max_lambda_in_z_cond = max((lambda_arrays[idx])[z_cond])
            indices = (
                lambda_arrays[idx] > max_lambda_in_z_cond * 0.2
            ) * z_cond
            ax[i].plot(z_domain[indices], (lambda_arrays[idx])[indices], "go")
            ax[i].plot(z_domain, lambda_arrays[idx])
            # quello che matcha con redMaPPer usato per il fit
            ax[i].plot(z_in[idx], lambda_in[idx], "ro")
            ax[i].plot(
                z_domain,
                data_model.Model(
                    (z_domain, sz_all[idx], zp_all[idx], lp_all[idx])
                ),
                label=(idx, chis[idx]),
            )
            ax[i].legend()

    @staticmethod
    def plot_sigmaz_scatter(plt, zpeaks, sig_z_kernel, chis):
        plt.scatter(zpeaks, sig_z_kernel, c=np.log(chis))
        plt.yscale("log")
        plt.xlim(0.10, 1.150)
        plt.ylim(0.01, 1.0)
        plt.xlabel("$z$", fontsize=24)
        plt.ylabel("$\sigma_z$", fontsize=24)
        plt.title(
            "Calculated $\sigma_z$ for redmapper run on CosmoDC2",
            fontsize=24,
            y=1.1,
        )
        cbar = plt.colorbar()
        cbar.ax.get_yaxis().labelpad = 20
        cbar.ax.set_ylabel("$\ln{\chi}$", rotation=0, fontsize=22)

    @staticmethod
    def plot_avg_combined_plot(fig, ax, plt, z_bins, avg, sem, comparison):
        """
        Helper method to plot the average sigma_z as a function of z_bin, will also plot other existing sigma_z curves next to it.
        """
        digitized_plots = {
            "des_y1": {
                "Label": "DES Y1",
                "Path": "digitized_plots/des_y1.csv",
                "Color": "k",
            },
            "buzza": {
                "Label": "Buzzard A",
                "Path": "digitized_plots/buzza.csv",
                "Color": "darkorange",
            },
            "buzzb": {
                "Label": "Buzzard B",
                "Path": "digitized_plots/buzzb.csv",
                "Color": "dodgerblue",
            },
            "buzzc": {
                "Label": "Buzzard C",
                "Path": "digitized_plots/buzzc.csv",
                "Color": "mediumseagreen",
            },
        }

        ax.set_xlim(0.20, 0.6)
        ax.set_ylim(0.02, 0.25)

        ax.set_xlabel("$z$", fontsize=24)
        ax.set_ylabel(r"$\langle \sigma_z \rangle$", fontsize=24)

        ax.xaxis.set_major_locator(MultipleLocator(0.05))
        ax.xaxis.set_major_formatter(StrMethodFormatter("{x:.2f}"))
        ax.xaxis.set_minor_locator(MultipleLocator(0.01))
        ax.yaxis.set_minor_locator(MultipleLocator(0.01))

        ax.tick_params(
            top=True, right=True, which="both", width=1, labelsize=16
        )
        ax.errorbar(
            z_bins,
            avg,
            yerr=sem,
            marker="8",
            linestyle="solid",
            capsize=2,
            elinewidth=0.5,
            linewidth=2,
            label="DC2",
            color="g",
        )

        avg_cosmodc2 = comparison["sz"]
        yerr_cosmodc2 = comparison["err"]
        z_bins_cosmodc2 = comparison["bins"]
        ax.errorbar(
            z_bins_cosmodc2,
            avg_cosmodc2,
            yerr=yerr_cosmodc2,
            marker="8",
            linestyle="solid",
            capsize=2,
            elinewidth=0.5,
            linewidth=2,
            label="CosmoDC2",
            color="r",
        )

        for (key, item) in digitized_plots.items():
            df = pd.read_csv(item["Path"], delimiter=",")

            if "X" in df:
                x_points = df["X"]
            if "x" in df:
                x_points = df["x"]

            if "Y" in df:
                y_points = df["Y"]
            if "y" in df:
                y_points = df["y"]

            del df

            ax.plot(
                x_points,
                y_points,
                linewidth=2,
                color=item["Color"],
                label=item["Label"],
            )

        ax.legend()
