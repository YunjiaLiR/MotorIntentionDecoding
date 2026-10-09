import numpy as np
import matplotlib.pyplot as plt
import mne
from scipy.stats import pearsonr


def plot_average_tfr(powers, channels, time_windows, fmin=25, fmax=35, vlim=None):
    # compute common vlim if not provided
    if vlim is None:
        all_vals = []
        for p in powers.values():
            for tmin, tmax in time_windows:
                pw = p.copy().crop(tmin=tmin, tmax=tmax, fmin=fmin, fmax=fmax)
                topo = pw.data.mean(axis=(1, 2))
                all_vals.append(topo)
        all_vals = np.concatenate(all_vals)
        v = np.nanmax(np.abs(all_vals))
        tf_vlim = (-v, v)
    else:
        tf_vlim = vlim

    conditions = list(powers.keys())
    fig, axes = plt.subplots(len(conditions), len(channels), figsize=(15, 10), sharex=True, sharey=True)

    for row_idx, cond in enumerate(conditions):
        power = powers[cond]
        power_plot = power.copy().crop(fmin=fmin, fmax=fmax)
        for col_idx, ch in enumerate(channels):
            ax = axes[row_idx, col_idx]

            power_plot.plot(
                picks=ch,
                axes=ax,
                colorbar=False,
                show=False,
                cmap="RdBu_r"
            )

            if row_idx == 0:
                ax.set_title(ch, fontsize=14)

            if col_idx == 0:
                ax.set_ylabel(f"{cond}", fontsize=12)
            else:
                ax.set_ylabel("")

            ax.axvline(0, color="g", linestyle="--")
            ax.axvline(5, color="k")
            ax.set_xlabel("")

            im = ax.images[0]
            if vlim is not None:
                im.set_clim(*vlim)
            im.set_interpolation('bilinear')

    fig.supxlabel("Time (s)", fontsize=14)
    fig.supylabel("Frequency (Hz)", fontsize=14)

    cbar = fig.colorbar(axes[0, 0].images[0], ax=axes.ravel().tolist())

    plt.show(block=False)
    plt.pause(0.1)

    return fig, axes


def plot_topomaps(powers, time_windows, fmin, fmax, tf_vlim=None):
    n_rows = len(powers)
    n_cols = len(time_windows)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2 * n_cols, 2 * n_rows))
    axes = np.atleast_2d(axes)

    for r, (cond_name, pwr) in enumerate(powers.items()):
        for c, (tmin, tmax) in enumerate(time_windows):
            ax = axes[r, c]
            pw = pwr.copy().crop(tmin=tmin, tmax=tmax, fmin=fmin, fmax=fmax)
            band_topo = pw.data.mean(axis=(1, 2))
            im, _ = mne.viz.plot_topomap(band_topo, pwr.info, axes=ax, show=False, sphere=(0., 0., 0., 0.13))
            if tf_vlim is not None:
                im.set_clim(tf_vlim)
            if r == 0:
                ax.set_title(f"{tmin}–{tmax} s", fontsize=10)

            if c == 0:
                ax.text(
                    -0.25, 0.5, cond_name,
                    transform=ax.transAxes,
                    rotation=90,
                    va="center", ha="center",
                    fontsize=12
                )

    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.85)

    plt.show(block=False)
    plt.pause(0.1)

    return fig, axes


def plot_trials_tfr(power_left_trial, channels, n_trials=10, fmin=8, fmax=40):
    n_trials = min(n_trials, len(power_left_trial))
    fig, axes = plt.subplots(n_trials, len(channels), figsize=(15, 3 * n_trials), sharex=True, sharey=True)
    axes = np.atleast_2d(axes)

    for trial_idx in range(n_trials):
        power_plot = power_left_trial.copy()
        power_plot.data = power_plot.data[trial_idx:trial_idx+1]
        power_plot.crop(fmin=fmin, fmax=fmax)

        for col_idx, ch in enumerate(channels):
            ax = axes[trial_idx, col_idx]

            power_plot.plot(
                picks=ch,
                axes=ax,
                colorbar=False,
                show=False,
                cmap="RdBu_r"
            )

            if trial_idx == 0:
                ax.set_title(ch, fontsize=14)

            if col_idx == 0:
                ax.set_ylabel(f"Trial {trial_idx+1}", fontsize=10)
            else:
                ax.set_ylabel("")

            ax.axvline(0, color="g", linestyle="--")
            ax.axvline(5, color="k")

            ax.set_xlabel("")

            im = ax.images[0]
            im.set_clim(-1.5, 1.5)

    fig.supxlabel("Time (s)", fontsize=14)
    fig.supylabel("Frequency (Hz)", fontsize=14)

    cbar = fig.colorbar(axes[0, 0].images[0], ax=axes.ravel().tolist())

    plt.show(block=False)
    plt.pause(0.1)

    return fig, axes


def compute_template_correlations(power_left_trial, ch_list):
    results = {}
    for ch in ch_list:
        ch_idx = power_left_trial.ch_names.index(ch)
        X = power_left_trial.data[:, ch_idx, :, :]

        template = X.mean(axis=0)

        corrs = []
        for i in range(X.shape[0]):
            r, _ = pearsonr(X[i].ravel(), template.ravel())
            corrs.append(r)

        corrs = np.array(corrs)

        results[ch] = {
            'mean': corrs.mean(),
            'std': corrs.std(),
            'min': corrs.min(),
            'max': corrs.max(),
            'corrs': corrs,
        }

    return results


__all__ = [
    'plot_average_tfr',
    'plot_topomaps',
    'plot_trials_tfr',
    'compute_template_correlations',
]