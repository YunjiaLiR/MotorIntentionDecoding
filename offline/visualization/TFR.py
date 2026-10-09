import numpy as np
import mne
import matplotlib.pyplot as plt
from scipy.io import loadmat


#Loading data
raws = []
for i in range(1,7):
    raw = mne.io.read_raw_eeglab(
        f'Preprocess_data/eeg/step1_preprocess/Subject 1/run{i}_pruned with ICA.set',
        preload=True
    )
    raws.append(raw)
raw_all = mne.concatenate_raws(raws)


#Extract events
events, event_id = mne.events_from_annotations(raw_all)

annotations = raw_all.annotations.copy()
rename_map = {}
for d in set(annotations.description):
    if str(d).lower() in ("boundary","edge boundary"):
        rename_map[d] = "BAD_boundary"
raw_all.set_annotations(annotations.rename(rename_map))


#Epochs
event_id_use = {'S 15': event_id['S 15']}
epochs = mne.Epochs(
    raw_all,
    events,
    event_id=event_id_use,
    tmin=-2.5,
    tmax=5.0,
    baseline=None,
    preload=True,
    reject_by_annotation=True
)

force_mat = loadmat('Preprocess_data/force/cue_labels_subject1.mat')
cue_labels = force_mat['labels'].squeeze()
print("Epochs length:", len(epochs), "Cue labels length:", len(cue_labels))

idx_left = cue_labels == 1
idx_right = cue_labels == 2
idx_blank = cue_labels == 3

epochs_left = epochs[idx_left]
epochs_right = epochs[idx_right]
epochs_blank = epochs[idx_blank]


# ============================================================================
# OLD PART: this is class-averaged, not suitable for trial-by-trial inspection
# ============================================================================

# freqs = np.arange(8,41,1)
# n_cycles = freqs/4.0
#
# power_left = mne.time_frequency.tfr_morlet(
#     epochs_left, freqs=freqs, n_cycles=n_cycles,
#     use_fft=True, return_itc=False, average=True, picks="eeg", decim=1
# )
# power_right = mne.time_frequency.tfr_morlet(
#     epochs_right, freqs=freqs, n_cycles=n_cycles,
#     use_fft=True, return_itc=False, average=True, picks="eeg", decim=1
# )
# power_blank = mne.time_frequency.tfr_morlet(
#     epochs_blank, freqs=freqs, n_cycles=n_cycles,
#     use_fft=True, return_itc=False, average=True, picks="eeg", decim=1
# )
#
# power_left.apply_baseline(baseline=(-2.0,-0.1), mode="zscore")
# power_right.apply_baseline(baseline=(-2.0,-0.1), mode="zscore")
# power_blank.apply_baseline(baseline=(-2.0,-0.1), mode="zscore")
#
# powers = {
#     "Left": power_left,
#     "Right": power_right,
#     "Blank": power_blank,
# }


# ============================================================================
# NEW PART: one class, all trials, NO averaging
# ============================================================================

# -------- choose one class --------
epochs_class = epochs_left
class_name = "Left"

# epochs_class = epochs_right
# class_name = "Right"

# epochs_class = epochs_blank
# class_name = "Blank"

print(f"Selected class: {class_name}, n_trials = {len(epochs_class)}")


# -------- compute trial-wise TFR --------
freqs = np.arange(8,41,1)
n_cycles = freqs / 4.0

power_class = mne.time_frequency.tfr_morlet(
    epochs_class,
    freqs=freqs,
    n_cycles=n_cycles,
    use_fft=True,
    return_itc=False,
    average=False,   # keep all trials
    picks="eeg",
    decim=1
)

print("power_class.data shape before baseline:", power_class.data.shape)
# shape: (n_trials, n_channels, n_freqs, n_times)

power_class.apply_baseline(baseline=(-2.0,-0.1), mode="zscore")

print("power_class.data shape after baseline:", power_class.data.shape)
print("Min / Max after baseline:", power_class.data.min(), power_class.data.max())


# ============================================================================
# TFR plot for ALL trials of selected class
# using manual plotting, no averaging
# ============================================================================

channels = ["T8","FT8","TP8","FT10"]
# for MI you may also want:
# channels = ["C3", "Cz", "C4", "FCz"]

ch_indices = [power_class.ch_names.index(ch) for ch in channels]
times = power_class.times
freqs = power_class.freqs

# crop range equivalent to your old crop(fmin=8, fmax=40)
fmask_plot = (freqs >= 8) & (freqs <= 40)
freqs_plot = freqs[fmask_plot]

# global color scale across all trials/channels for fair comparison
all_tfr_vals = power_class.data[:, ch_indices][:, :, fmask_plot, :]
v = np.nanmax(np.abs(all_tfr_vals))
tf_vlim = (-v, v)

n_trials = power_class.data.shape[0]
fig, axes = plt.subplots(
    n_trials,
    len(channels),
    figsize=(4 * len(channels), 2.2 * n_trials),
    sharex=True,
    sharey=True
)

axes = np.atleast_2d(axes)

for row_idx in range(n_trials):
    for col_idx, ch in enumerate(channels):
        ax = axes[row_idx, col_idx]
        ch_idx = power_class.ch_names.index(ch)

        # data shape: (n_freqs, n_times)
        data = power_class.data[row_idx, ch_idx, fmask_plot, :]

        im = ax.imshow(
            data,
            aspect="auto",
            origin="lower",
            extent=[times[0], times[-1], freqs_plot[0], freqs_plot[-1]],
            cmap="RdBu_r",
            vmin=tf_vlim[0],
            vmax=tf_vlim[1],
            interpolation="bilinear"
        )

        if row_idx == 0:
            ax.set_title(ch, fontsize=14)

        if col_idx == 0:
            ax.set_ylabel(f"Trial {row_idx}", fontsize=10)
        else:
            ax.set_ylabel("")

        ax.axvline(0, color="g", linestyle="--")
        ax.axvline(5, color="k")
        ax.set_xlabel("")

fig.supxlabel("Time (s)", fontsize=14)
fig.supylabel("Frequency (Hz)", fontsize=14)
fig.suptitle(f"TFR of all trials - {class_name} class", fontsize=16)

cbar = fig.colorbar(im, ax=axes.ravel().tolist())
plt.tight_layout()
plt.show()


# ============================================================================
# Topomap for ALL trials of selected class
# using manual single-trial data, no averaging
# ============================================================================

time_windows = [(-2,-1),(-1,0),(0,1),(1,2),(2,3),(3,4),(4,5)]
fmin, fmax = 25,35

fmask_topo = (power_class.freqs >= fmin) & (power_class.freqs <= fmax)

# global color scale for topomap across all trials/windows
all_vals = []
for trial_idx in range(n_trials):
    for tmin, tmax in time_windows:
        tmask = (power_class.times >= tmin) & (power_class.times <= tmax)
        topo = power_class.data[trial_idx, :, :, :][:, fmask_topo][:, :, tmask].mean(axis=(1,2))
        all_vals.append(topo)

all_vals = np.concatenate(all_vals)
v_topo = np.nanmax(np.abs(all_vals))
topo_vlim = (-v_topo, v_topo)

# use EEG channel info only
info_eeg = mne.pick_info(power_class.info, mne.pick_types(power_class.info, eeg=True))

fig, axes = plt.subplots(
    n_trials,
    len(time_windows),
    figsize=(2 * len(time_windows), 2.2 * n_trials)
)
axes = np.atleast_2d(axes)

for r in range(n_trials):
    for c, (tmin, tmax) in enumerate(time_windows):
        ax = axes[r, c]

        tmask = (power_class.times >= tmin) & (power_class.times <= tmax)

        # average only across freq and time within THIS TRIAL for topomap scalar map
        band_topo = power_class.data[r, :, :, :][:, fmask_topo][:, :, tmask].mean(axis=(1,2))

        im, _ = mne.viz.plot_topomap(
            band_topo,
            info_eeg,
            axes=ax,
            show=False,
            sphere=(0., 0., 0., 0.13),
            cmap="RdBu_r"
        )
        im.set_clim(topo_vlim)

        if r == 0:
            ax.set_title(f"{tmin}–{tmax} s", fontsize=10)

        if c == 0:
            ax.text(
                -0.25, 0.5, f"Trial {r}",
                transform=ax.transAxes,
                rotation=90,
                va="center", ha="center",
                fontsize=10
            )

fig.suptitle(f"Topomap of all trials - {class_name} class ({fmin}-{fmax} Hz)", fontsize=16)
cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.85)
plt.tight_layout()
plt.show()