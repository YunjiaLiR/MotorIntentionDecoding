import mne
import numpy as np
import matplotlib.pyplot as plt
from mne.time_frequency import tfr_morlet
from scipy.io import loadmat
from mne.viz import plot_topomap
import matplotlib.lines as mlines

raw1 = mne.io.read_raw_eeglab('step1_preprocess/Subject1_After ICA/Run1.set', preload=True)
raw2 = mne.io.read_raw_eeglab('step1_preprocess/Subject1_After ICA/run2.set', preload=True)
raw3 = mne.io.read_raw_eeglab('step1_preprocess/Subject1_After ICA/run3.set', preload=True)
raw4 = mne.io.read_raw_eeglab('step1_preprocess/Subject1_After ICA/run4.set', preload=True)
raw5 = mne.io.read_raw_eeglab('step1_preprocess/Subject1_After ICA/run5.set', preload=True)
raw6 = mne.io.read_raw_eeglab('step1_preprocess/Subject1_After ICA/run6.set', preload=True)
raw_all = mne.concatenate_raws([raw1, raw2, raw3, raw4, raw5, raw6])
print(len(raw_all.info['ch_names']))
print(raw_all.info['ch_names'])


events, event_id = mne.events_from_annotations(raw_all)
print(events[:20])
event_id_clean = {}
for k, v in event_id.items():
    if k != 'boundary':
        event_id_clean[k] = v
#print ("event_id_clean", event_id_clean)

event_id_cue_on = {'S 15': event_id_clean['S 15']}

epochs = mne.Epochs(raw_all,events, event_id_cue_on,-2.0,5.0,(-2.0,0.0),preload=True)

lab_mat = loadmat('Force Subject 1/cue_labels_subject1.mat')
cue_labels = lab_mat['labels'].squeeze()
print(len(epochs), len(cue_labels))

idx_left = cue_labels == 1
idx_right = cue_labels == 2
idx_blank = cue_labels == 3

epochs_left   = epochs[idx_left]
epochs_right  = epochs[idx_right]
epochs_blank  = epochs[idx_blank]

print(len(epochs_left), len(epochs_right), len(epochs_blank))

freqs = np.arange(8,41,1) #does not include the stop value
n_cyc = freqs / 4.0

power_left = tfr_morlet(epochs_left, freqs, n_cyc, return_itc=False, average = True)
power_right = tfr_morlet(epochs_right, freqs, n_cyc, return_itc=False, average = True)
power_blank = tfr_morlet(epochs_blank, freqs, n_cyc, return_itc=False, average = True)
#average
baseline = (-2.0, 0.0)

power_left_bs  = power_left.copy().apply_baseline(baseline=(-2.0, 0.0), mode="zscore")
power_right_bs = power_right.copy().apply_baseline(baseline=(-2.0, 0.0), mode="zscore")
power_blank_bs = power_blank.copy().apply_baseline(baseline=(-2.0, 0.0), mode="zscore")


powers = {
    "Left": power_left_bs,
    "Right": power_right_bs,
    "Blank": power_blank_bs,
}


conditions = ["Left", "Right", "Blank"]
channels = ["Cz", "C4", "C3"]


fig, axes = plt.subplots(3, 3,figsize=(15, 10),sharex=True,sharey=True)

for row_idx, cond in enumerate(conditions):
    power = powers[cond]
    power_plot = power.copy().crop(fmin=20)
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

        ax.axvline(0, color="k", linestyle="--")
        ax.axvline(5, color="k")

        ax.set_xlabel("")

        im = ax.images[0]
        im.set_clim(-2, 2)
        im.set_interpolation('bilinear')

fig.supxlabel("Time (s)", fontsize=14)
fig.supylabel("Frequency (Hz)", fontsize=14)

cbar = fig.colorbar(axes[0, 0].images[0],ax=axes.ravel().tolist(),label="Power (z-score)")

plt.show()


#
freq_band = (25, 35)
time_windows = [(-2.0,-1.0),(-1.0,0.0), (0.0,1.0),(1.0,2.0),(2.0,3.0),(3.0,4.0),(4.0,5.0)]

fig, axes = plt.subplots(len(conditions),len(time_windows),figsize=(12, 6),constrained_layout=True)

for row, cond in enumerate(conditions):
    p = powers[cond]
    fmask = (p.freqs >= freq_band[0]) & (p.freqs <= freq_band[1])
    for col, (tmin, tmax) in enumerate(time_windows):
        ax = axes[row, col]
        tmask = (p.times >= tmin) & (p.times <= tmax)
        data = p.data[:, fmask][:, :, tmask].mean(axis=(1, 2))
        im, _ = plot_topomap(
            data,
            p.info,
            axes=ax,
            cmap="RdBu_r",
            contours=10, # add contour lines
            outlines='head',
            show=False,
        )
        im.set_clim(-2, 2)

        if col == 0:
            ax.set_ylabel(cond, fontsize=12)
        else:
            ax.set_ylabel('')

"""
pos_left  = axes[0, 1].get_position()
pos_right = axes[0, 2].get_position()
x0 = (pos_left.x1 + pos_right.x0) / 2  # mid between the two columns

pos_top = axes[0, 0].get_position()
pos_bot = axes[-1, 0].get_position()

line = mlines.Line2D(
    [x0, x0],
    [pos_bot.y0, pos_top.y1],
    transform=fig.transFigure,
    color='k',
    linestyle='--',
    linewidth=1.5
)
fig.add_artist(line)
"""

#State
prep_left  = axes[0, 0].get_position()
prep_right = axes[0, 1].get_position()
prep_x = (prep_left.x0 + prep_right.x1) / 2

img_left  = axes[0, 2].get_position()
img_right = axes[0, 6].get_position()
img_x = (img_left.x0 + img_right.x1) / 2
title_y = prep_left.y1 + 0.08  # a bit above top row

"""
fig.text(prep_x, title_y, "Preparation (Pr)", ha='center', va='bottom', fontsize=13, fontweight='bold')
fig.text(img_x, title_y, "Imagery (Im)", ha='center', va='bottom', fontsize=13, fontweight='bold')
"""
#Bottom Time Arrow
# ---- Bottom Time Axis with Arrow ----
bottom_left  = axes[-1, 0].get_position()
bottom_right = axes[-1, -1].get_position()

x0 = bottom_left.x0
y0 = bottom_left.y0 - 0.1          # slightly below the bottom row
width = bottom_right.x1 - bottom_left.x0
height = 0.06                       # small strip for the time axis

ax_time = fig.add_axes([x0, y0, width, height])
t_min = time_windows[0][0]
t_max = time_windows[-1][1]
ax_time.set_xlim(time_windows[0][0], time_windows[-1][1])  # -2 to 5
ax_time.set_ylim(0, 1)
ax_time.axis('off')


ax_time.annotate(
    '',
    xy=(t_max, 0.5),
    xytext=(t_min, 0.5),
    arrowprops=dict(arrowstyle='->', linewidth=1.5, color='k'),
)

ax_time.axvline(0, ymin=0.25, ymax=0.95, color='k', linestyle='--')

centers = [(t0 + t1) / 2 for (t0, t1) in time_windows]
labels  = [f"[{int(t0)},{int(t1)}]" for (t0, t1) in time_windows]
for c, lab in zip(centers, labels):
    ax_time.text(c, 0.9, lab, ha='center', va='bottom', fontsize=8)

"""
# main label in the middle
mid = (t_min + t_max) / 2
ax_time.text(mid, 1, "Time wrt Go (s)", ha='center', va='bottom', fontsize=12)
"""
# shared colorbar
cbar = fig.colorbar(im, ax=axes.ravel().tolist(), label="Power (z-score)")

plt.show()