# %%
import numpy as np
import mne
import matplotlib.pyplot as plt
from scipy.io import loadmat

# %%
#Loading data
raws = []
for i in range(1,7):
    raw = mne.io.read_raw_eeglab(f'eeg/step1_preprocess/Subject 7/run{i}_pruned with ICA.set',preload=True)
    # print(raw)
    # print('n_channels: ', raw.info['nchan']) #61
    # print('sfreq: ', raw.info['sfreq']) #1000.0
    # print('durations(s): ', raw.times[-1])
    montage = raw.get_montage()
    # print("Has montage: ", montage is not None)
    # print("Has dig point", raw.info['dig'])
    # print (raw.get_montage())
    # raw.plot_sensors(show_names = True)
    # plt.show(block=True)
    raws.append(raw)
raw_all = mne.concatenate_raws(raws)


#Extract events
events, event_id = mne.events_from_annotations(raw_all)
# print("event_id keys: ", list(event_id.keys())[:30])
# print("events shape", events.shape)
# print("first events\n", events[:10])
annotations = raw_all.annotations.copy()
rename_map = {}
for d in set(annotations.description):
    if str(d).lower() in ("boundary","edge boundary"):
        rename_map[d] = "BAD_boundary"
raw_all.set_annotations(annotations.rename(rename_map))


#Epochs
event_id_use = {'S 15': event_id['S 15']}
epochs = mne.Epochs(raw_all,events, event_id=event_id_use, tmin=-2.5,tmax=5.0,baseline=None,preload=True,reject_by_annotation=True)
force_mat = loadmat('force/cue_labels_subject7.mat')
cue_labels = force_mat['labels'].squeeze()
print("Epochs length:",len(epochs), "Cue labels length: ",len(cue_labels))

idx_left = cue_labels == 1
idx_right = cue_labels == 2
idx_blank = cue_labels == 3
epochs_left = epochs[idx_left]
epochs_right = epochs[idx_right]
epochs_blank = epochs[idx_blank]

# %%
def psd_per_trial(epochs, tmin, tmax, fmin=1, fmax=40, picks="eeg",n_fft=1024, n_per_seg=512, n_overlap=256):
    ep = epochs.copy().crop(tmin=tmin, tmax=tmax)
    # Welch's PSD estimator
    psd_obj = ep.compute_psd(method="welch",fmin=fmin, fmax=fmax,picks=picks,n_fft=n_fft,n_per_seg=n_per_seg,n_overlap=n_overlap,verbose=False)
    psds, freqs = psd_obj.get_data(return_freqs=True)  # (n_trials, n_ch, n_freqs)
    return psds, freqs



def normalized_metric_per_trial(epochs,picks,task_win=(1, 4),base_win=(-2, 0),band=(25, 35),total_band=(1, 40),mode="task_over_baseline",  welch_params=dict(n_fft=1024, n_per_seg=512, n_overlap=256)):

    psd_task, freqs = psd_per_trial(epochs, task_win[0], task_win[1],fmin=total_band[0], fmax=total_band[1],picks=list(picks),**welch_params)
    psd_base, freqs2 = psd_per_trial(epochs, base_win[0], base_win[1],fmin=total_band[0], fmax=total_band[1],picks=list(picks),**welch_params)

    fmin_b, fmax_b = band
    band_mask  = (freqs >= fmin_b) & (freqs <= fmax_b)
    total_mask = (freqs >= total_band[0]) & (freqs <= total_band[1])

    # Trapezoid accounts for actual frequency spacing
    bp_task = np.trapezoid(psd_task[..., band_mask], freqs[band_mask], axis=-1)
    bp_base = np.trapezoid(psd_base[..., band_mask], freqs[band_mask], axis=-1)


    tp_task = np.trapezoid(psd_task[..., total_mask], freqs[total_mask], axis=-1)
    # metric_ch = bp_task / (tp_task + 1e-12)   # 1e-12 prevent division by zero
    metric_ch = (bp_task - bp_base) / (bp_base + 1e-12)

    return metric_ch

# %%
motor_chs = ["FC3","FC1","FC2","FC4","C3","C1","Cz","C2","C4","CP3","CP1","CPz","CP2","CP4"]
roi_chs = ['Fp1', 'Fz', 'F3', 'F7', 'FT9', 'FC5', 'FC1', 'C3', 'T7', 'TP9', 'CP5', 'CP1', 'Pz', 'TP10', 'CP6', 'CP2', 'Cz', 'C4', 'T8', 'FT10', 'FC6', 'FC2', 'F4', 'F8', 'Fp2', 'AF7', 'AF3', 'AFz', 'F1', 'F5', 'FT7', 'FC3', 'C1', 'C5', 'TP7', 'CP3', 'P1', 'P5', 'P6', 'P2', 'CPz', 'CP4', 'TP8', 'C6', 'C2', 'FC4', 'FT8', 'F6', 'AF8', 'AF4', 'F2']

g_left  = normalized_metric_per_trial(epochs_left,  picks=motor_chs, band=(20,35), mode="task_ratio_total")
g_right = normalized_metric_per_trial(epochs_right, picks=motor_chs, band=(20,35), mode="task_ratio_total")
g_blank = normalized_metric_per_trial(epochs_blank, picks=motor_chs, band=(20,35), mode="task_ratio_total")


n_channels = len(motor_chs)

n_cols = 4
n_rows = int(np.ceil(n_channels / n_cols))

fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 3*n_rows), sharex=False)

axes = axes.flatten()

for ch_idx, ch_name in enumerate(motor_chs):

    ax = axes[ch_idx]

    ax.plot(g_left[:, ch_idx],  label="Left")
    ax.plot(g_right[:, ch_idx], label="Right")
    ax.plot(g_blank[:, ch_idx], label="Blank")

    ax.set_title(ch_name)
    ax.set_xlabel("Trial")
    ax.set_ylabel("Normalized PSD")

# Hide unused subplots if any
for i in range(n_channels, len(axes)):
    axes[i].axis("off")
handles, labels = axes[0].get_legend_handles_labels()

fig.legend(
    handles,
    labels,
    loc = 'lower right',
    ncol=3,
    fontsize=12,
    frameon=False
)
plt.tight_layout(rect=[0, 0, 0.95, 1])
plt.title ('Band: 20-35Hz')
plt.show()