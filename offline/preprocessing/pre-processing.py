import numpy as np
from pathlib import Path

import matplotlib.pyplot as plt
import mne
from scipy.io import loadmat

from plot_tfr_topomap import plot_average_tfr, plot_topomaps


FS = 1000
WINDOW_LENGTH = FS
STEP_LENGTH = round(0.5 * FS)
EXPECTED_WINDOW_NUM = 8
DEFAULT_FREQS = np.arange(8, 41, 1)
DEFAULT_TIME_WINDOWS = [(0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)]
MOTOR_CHANNEL_NAMES = [
    "FC3", "FC1", "FC2", "FC4",
    "C3", "C1", "Cz", "C2", "C4",
    "CP3", "CP1", "CPz", "CP2", "CP4",
]


def _as_1d_int(values):
    return np.asarray(values).reshape(-1).astype(int)


def _prepare_eeg(eeg, motor_channel_indices):
    eeg = np.asarray(eeg, dtype=float)
    indices = np.asarray(motor_channel_indices, dtype=int).reshape(-1)

    if indices.size == 0:
        raise ValueError("motorChannelIndices must not be empty.")

    if indices.min() == 1:
        indices = indices - 1

    if eeg.ndim != 2:
        raise ValueError(f"EEG data must be 2D, got shape {eeg.shape}.")

    if eeg.shape[0] > eeg.shape[1]:
        eeg = eeg.T

    if indices.max() >= eeg.shape[0]:
        raise ValueError(
            f"motorChannelIndices do not match EEG shape {eeg.shape}."
        )

    eeg = eeg[indices, :]
    eeg = eeg - eeg.mean(axis=1, keepdims=True)
    return eeg


def onlinePreProcess(eeg, fs, motorChannelIndices):
    return _prepare_eeg(eeg, motorChannelIndices)


def _load_run_file(mat_file):
    mat = loadmat(mat_file, squeeze_me=True, struct_as_record=False, simplify_cells=True)

    eeg = mat["data"]
    starts = _as_1d_int(mat["event"]["start"])
    labels = _as_1d_int(mat["parameter"]["randomOrder"])

    return eeg, starts, labels


def prepare_data(subject, exp, day, runs, trainRuns, motorChannelIndices):
    folder = Path("Calibration_data") / f"Subject{subject}" / f"exp{exp}" / f"Day{day}" / "arm_oriented"

    trainData = []
    trainLabels = []
    testData = []
    testLabels = []

    for r in runs:
        matFile = folder / f"sub{subject}_exp{exp}_run{r}.mat"
        eeg, starts, labels = _load_run_file(matFile)
        eeg = onlinePreProcess(eeg, FS, motorChannelIndices)

        if len(starts) != len(labels):
            raise ValueError(
                f"Run {r}: event starts and labels have different lengths."
            )

        for startSample, trialLabel in zip(starts, labels):
            for w in range(1, EXPECTED_WINDOW_NUM + 1):
                startWindow = int(startSample + (w - 1) * STEP_LENGTH)
                endWindow = int(startWindow + WINDOW_LENGTH)

                if endWindow > eeg.shape[1]:
                    break

                windowData = eeg[:, startWindow:endWindow]

                if r in trainRuns:
                    trainData.append(windowData)
                    trainLabels.append(int(trialLabel))
                else:
                    testData.append(windowData)
                    testLabels.append(int(trialLabel))

    trainLabels = np.asarray(trainLabels, dtype=int)
    testLabels = np.asarray(testLabels, dtype=int)

    print(f"Training windows: {len(trainData)}")
    print(f"Testing windows: {len(testData)}")

    return trainData, trainLabels, testData, testLabels


def _windows_to_epochs(windows, ch_names, fs):
    if len(windows) == 0:
        return None

    data = np.stack(windows, axis=0)
    info = mne.create_info(ch_names=list(ch_names), sfreq=fs, ch_types="eeg")
    info.set_montage(mne.channels.make_standard_montage("standard_1020"), on_missing="ignore")
    return mne.EpochsArray(data, info, verbose=False)


def _label_name(label):
    mapping = {1: "Left", 2: "Right", 3: "Blank"}
    return mapping.get(int(label), f"Class {int(label)}")


def plot_tfr_and_topomap(
    data,
    labels,
    title,
    fs=FS,
    channel_names=MOTOR_CHANNEL_NAMES,
    channels_to_plot=("FC3", "FC1", "C3", "Cz"),
    freqs=None,
    time_windows=None,
    fmin=25,
    fmax=35,
    baseline=None,
):
    if freqs is None:
        freqs = DEFAULT_FREQS
    if time_windows is None:
        time_windows = DEFAULT_TIME_WINDOWS

    powers = {}
    labels = np.asarray(labels, dtype=int)

    for label in np.unique(labels):
        class_windows = [window for window, window_label in zip(data, labels) if window_label == label]
        epochs = _windows_to_epochs(class_windows, channel_names, fs)
        if epochs is None:
            continue

        power = mne.time_frequency.tfr_morlet(
            epochs,
            freqs=freqs,
            n_cycles=freqs / 4.0,
            use_fft=True,
            return_itc=False,
            average=True,
            picks="eeg",
            decim=1,
        )

        if baseline is not None:
            power.apply_baseline(baseline=baseline, mode="zscore")

        powers[_label_name(label)] = power

    if not powers:
        print(f"No data available for {title}.")
        return None, None

    fig_tfr, _ = plot_average_tfr(
        powers,
        list(channels_to_plot),
        time_windows,
        fmin=fmin,
        fmax=fmax,
    )
    fig_topo, _ = plot_topomaps(
        powers,
        time_windows,
        fmin=fmin,
        fmax=fmax,
    )

    fig_tfr.suptitle(f"{title} TFR", fontsize=16)
    fig_topo.suptitle(f"{title} Topomap", fontsize=16)

    plt.show(block=False)
    plt.pause(0.1)

    return fig_tfr, fig_topo


if __name__ == "__main__":
    subject = 1
    exp = 1
    day = 1
    runs = range(1, 7)
    trainRuns = range(1, 5)
    motorChannelIndices = [39, 7, 29, 58, 8, 40, 24, 57, 25, 43, 12, 53, 23, 54]

    trainData, trainLabels, testData, testLabels = prepare_data(
        subject,
        exp,
        day,
        runs,
        trainRuns,
        motorChannelIndices,
    )

    plot_tfr_and_topomap(trainData, trainLabels, "Training")
    if len(testData) > 0:
        plot_tfr_and_topomap(testData, testLabels, "Testing")