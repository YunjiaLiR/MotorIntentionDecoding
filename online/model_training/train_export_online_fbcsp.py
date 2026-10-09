import os
import numpy as np

from scipy.io import loadmat, savemat
from scipy.signal import butter, filtfilt

from mne.decoding import CSP

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score


# ============================================================
# Settings: these must later match the MATLAB online app
# ============================================================

subject = 1
runs = range(1, 7)

sfreq = 1000.0

motor_channels = [
    "FC3", "FC1", "FC2", "FC4",
    "C3", "C1", "Cz", "C2", "C4",
    "CP3", "CP1", "CPz", "CP2", "CP4"
]

bands = [(8, 40), (25, 35)]
n_classes = 3
n_components = 4

window_length = 1.0       # seconds
window_step = 0.5         # seconds

# Create windows between 0.5 s and 5.0 s after MI cue.
# With 1-second windows every 0.5 s, this gives 8 windows/trial:
# 0.5-1.5, 1.0-2.0, ..., 4.0-5.0
window_region_start = 0.5
window_region_end = 5.0

filter_order = 4
calibration_day_dir = r"C:\Users\Asus\OneDrive - Imperial College London\EEG4MI\MIGUI\Calibration_data\Subject1\exp1\Day1"
eeg_file_name_pattern = "sub1_exp1_run{}.mat"

save_file = "Models/OnlineFBCSPModel.mat"


# ============================================================
# Functions
# ============================================================

def design_band_filters(bands, sfreq, order):
    filter_b = []
    filter_a = []

    for low, high in bands:
        b, a = butter(
            order,
            [low, high],
            btype="bandpass",
            fs=sfreq
        )
        filter_b.append(b)
        filter_a.append(a)

    return np.asarray(filter_b), np.asarray(filter_a)


def apply_band_filter(X, b, a):
    """
    X shape:
        trials/windows x channels x samples
    """
    return filtfilt(b, a, X, axis=-1)


def load_trial_boundaries_and_data(mat_path):

    mat = loadmat(mat_path, squeeze_me=True, struct_as_record=False, simplify_cells=True)

    data = None
    for candidate_key in ("data", "eeg", "EEG", "signal", "signals"):
        if candidate_key in mat:
            data = np.asarray(mat[candidate_key])
            break

    if data is None:
        raise KeyError(
            f"Could not find EEG data in {mat_path}. Expected one of: data, eeg, EEG, signal, signals."
        )

    event_struct = mat.get("event")
    if isinstance(event_struct, dict):
        start_samples = event_struct.get("start")
        end_samples = event_struct.get("end")
    else:
        start_samples = mat.get("event_start", mat.get("start"))
        end_samples = mat.get("event_end", mat.get("end"))

    if start_samples is None or end_samples is None:
        raise KeyError(
            f"Could not find event start/end in {mat_path}. Expected event.start/event.end or event_start/event_end."
        )

    start_samples = np.asarray(start_samples).squeeze().astype(int)
    end_samples = np.asarray(end_samples).squeeze().astype(int)

    if data.ndim != 2:
        raise ValueError(f"EEG data in {mat_path} must be 2D, got shape {data.shape}.")

    max_end = int(np.max(end_samples))
    if data.shape[0] > max_end and data.shape[1] <= max_end:
        data = data.T

    if data.shape[1] < max_end:
        raise ValueError(
            f"EEG data in {mat_path} is too short for the requested event end indices. "
            f"Data shape: {data.shape}, max event end: {max_end}."
        )

    return data, start_samples, end_samples


def resolve_run_mat_path(run):
    file_name = eeg_file_name_pattern.format(run)
    direct_path = os.path.join(calibration_day_dir, file_name)

    if os.path.exists(direct_path):
        return direct_path

    matches = []
    for root, _, files in os.walk(calibration_day_dir):
        if file_name in files:
            matches.append(os.path.join(root, file_name))

    if not matches:
        raise FileNotFoundError(
            f"Could not find {file_name} under {calibration_day_dir}."
        )

    if len(matches) > 1:
        raise FileExistsError(
            f"Found more than one {file_name}: {matches}. "
            "Please move/rename duplicates or choose the folder explicitly."
        )

    return matches[0]


def load_cue_labels(mat_path):
    mat = loadmat(mat_path, squeeze_me=True, struct_as_record=False, simplify_cells=True)

    labels = mat.get("labels")
    if labels is None:
        parameter = mat.get("parameter")
        if isinstance(parameter, dict):
            labels = parameter.get("randomOrder")

    if labels is None:
        raise KeyError(
            f"Could not find cue labels in {mat_path}. Expected labels or parameter.randomOrder."
        )

    labels = np.asarray(labels).squeeze().astype(int)

    if np.all(np.isin(labels, [5, 6, 7])):
        return labels - 5

    if np.all(np.isin(labels, [1, 2, 3])):
        return labels - 1

    raise ValueError(
        f"Unsupported cue labels in {mat_path}: {np.unique(labels)}. "
        "Expected 1/2/3 or 5/6/7."
    )


def cut_trials_from_boundaries(data, start_samples, end_samples):
    trials = []

    for start_sample, end_sample in zip(start_samples, end_samples):
        start_index = int(start_sample)
        end_index = int(end_sample)

        if end_index <= start_index:
            raise ValueError(
                f"Invalid trial boundary: start={start_index}, end={end_index}."
            )

        trials.append(data[:, start_index:end_index])

    return np.asarray(trials, dtype=object)


def create_sliding_windows(X_trials, y_trials, sfreq,
                           window_length, window_step,
                           region_start, region_end):
    """
    X_trials shape:
        full trials x channels x samples

    Epochs are expected to start at t=0 seconds.
    """
    window_samples = int(round(window_length * sfreq))
    step_samples = int(round(window_step * sfreq))
    start_sample = int(round(region_start * sfreq))
    end_sample = int(round(region_end * sfreq))

    X_windows = []
    y_windows = []
    trial_group = []

    for trial_index, (trial_data, label) in enumerate(
        zip(X_trials, y_trials)
    ):
        trial_duration = trial_data.shape[1] / sfreq
        effective_end = min(region_end, trial_duration)

        if effective_end - region_start < window_length:
            continue

        for start in range(
            start_sample,
            int(round(effective_end * sfreq)) - window_samples + 1,
            step_samples
        ):
            stop = start + window_samples
            X_windows.append(trial_data[:, start:stop])
            y_windows.append(label)
            trial_group.append(trial_index)

    return (
        np.asarray(X_windows),
        np.asarray(y_windows),
        np.asarray(trial_group)
    )


class OnlineFBCSP:
    def __init__(self, bands, filter_b, filter_a,
                 n_classes=3, n_components=4):
        self.bands = bands
        self.filter_b = filter_b
        self.filter_a = filter_a
        self.n_classes = n_classes
        self.n_components = n_components
        self.csps = None

    def fit(self, X, y):
        self.csps = []

        for band_index in range(len(self.bands)):
            X_filtered = apply_band_filter(
                X,
                self.filter_b[band_index],
                self.filter_a[band_index]
            )

            class_csps = []

            for class_index in range(self.n_classes):
                binary_labels = (y == class_index).astype(int)

                csp = CSP(
                    n_components=self.n_components,
                    reg="ledoit_wolf",
                    log=True,
                    norm_trace=False
                )

                csp.fit(X_filtered, binary_labels)
                class_csps.append(csp)

            self.csps.append(class_csps)

        return self

    def transform(self, X):
        features = []

        for band_index in range(len(self.bands)):
            X_filtered = apply_band_filter(
                X,
                self.filter_b[band_index],
                self.filter_a[band_index]
            )

            for class_index in range(self.n_classes):
                csp = self.csps[band_index][class_index]
                features.append(csp.transform(X_filtered))

        return np.concatenate(features, axis=1)


# Load EEG and labels from MATLAB files with event.start/event.end boundaries

trial_data_list = []
trial_label_list = []

for run in runs:
    mat_path = resolve_run_mat_path(run)
    print(f"Loading run {run}: {mat_path}")

    eeg_data, event_start, event_end = load_trial_boundaries_and_data(
        mat_path
    )

    y_run = load_cue_labels(mat_path)

    if len(event_start) != len(event_end):
        raise ValueError(
            f"Run {run} has mismatched event start/end counts: {len(event_start)} vs {len(event_end)}."
        )

    if len(event_start) != len(y_run):
        raise ValueError(
            f"Run {run} has {len(event_start)} trial boundaries but {len(y_run)} labels."
        )

    for start_sample, end_sample in zip(event_start, event_end):
        trial_data_list.append(eeg_data[:, int(start_sample):int(end_sample)])

    trial_label_list.extend(y_run.tolist())

X_trials = np.empty(len(trial_data_list), dtype=object)
X_trials[:] = trial_data_list
# Python classifier labels:
# MATLAB/tablet 5=Left -> Python 0
# MATLAB/tablet 6=Right -> Python 1
# MATLAB/tablet 7=Rest/Blank -> Python 2
y_trials = np.asarray(trial_label_list)

print("Full trials:", X_trials.shape)
print("Trial labels:", np.bincount(y_trials))

# Construct sliding windows matching live prediction

X_windows, y_windows, trial_groups = create_sliding_windows(
    X_trials,
    y_trials,
    sfreq,
    window_length,
    window_step,
    window_region_start,
    window_region_end
)

# Remove window-wise DC offset, matching onlinePreprocessWindow.m.
X_windows = X_windows - X_windows.mean(axis=2, keepdims=True)

print("Windows:", X_windows.shape)
print("Window labels:", np.bincount(y_windows))


# Design exportable filters

filter_b, filter_a = design_band_filters(
    bands,
    sfreq,
    filter_order
)

# Cross-validation evaluation

cv = GroupShuffleSplit(
    n_splits=10,
    test_size=0.2,
    random_state=1
)

accuracy = []
balanced_accuracy = []
macro_f1 = []

for train_indices, test_indices in cv.split(
    X_windows,
    y_windows,
    groups=trial_groups
):

    fbcsp = OnlineFBCSP(
        bands=bands,
        filter_b=filter_b,
        filter_a=filter_a,
        n_classes=n_classes,
        n_components=n_components
    )

    X_train_features = fbcsp.fit(
        X_windows[train_indices],
        y_windows[train_indices]
    ).transform(X_windows[train_indices])

    X_test_features = fbcsp.transform(
        X_windows[test_indices]
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_features)
    X_test_scaled = scaler.transform(X_test_features)

    classifier = LinearDiscriminantAnalysis(
        solver="lsqr",
        shrinkage="auto"
    )

    classifier.fit(X_train_scaled, y_windows[train_indices])
    predictions = classifier.predict(X_test_scaled)

    accuracy.append(
        accuracy_score(y_windows[test_indices], predictions)
    )
    balanced_accuracy.append(
        balanced_accuracy_score(y_windows[test_indices], predictions)
    )
    f1_score_value = f1_score(
        y_windows[test_indices],
        predictions,
        average="macro"
    )
    macro_f1.append(f1_score_value)

print(
    "FBCSP + LDA window accuracy: "
    f"{np.mean(accuracy):.3f} +/- {np.std(accuracy):.3f}"
)

print(
    "FBCSP + LDA balanced accuracy: "
    f"{np.mean(balanced_accuracy):.3f} +/- "
    f"{np.std(balanced_accuracy):.3f}"
)

print(
    "FBCSP + LDA macro F1: "
    f"{np.mean(macro_f1):.3f} +/- {np.std(macro_f1):.3f}"
)

# Train final model on all available windows
final_fbcsp = OnlineFBCSP(
    bands=bands,
    filter_b=filter_b,
    filter_a=filter_a,
    n_classes=n_classes,
    n_components=n_components
)

X_features = final_fbcsp.fit(
    X_windows,
    y_windows
).transform(X_windows)

print("FBCSP feature size:", X_features.shape)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_features)

final_classifier = LinearDiscriminantAnalysis(
    solver="lsqr",
    shrinkage="auto"
)

final_classifier.fit(X_scaled, y_windows)

# Export learned FBCSP and LDA model to MATLAB
number_bands = len(bands)
number_channels = len(motor_channels)

spatial_filters = np.zeros(
    (
        number_bands,
        n_classes,
        n_components,
        number_channels
    )
)

for band_index in range(number_bands):
    for class_index in range(n_classes):
        csp = final_fbcsp.csps[band_index][class_index]

        spatial_filters[band_index, class_index, :, :] = (
            csp.filters_[:n_components, :]
        )

# Export app labels rather than Python zero-based labels:
# Python 0=Left -> MATLAB/tablet 5
# Python 1=Right -> MATLAB/tablet 6
# Python 2=Rest -> MATLAB/tablet 7
class_labels = np.array([5, 6, 7])

model = {
    "bands": np.asarray(bands, dtype=float),
    "filterB": filter_b,
    "filterA": filter_a,
    "spatialFilters": spatial_filters,
    "samplingRate": np.asarray([[sfreq]]),
    "nBands": np.asarray([[number_bands]]),
    "nClasses": np.asarray([[n_classes]]),
    "nComponents": np.asarray([[n_components]]),
    "windowLength": np.asarray([[window_length]]),
    "predictEvery": np.asarray([[window_step]]),
    "channelNames": np.asarray(motor_channels, dtype=object),
    "scalerMean": scaler.mean_,
    "scalerScale": scaler.scale_,
    "ldaCoef": final_classifier.coef_,
    "ldaIntercept": final_classifier.intercept_,
    "classLabels": class_labels
}

os.makedirs(os.path.dirname(save_file), exist_ok=True)
savemat(save_file, {"model": model})

print("Saved:", save_file)