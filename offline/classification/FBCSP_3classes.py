"""
Direct 3-class FBCSP classifier: Left vs Right vs Blank.

Run after `epochs` and `cue_labels` have been created.
The final classifier makes one multiclass decision directly. The one-vs-rest
CSP models are feature extractors only; there is no pairwise voting or
hierarchical classification.
"""

from collections import Counter

from tqdm import tqdm

import mne
from scipy.io import loadmat
import numpy as np
from mne.decoding import CSP
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler


CLASS_NAMES = ("Left", "Right", "Blank")
MOTOR_CHANNELS = [
    "FC3", "FC1", "FC2", "FC4",
    "C3", "C1", "Cz", "C2", "C4",
    "CP3", "CP1", "CPz", "CP2", "CP4",
]
FILTER_BANK = [
    (8, 12),
    (12, 16),
    (16, 20),
    (20, 24),
    (24, 28),
    (28, 32),
    (32, 36),
    (36, 40),
]


raws = []
for i in tqdm(range(1,7), desc = "Loading EEG data"):
    raw = mne.io.read_raw_eeglab(f'eeg/step1_preprocess/Subject 1/run{i}_pruned with ICA.set',preload=True)
    montage = raw.get_montage()
    raws.append(raw)
raw_all = mne.concatenate_raws(raws)

events, event_id = mne.events_from_annotations(raw_all)
annotations = raw_all.annotations.copy()
rename_map = {}
for d in set(annotations.description):
    if str(d).lower() in ("boundary","edge boundary"):
        rename_map[d] = "BAD_boundary"
raw_all.set_annotations(annotations.rename(rename_map))

print(f"Annotation description counts:", Counter(raw_all.annotations.description))

inv = {v:k for k,v in event_id.items()}
event_names = [inv[e] for e in events[:,2]]
print(f"Event counts:", Counter(event_names))
print(f"S 15 count in events:", np.sum(events[:,2] == event_id["S 15"]))


#Epochs
event_id_use = {'S 15': event_id['S 15']}
epochs = mne.Epochs(raw_all,events, event_id=event_id_use, tmin=-2.5,tmax=5.0,baseline=(-2,0),preload=True,reject_by_annotation=True)
force_mat = loadmat('force/cue_labels_subject1.mat')
cue_labels = force_mat['labels'].squeeze()
print("class counts from cue_labels:", np.bincount(cue_labels.astype(int)))
print("epoch counts:", len(epochs))


idx_left = cue_labels == 1
idx_right = cue_labels == 2
idx_blank = cue_labels == 3
epochs_left = epochs[idx_left]
epochs_right = epochs[idx_right]
epochs_blank = epochs[idx_blank]

# ===== Build 3-class epochs and labels =====
epochs_3 = mne.concatenate_epochs([epochs_left, epochs_right, epochs_blank])
y_3 = np.r_[np.zeros(len(epochs_left), dtype=int),          # Left = 0
    np.ones(len(epochs_right), dtype=int),          # Right = 1
    2 * np.ones(len(epochs_blank), dtype=int)       # Blank = 2
]

epochs_3 = epochs_3.crop(tmin=0.5, tmax=3)

motor_chs = ["FC3","FC1","FC2","FC4","C3","C1","Cz","C2","C4","CP3","CP1","CPz","CP2","CP4"]
roi_chs = ['Fp1', 'Fz', 'F3', 'F7', 'FT9', 'FC5', 'FC1', 'C3', 'T7', 'TP9', 'CP5', 'CP1', 'Pz', 'TP10', 'CP6', 'CP2', 'Cz', 'C4', 'T8', 'FT10', 'FC6', 'FC2', 'F4', 'F8', 'Fp2', 'AF7', 'AF3', 'AFz', 'F1', 'F5', 'FT7', 'FC3', 'C1', 'C5', 'TP7', 'CP3', 'P1', 'P5', 'P6', 'P2', 'CPz', 'CP4', 'TP8', 'C6', 'C2', 'FC4', 'FT8', 'F6', 'AF8', 'AF4', 'F2']
epochs_motor = epochs_3.copy().pick_channels(motor_chs, ordered=True)
X_use = epochs_motor.get_data()
sfreq = epochs_motor.info["sfreq"]

print("X_use shape:", X_use.shape)
print("y_3 shape:", y_3.shape)
print("class counts:", np.bincount(y_3))

# bands = [(8,12), (12,16), (16,20), (20,24), (24,28), (28,35),(35,40)]
bands = [(8,40),(25,35)]


class DirectThreeClassFBCSP(BaseEstimator, TransformerMixin):
    """Filter-bank CSP features for a direct 3-class classifier."""

    def __init__(
        self,
        sfreq,
        bands=FILTER_BANK,
        n_components=2,
        reg="ledoit_wolf",
    ):
        self.sfreq = sfreq
        self.bands = bands
        self.n_components = n_components
        self.reg = reg

    def fit(self, X, y):
        classes = np.unique(y)
        if not np.array_equal(classes, np.array([0, 1, 2])):
            raise ValueError(f"Expected labels [0, 1, 2], got {classes}.")
        # Accept either raw X (n_epochs, n_channels, n_times) or a list of prefiltered
        # arrays one per band. If prefiltered, `X` should be a list/tuple matching
        # `self.bands`.
        self.csps_ = []
        if isinstance(X, (list, tuple)):
            X_bands = X
        else:
            X_bands = [self._filter(X, fmin, fmax) for fmin, fmax in self.bands]

        for X_band in X_bands:
            band_csps = []
            for class_id in classes:
                csp = CSP(
                    n_components=self.n_components,
                    reg=self.reg,
                    log=True,
                    norm_trace=False,
                )
                csp.fit(X_band, (y == class_id).astype(int))
                band_csps.append(csp)
            self.csps_.append(band_csps)
        return self

    def transform(self, X):
        features = []
        # Support prefiltered input (list of arrays) or raw X
        if isinstance(X, (list, tuple)):
            X_bands = X
        else:
            X_bands = [self._filter(X, fmin, fmax) for fmin, fmax in self.bands]

        for X_band, band_csps in zip(X_bands, self.csps_):
            for csp in band_csps:
                features.append(csp.transform(X_band))
        return np.concatenate(features, axis=1)

    def _filter(self, X, fmin, fmax):
        return mne.filter.filter_data(
            X,
            sfreq=self.sfreq,
            l_freq=fmin,
            h_freq=fmax,
            verbose=False,
        )


def align_labels_with_epochs(epochs, cue_labels):
    """
    Keep labels aligned if MNE dropped epochs due to BAD annotations.

    If cue_labels already contains exactly one label per retained epoch, no
    indexing is needed. Otherwise, epochs.selection identifies retained events.
    """
    labels = np.asarray(cue_labels).squeeze().astype(int)
    if len(labels) == len(epochs):
        return labels

    selection = np.asarray(epochs.selection)
    if selection.size and selection.max() < len(labels):
        return labels[selection]

    raise ValueError(
        f"Cannot align {len(labels)} cue labels with {len(epochs)} epochs."
    )


def run_direct_three_class_fbcsp(
    epochs,
    cue_labels,
    channels=MOTOR_CHANNELS,
    bands=FILTER_BANK,
    tmin=0.5,
    tmax=3.0,
    n_components=2,
    n_splits=20,
    test_size=0.2,
    random_state=42,
):
    labels = align_labels_with_epochs(epochs, cue_labels)
    keep = np.isin(labels, [1, 2, 3])
    y = labels[keep] - 1  # Left=0, Right=1, Blank=2

    available_channels = [ch for ch in channels if ch in epochs.ch_names]
    if not available_channels:
        raise ValueError("None of the requested motor channels exist in epochs.")

    X = (
        epochs[keep]
        .copy()
        .pick(available_channels)
        .crop(tmin=tmin, tmax=tmax)
        .get_data(copy=True)
    )
    sfreq = epochs.info["sfreq"]

    print("X shape:", X.shape)
    print("Class counts:", {
        CLASS_NAMES[class_id]: int(count)
        for class_id, count in sorted(Counter(y).items())
    })

    cv = StratifiedShuffleSplit(
        n_splits=n_splits,
        test_size=test_size,
        random_state=random_state,
    )
    classifier = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    accs, baccs, f1s, cms = [], [], [], []
    # Pre-filter the full dataset once per band to avoid repeated filtering.
    print("Prefiltering bands for entire dataset...")
    X_bands = [
        mne.filter.filter_data(
            X.copy(),
            sfreq=sfreq,
            l_freq=fmin,
            h_freq=fmax,
            verbose=False,
        )
        for fmin, fmax in bands
    ]

    for train_idx, test_idx in cv.split(X, y):
        fbcsp = DirectThreeClassFBCSP(
            sfreq=sfreq,
            bands=bands,
            n_components=n_components,
        )
        # Pass prefiltered band arrays sliced by indices.
        X_train_bands = [xb[train_idx] for xb in X_bands]
        X_test_bands = [xb[test_idx] for xb in X_bands]

        X_train = fbcsp.fit_transform(X_train_bands, y[train_idx])
        X_test = fbcsp.transform(X_test_bands)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        model = clone(classifier)
        model.fit(X_train, y[train_idx])
        y_pred = model.predict(X_test)

        accs.append(accuracy_score(y[test_idx], y_pred))
        baccs.append(balanced_accuracy_score(y[test_idx], y_pred))
        f1s.append(f1_score(y[test_idx], y_pred, average="macro"))
        cms.append(
            confusion_matrix(
                y[test_idx],
                y_pred,
                labels=[0, 1, 2],
                normalize="true",
            )
        )

    accs = np.asarray(accs)
    baccs = np.asarray(baccs)
    f1s = np.asarray(f1s)
    mean_cm = np.mean(cms, axis=0)

    print("Chance level: 0.333")
    print(f"Accuracy:          {accs.mean():.3f} +/- {accs.std():.3f}")
    print(f"Balanced accuracy: {baccs.mean():.3f} +/- {baccs.std():.3f}")
    print(f"Macro F1:          {f1s.mean():.3f} +/- {f1s.std():.3f}")
    print("Mean confusion matrix (rows=true, columns=predicted):")
    print("                 Left  Right  Blank")
    for name, row in zip(CLASS_NAMES, mean_cm):
        print(f"{name:>5}  {np.round(row, 3)}")

    # Fit deployable objects once using every available trial.
    final_fbcsp = DirectThreeClassFBCSP(
        sfreq=sfreq,
        bands=bands,
        n_components=n_components,
    )
    X_features = final_fbcsp.fit_transform(X, y)
    final_scaler = StandardScaler().fit(X_features)
    final_model = clone(classifier).fit(final_scaler.transform(X_features), y)

    return {
        "fbcsp": final_fbcsp,
        "scaler": final_scaler,
        "model": final_model,
        "channels": available_channels,
        "bands": bands,
        "accuracy": accs,
        "balanced_accuracy": baccs,
        "macro_f1": f1s,
        "mean_confusion_matrix": mean_cm,
    }


# Run the direct classifier on the 3-class epochs we constructed above.
direct_fbcsp_result = run_direct_three_class_fbcsp(epochs_3, cue_labels)
