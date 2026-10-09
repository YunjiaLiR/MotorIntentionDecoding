"""Buffered online-style experiment for hierarchical MI decoding.

This keeps the original null-space hierarchy but avoids filtering isolated
one-second task windows. Each prediction window is processed with past and
future context, then cropped back to its central one-second region.
"""

from pathlib import Path

import mne
import numpy as np
from scipy.io import loadmat, savemat
from scipy.signal import filtfilt, firwin, hilbert
from scipy.ndimage import uniform_filter1d
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


RUNS = range(1, 7)
EEG_TEMPLATE = "eeg/step1_preprocess/Subject 1/run{}.set"
LABEL_FILE = "force/cue_labels_subject1.mat"
MODEL_FILE = Path("Models") / "OnlineHierarchicalPCA.mat"

MOTOR_CHANNELS = [
    "FC3", "FC1", "FC2", "FC4", "C3", "C1", "Cz",
    "C2", "C4", "CP3", "CP1", "CPz", "CP2", "CP4",
]

LEFT_LABEL = 1
RIGHT_LABEL = 2
REST_LABEL = 3

BANDS = [(8.0, 13.0), (13.0, 22.0), (22.0, 35.0), (35.0, 40.0)]

WINDOW_SECONDS = 1.0
WINDOW_STEP_SECONDS = 0.5
PAST_CONTEXT_SECONDS = 1.0
FUTURE_CONTEXT_SECONDS = 0.5
TASK_START_SECONDS = 0.0
TASK_STOP_SECONDS = 5.0
BASELINE_START_SECONDS = -2.0
BASELINE_STOP_SECONDS = -0.5
SMOOTH_SECONDS = 0.5
FILTER_TAPS = 101
NULL_VARIANCE = 0.95
EXCLUSIVE_VARIANCE = 0.95
MAX_EXCLUSIVE_COMPONENTS = 12


def design_filters(sfreq):
    return np.asarray([
        firwin(FILTER_TAPS, [low, high], pass_zero=False, fs=sfreq)
        for low, high in BANDS
    ])


def common_median_reference(data):
    return data - np.median(data, axis=0, keepdims=True)


def spectral_envelope(data, filter_b, smooth_samples):
    """Return channels x bands x samples Hilbert envelopes."""
    envelopes = []
    for b in filter_b:
        filtered = filtfilt(b, [1.0], data, axis=-1)
        envelope = np.abs(hilbert(filtered, axis=-1))
        if smooth_samples > 1:
            envelope = uniform_filter1d(
                envelope, size=smooth_samples, axis=-1, mode="reflect"
            )
        envelopes.append(envelope)
    return np.stack(envelopes, axis=1)


def build_windows(epochs, labels, filter_b):
    """Build online-style windows with context for signal processing."""
    data = epochs.get_data()
    sfreq = float(epochs.info["sfreq"])

    baseline_start = int(round((BASELINE_START_SECONDS - epochs.tmin) * sfreq))
    baseline_stop = int(round((BASELINE_STOP_SECONDS - epochs.tmin) * sfreq))
    task_start = int(round((TASK_START_SECONDS - epochs.tmin) * sfreq))
    task_stop = int(round((TASK_STOP_SECONDS - epochs.tmin) * sfreq))
    window_samples = int(round(WINDOW_SECONDS * sfreq))
    step_samples = int(round(WINDOW_STEP_SECONDS * sfreq))
    past_context_samples = int(round(PAST_CONTEXT_SECONDS * sfreq))
    future_context_samples = int(round(FUTURE_CONTEXT_SECONDS * sfreq))
    smooth_samples = int(round(SMOOTH_SECONDS * sfreq))

    features, window_labels, groups = [], [], []

    for trial_index, (trial_data, label) in enumerate(zip(data, labels)):
        trial_data = common_median_reference(trial_data)
        trial_env = spectral_envelope(trial_data, filter_b, smooth_samples)

        # Use the same reference baseline for every window in this trial.
        baseline_env = trial_env[:, :, baseline_start:baseline_stop]
        baseline_mean = baseline_env.mean(axis=-1, keepdims=True)
        baseline_std = baseline_env.std(axis=-1, keepdims=True) + 1e-12

        for start in range(task_start, task_stop - window_samples + 1, step_samples):
            stop = start + window_samples

            central_start = start
            central_stop = stop
            buffer_start = central_start - past_context_samples
            buffer_stop = central_stop + future_context_samples

            if buffer_start < 0 or buffer_stop > trial_env.shape[-1]:
                continue

            task_env = trial_env[:, :, central_start:central_stop]

            task_z = (task_env - baseline_mean) / baseline_std
            features.append(task_z.reshape(-1))
            window_labels.append(label)
            groups.append(trial_index)

    return np.asarray(features), np.asarray(window_labels), np.asarray(groups)


def select_component_count(explained_variance_ratio, threshold, maximum=None):
    count = int(np.searchsorted(np.cumsum(explained_variance_ratio), threshold) + 1)
    if maximum is not None:
        count = min(count, maximum)
    return max(1, count)


def fit_exclusive_space(scaled, labels, reference_label, target_label):
    reference = scaled[labels == reference_label]
    target = scaled[labels == target_label]
    reference_mean = reference.mean(axis=0)

    reference_pca = PCA(svd_solver="randomized", random_state=0).fit(
        reference - reference_mean
    )
    reference_count = select_component_count(
        reference_pca.explained_variance_ratio_, NULL_VARIANCE
    )
    reference_basis = reference_pca.components_[:reference_count]

    centered_target = target - reference_mean
    target_residual = centered_target - (
        centered_target @ reference_basis.T @ reference_basis
    )

    exclusive_pca = PCA(svd_solver="randomized", random_state=0).fit(target_residual)
    exclusive_count = select_component_count(
        exclusive_pca.explained_variance_ratio_,
        EXCLUSIVE_VARIANCE,
        MAX_EXCLUSIVE_COMPONENTS,
    )
    exclusive_basis = exclusive_pca.components_[:exclusive_count]
    projection_matrix = exclusive_basis.T
    projection_offset = -reference_mean @ projection_matrix
    return projection_matrix, projection_offset


def fit_stage(features, labels, exclusive_pairs, classifier_kind="lda"):
    feature_scaler = StandardScaler().fit(features)
    scaled = feature_scaler.transform(features)

    spaces = [
        fit_exclusive_space(scaled, labels, reference_label, target_label)
        for reference_label, target_label in exclusive_pairs
    ]
    projection_matrix = np.concatenate([space[0] for space in spaces], axis=1)
    projection_offset = np.concatenate([space[1] for space in spaces])
    projected = scaled @ projection_matrix + projection_offset

    coordinate_scaler = StandardScaler().fit(projected)
    coordinates = coordinate_scaler.transform(projected)

    if classifier_kind == "lda":
        classifier = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    elif classifier_kind == "logreg":
        classifier = LogisticRegression(
            max_iter=5000, class_weight="balanced", C=0.5, solver="liblinear"
        )
    else:
        raise ValueError("classifier_kind must be 'lda' or 'logreg'.")

    classifier.fit(coordinates, labels)

    decision_threshold = 0.5
    if np.unique(labels).size == 2 and hasattr(classifier, "predict_proba"):
        positive_scores = classifier.predict_proba(coordinates)[:, 1]
        candidate_thresholds = np.linspace(0.05, 0.95, 91)
        threshold_scores = [
            balanced_accuracy_score(labels, (positive_scores >= threshold).astype(int))
            for threshold in candidate_thresholds
        ]
        decision_threshold = float(
            candidate_thresholds[int(np.argmax(threshold_scores))]
        )

    return {
        "feature_scaler": feature_scaler,
        "projection_matrix": projection_matrix,
        "projection_offset": projection_offset,
        "coordinate_scaler": coordinate_scaler,
        "classifier": classifier,
        "classifier_kind": classifier_kind,
        "decision_threshold": decision_threshold,
    }


def predict_stage(features, stage):
    scaled = stage["feature_scaler"].transform(features)
    projected = scaled @ stage["projection_matrix"] + stage["projection_offset"]
    coordinates = stage["coordinate_scaler"].transform(projected)

    if "decision_threshold" in stage and hasattr(stage["classifier"], "predict_proba"):
        positive_scores = stage["classifier"].predict_proba(coordinates)[:, 1]
        return (positive_scores >= stage["decision_threshold"]).astype(int)
    return stage["classifier"].predict(coordinates)


def fit_hierarchical(features, labels):
    stage1_labels = (labels != REST_LABEL).astype(int)
    stage1 = fit_stage(
        features, stage1_labels, exclusive_pairs=[(0, 1)], classifier_kind="logreg"
    )

    mi_mask = labels != REST_LABEL
    stage2_labels = (labels[mi_mask] == RIGHT_LABEL).astype(int)
    stage2 = fit_stage(
        features[mi_mask],
        stage2_labels,
        exclusive_pairs=[(1, 0), (0, 1)],
        classifier_kind="logreg",
    )
    return stage1, stage2


def predict_hierarchical(features, stage1, stage2):
    mi_prediction = predict_stage(features, stage1)
    predictions = np.full(features.shape[0], REST_LABEL, dtype=int)
    mi_rows = mi_prediction == 1
    if np.any(mi_rows):
        left_right = predict_stage(features[mi_rows], stage2)
        predictions[mi_rows] = np.where(left_right == 0, LEFT_LABEL, RIGHT_LABEL)
    return predictions


def majority_vote_balanced_accuracy(predictions, labels, groups):
    trial_true, trial_predicted = [], []
    for group in np.unique(groups):
        mask = groups == group
        values, counts = np.unique(predictions[mask], return_counts=True)
        trial_true.append(labels[mask][0])
        trial_predicted.append(values[np.argmax(counts)])
    return balanced_accuracy_score(trial_true, trial_predicted)


def process_split(train_index, test_index, features, labels, groups):
    stage1, stage2 = fit_hierarchical(features[train_index], labels[train_index])
    predicted = predict_hierarchical(features[test_index], stage1, stage2)

    score = balanced_accuracy_score(labels[test_index], predicted)
    trial_score = majority_vote_balanced_accuracy(
        predicted, labels[test_index], groups[test_index]
    )
    cm = confusion_matrix(
        labels[test_index], predicted, labels=[LEFT_LABEL, RIGHT_LABEL, REST_LABEL]
    )

    test_labels = labels[test_index]
    stage1_true = (test_labels != REST_LABEL).astype(int)
    stage1_predicted = predict_stage(features[test_index], stage1)
    stage1_score = balanced_accuracy_score(stage1_true, stage1_predicted)

    test_mi = test_labels != REST_LABEL
    stage2_true = (test_labels[test_mi] == RIGHT_LABEL).astype(int)
    if np.any(test_mi):
        stage2_predicted = predict_stage(features[test_index][test_mi], stage2)
        stage2_score = balanced_accuracy_score(stage2_true, stage2_predicted)
    else:
        stage2_score = np.nan

    return score, trial_score, cm, stage1_score, stage2_score


def export_stage(stage):
    classifier = stage["classifier"]
    return {
        "featureScalerMean": stage["feature_scaler"].mean_,
        "featureScalerScale": stage["feature_scaler"].scale_,
        "projectionMatrix": stage["projection_matrix"],
        "projectionOffset": stage["projection_offset"],
        "coordinateScalerMean": stage["coordinate_scaler"].mean_,
        "coordinateScalerScale": stage["coordinate_scaler"].scale_,
        "classifierKind": stage["classifier_kind"],
        "decisionThreshold": stage["decision_threshold"],
        "classifierCoef": classifier.coef_,
        "classifierIntercept": classifier.intercept_,
        "classifierClasses": classifier.classes_,
    }


def main():
    mne.set_log_level("ERROR")
    raws = [
        mne.io.read_raw_eeglab(EEG_TEMPLATE.format(run), preload=True)
        for run in RUNS
    ]
    raw_all = mne.concatenate_raws(raws)

    annotations = raw_all.annotations.copy()
    rename_map = {
        description: "BAD_boundary"
        for description in set(annotations.description)
        if str(description).lower() in ("boundary", "edge boundary")
    }
    if rename_map:
        raw_all.set_annotations(annotations.rename(rename_map))

    events, event_id = mne.events_from_annotations(raw_all)
    cue_code = event_id["S 15"]
    cue_events = events[events[:, 2] == cue_code]
    epochs = mne.Epochs(
        raw_all,
        cue_events,
        event_id={"S 15": cue_code},
        tmin=-2.0,
        tmax=5.0,
        baseline=None,
        preload=True,
        reject_by_annotation=True,
    ).pick_channels(MOTOR_CHANNELS, ordered=True)

    cue_labels = loadmat(LABEL_FILE)["labels"].squeeze().astype(int)
    labels = cue_labels[epochs.selection]

    sfreq = float(epochs.info["sfreq"])
    filter_b = design_filters(sfreq)
    features, window_labels, groups = build_windows(epochs, labels, filter_b)
    print("Feature matrix:", features.shape)
    print("Window labels:", dict(zip(*np.unique(window_labels, return_counts=True))))

    splitter = GroupShuffleSplit(n_splits=10, test_size=0.2, random_state=1)
    results = [
        process_split(train_index, test_index, features, window_labels, groups)
        for train_index, test_index in splitter.split(features, window_labels, groups)
    ]

    scores, trial_scores, cms, stage1_scores, stage2_scores = zip(*results)
    print(f"Window hierarchical balanced accuracy: {np.mean(scores):.3f} +/- {np.std(scores):.3f}")
    print(f"Trial majority-vote balanced accuracy: {np.mean(trial_scores):.3f} +/- {np.std(trial_scores):.3f}")
    print(f"Stage 1 Rest-vs-MI balanced accuracy: {np.mean(stage1_scores):.3f} +/- {np.std(stage1_scores):.3f}")
    print(f"Stage 2 Left-vs-Right balanced accuracy: {np.mean(stage2_scores):.3f} +/- {np.std(stage2_scores):.3f}")
    print("Mean confusion matrix [Left, Right, Rest]:")
    print(np.round(np.mean(cms, axis=0), 2))

    stage1, stage2 = fit_hierarchical(features, window_labels)
    print("Stage 1 exclusive dimensions:", stage1["projection_matrix"].shape[1])
    print("Stage 2 exclusive dimensions:", stage2["projection_matrix"].shape[1])

    model = {
        "samplingRate": sfreq,
        "bands": np.asarray(BANDS),
        "filterB": filter_b,
        "windowLength": WINDOW_SECONDS,
        "predictEvery": WINDOW_STEP_SECONDS,
        "pastContextLength": PAST_CONTEXT_SECONDS,
        "futureContextLength": FUTURE_CONTEXT_SECONDS,
        "baselineLength": abs(BASELINE_STOP_SECONDS - BASELINE_START_SECONDS),
        "smoothSamples": int(round(SMOOTH_SECONDS * sfreq)),
        "channelNames": np.asarray(MOTOR_CHANNELS, dtype=object),
        "stage1": export_stage(stage1),
        "stage2": export_stage(stage2),
    }
    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    savemat(MODEL_FILE, {"model": model})
    print("Saved:", MODEL_FILE)


if __name__ == "__main__":
    main()