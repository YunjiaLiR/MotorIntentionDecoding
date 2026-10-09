from pathlib import Path
import os
import numpy as np
import matplotlib.pyplot as plt

import hierarchical_PCA as hp


OUTDIR = Path("plots")


def ensure_outdir():
    OUTDIR.mkdir(exist_ok=True)


def save_fig(fig, name):
    path = OUTDIR / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", path)


def plot_scree(explained_ratio, title):
    fig, ax = plt.subplots()
    ax.plot(np.arange(1, explained_ratio.size + 1), np.cumsum(explained_ratio), marker="o")
    ax.set_xlabel("Component")
    ax.set_ylabel("Cumulative explained variance")
    ax.set_title(title)
    return fig


def channel_weights_from_component(component, n_channels, n_bands, n_time):
    # component is length n_channels * n_bands * n_time
    arr = component.reshape(n_channels, n_bands, n_time)
    # summarize bands/time per channel using RMS
    weights = np.sqrt(np.mean(arr ** 2, axis=(1, 2)))
    return weights


def plot_channel_bar(weights, channels, title):
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.bar(np.arange(len(channels)), weights)
    ax.set_xticks(np.arange(len(channels)))
    ax.set_xticklabels(channels, rotation=45, ha="right")
    ax.set_title(title)
    return fig


def plot_scatter(coords, labels, classes, title):
    fig, ax = plt.subplots()
    if coords.shape[1] < 2:
        ax.plot(coords[:, 0], np.zeros_like(coords[:, 0]), 'o')
    else:
        for lab in np.unique(labels):
            mask = labels == lab
            ax.scatter(coords[mask, 0], coords[mask, 1], label=classes.get(lab, str(lab)), alpha=0.6)
        ax.legend()
    ax.set_title(title)
    return fig


def plot_score_hist(scores, labels, title):
    fig, ax = plt.subplots()
    for lab in np.unique(labels):
        mask = labels == lab
        ax.hist(scores[mask], bins=30, alpha=0.6, label=str(lab))
    ax.legend()
    ax.set_title(title)
    return fig


def visualize(features, labels, sfreq, stage1, stage2):
    ensure_outdir()
    n_channels = len(hp.MOTOR_CHANNELS)
    n_bands = len(hp.BANDS)
    n_time = int(round(hp.WINDOW_SECONDS * sfreq))

    # Stage1: Rest (0) vs MI (1) labels inside stage1
    # Use stage1's scalers and projection to compute coordinates
    fs = stage1["feature_scaler"]
    proj = stage1["projection_matrix"]
    offset = stage1["projection_offset"]
    coord_scaler = stage1["coordinate_scaler"]

    scaled = fs.transform(features)
    projected = scaled @ proj + offset
    coords = coord_scaler.transform(projected)

    # Scree and components for the Rest reference used by stage1: replicate fit_exclusive_space logic
    stage1_labels = (labels != hp.REST_LABEL).astype(int)
    reference = scaled[stage1_labels == 0]
    target = scaled[stage1_labels == 1]
    reference_mean = reference.mean(axis=0)

    from sklearn.decomposition import PCA

    reference_pca = PCA(svd_solver="full").fit(reference - reference_mean)
    reference_count = hp.select_component_count(reference_pca.explained_variance_ratio_, hp.NULL_VARIANCE)
    reference_basis = reference_pca.components_[:reference_count]

    centered_target = target - reference_mean
    target_residual = centered_target - (centered_target @ reference_basis.T @ reference_basis)

    exclusive_pca = PCA(svd_solver="full").fit(target_residual)
    exclusive_count = hp.select_component_count(exclusive_pca.explained_variance_ratio_, hp.EXCLUSIVE_VARIANCE, hp.MAX_EXCLUSIVE_COMPONENTS)

    # Scree
    fig = plot_scree(reference_pca.explained_variance_ratio_, "Stage1 Reference (Rest) cumulative variance")
    save_fig(fig, "stage1_rest_scree")

    # Top exclusive component channel weights
    comp = exclusive_pca.components_[0]
    weights = channel_weights_from_component(comp, n_channels, n_bands, n_time)
    fig = plot_channel_bar(weights, hp.MOTOR_CHANNELS, "Stage1 Exclusive Component 1 (channel RMS)")
    save_fig(fig, "stage1_exclusive_comp1_channels")

    # Scatter of first two coordinates colored by true class (Left/Right/Rest)
    classes = {hp.LEFT_LABEL: "Left", hp.RIGHT_LABEL: "Right", hp.REST_LABEL: "Rest"}
    fig = plot_scatter(coords, stage1_labels, {0: "Rest", 1: "MI"}, "Stage1 coords (Rest vs MI)")
    save_fig(fig, "stage1_coords_rest_vs_mi")

    # Score histogram if classifier supports predict_proba
    clf = stage1["classifier"]
    if hasattr(clf, "predict_proba"):
        scores = clf.predict_proba(coords)[:, 1]
        fig = plot_score_hist(scores, stage1_labels, "Stage1 positive scores (Rest vs MI)")
        save_fig(fig, "stage1_scores_hist")

    # Stage2: Left vs Right inside MI mask
    mi_mask = labels != hp.REST_LABEL
    features_mi = features[mi_mask]
    labels_mi = (labels[mi_mask] == hp.RIGHT_LABEL).astype(int)

    fs2 = stage2["feature_scaler"]
    proj2 = stage2["projection_matrix"]
    offset2 = stage2["projection_offset"]
    coord_scaler2 = stage2["coordinate_scaler"]

    scaled2 = fs2.transform(features_mi)
    projected2 = scaled2 @ proj2 + offset2
    coords2 = coord_scaler2.transform(projected2)

    # For each exclusive pair used in stage2, replicate PCA to show their scree and first component
    # The stage2 in this code was trained with pairs [(1,0),(0,1)] meaning reference=Right,target=Left and vice versa
    # First pair
    reference = scaled2[labels_mi == 1]
    target = scaled2[labels_mi == 0]
    if reference.size and target.size:
        ref_mean = reference.mean(axis=0)
        ref_pca = PCA(svd_solver="full").fit(reference - ref_mean)
        fig = plot_scree(ref_pca.explained_variance_ratio_, "Stage2 Pair1 Reference (Right) cumulative variance")
        save_fig(fig, "stage2_pair1_ref_scree")

        # exclusive comp
        ref_count = hp.select_component_count(ref_pca.explained_variance_ratio_, hp.NULL_VARIANCE)
        ref_basis = ref_pca.components_[:ref_count]
        centered_t = target - ref_mean
        targ_res = centered_t - (centered_t @ ref_basis.T @ ref_basis)
        ex_pca = PCA(svd_solver="full").fit(targ_res)
        comp = ex_pca.components_[0]
        weights = channel_weights_from_component(comp, n_channels, n_bands, n_time)
        fig = plot_channel_bar(weights, hp.MOTOR_CHANNELS, "Stage2 Pair1 Exclusive Comp 1 (channel RMS)")
        save_fig(fig, "stage2_pair1_comp1_channels")

    # Scatter of first two coords for Left vs Right
    fig = plot_scatter(coords2, labels_mi, {0: "Left", 1: "Right"}, "Stage2 coords (Left vs Right)")
    save_fig(fig, "stage2_coords_left_vs_right")

    clf2 = stage2["classifier"]
    if hasattr(clf2, "predict_proba"):
        scores2 = clf2.predict_proba(coords2)[:, 1]
        fig = plot_score_hist(scores2, labels_mi, "Stage2 positive scores (Left vs Right)")
        save_fig(fig, "stage2_scores_hist")


def main():
    # replicate data loading from hierarchical_PCA.main
    mne_raws = [
        hp.mne.io.read_raw_eeglab(hp.EEG_TEMPLATE.format(run), preload=True)
        for run in hp.RUNS
    ]
    raw_all = hp.mne.concatenate_raws(mne_raws)

    # handle boundary annotations same as original
    annotations = raw_all.annotations.copy()
    rename_map = {
        description: "BAD_boundary"
        for description in set(annotations.description)
        if str(description).lower() in ("boundary", "edge boundary")
    }
    if rename_map:
        raw_all.set_annotations(annotations.rename(rename_map))

    events, event_id = hp.mne.events_from_annotations(raw_all)
    cue_code = event_id["S 15"]
    cue_events = events[events[:, 2] == cue_code]
    epochs = hp.mne.Epochs(
        raw_all,
        cue_events,
        event_id={"S 15": cue_code},
        tmin=-2.0,
        tmax=5.0,
        baseline=None,
        preload=True,
        reject_by_annotation=True,
    ).pick_channels(hp.MOTOR_CHANNELS, ordered=True)

    cue_labels = hp.loadmat(hp.LABEL_FILE)["labels"].squeeze().astype(int)
    labels = np.choose(cue_labels[epochs.selection] - 1, [hp.LEFT_LABEL, hp.RIGHT_LABEL, hp.REST_LABEL])

    sfreq = float(epochs.info["sfreq"])
    filter_b = hp.design_filters(sfreq)

    # Build windows efficiently: precompute spectral envelopes per trial to avoid repeated filtering.
    data = epochs.get_data()
    baseline_start = int(round((hp.BASELINE_START_SECONDS - epochs.tmin) * sfreq))
    baseline_stop = int(round((hp.BASELINE_STOP_SECONDS - epochs.tmin) * sfreq))
    task_start = int(round((hp.TASK_START_SECONDS - epochs.tmin) * sfreq))
    task_stop = int(round((hp.TASK_STOP_SECONDS - epochs.tmin) * sfreq))
    window_samples = int(round(hp.WINDOW_SECONDS * sfreq))
    step_samples = int(round(hp.WINDOW_STEP_SECONDS * sfreq))

    smooth_samples = int(round(hp.SMOOTH_SECONDS * sfreq))
    features, window_labels, groups = [], [], []
    for trial_index, (trial_data, label) in enumerate(zip(data, labels)):
        # trial_data: channels x samples
        env = hp.spectral_envelope(trial_data, filter_b, smooth_samples)
        baseline_env = env[:, :, baseline_start:baseline_stop]
        baseline_mean = baseline_env.mean(axis=-1, keepdims=True)
        baseline_std = baseline_env.std(axis=-1, keepdims=True) + 1e-12
        for start in range(task_start, task_stop - window_samples + 1, step_samples):
            task_env = env[:, :, start:start + window_samples]
            task_z = (task_env - baseline_mean) / baseline_std
            features.append(task_z.reshape(-1))
            window_labels.append(label)
            groups.append(trial_index)

    features = np.asarray(features)
    window_labels = np.asarray(window_labels)
    groups = np.asarray(groups)

    print("Feature matrix:", features.shape)

    stage1, stage2 = hp.fit_hierarchical(features, window_labels)

    visualize(features, window_labels, sfreq, stage1, stage2)


if __name__ == "__main__":
    main()