# %%
import numpy as np
from joblib import Parallel, delayed
import mne
from scipy.io import loadmat
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit, cross_val_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.base import clone
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score,balanced_accuracy_score,f1_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model  import LogisticRegression
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.feature_selection import SelectKBest, mutual_info_classif,f_classif
import warnings
from FBCSP import FBCSP
from joblib import Parallel
import time
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
import os
import warnings
from collections import Counter
from sklearn.base import BaseEstimator, TransformerMixin

class OvRCSP(BaseEstimator, TransformerMixin):
    def __init__(self, sfreq, fmin, fmax, n_classes=3, n_components=2,
                 reg="ledoit_wolf", log=True, norm_trace=False, verbose=False):
        self.sfreq = sfreq
        self.fmin = fmin
        self.fmax = fmax
        self.n_classes = n_classes
        self.n_components = n_components
        self.reg = reg
        self.log = log
        self.norm_trace = norm_trace
        self.verbose = verbose
        self._csps = None

    def fit(self, X, y):
        Xf = mne.filter.filter_data(
            X, self.sfreq, self.fmin, self.fmax, verbose=self.verbose
        )

        self._csps = []
        for k in range(self.n_classes):
            y_bin = (y == k).astype(int)
            csp = CSP(
                n_components=self.n_components,
                reg=self.reg,
                log=self.log,
                norm_trace=self.norm_trace
            )
            csp.fit(Xf, y_bin)
            self._csps.append(csp)
        return self

    def transform(self, X):
        Xf = mne.filter.filter_data(
            X, self.sfreq, self.fmin, self.fmax, verbose=self.verbose
        )

        feats = []
        for csp in self._csps:
            feats.append(csp.transform(Xf))
        return np.concatenate(feats, axis=1)

print(os.cpu_count())

start = time.perf_counter()

warnings.filterwarnings("ignore")
mne.set_log_level("ERROR")

# %%

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

# %%
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

cv = StratifiedShuffleSplit(n_splits=20, test_size=0.2, random_state=1)

classifiers = {
    "LDA": LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
    "SVM": SVC(kernel="linear", class_weight="balanced"),
    "KNN": KNeighborsClassifier(),
    "LogReg": LogisticRegression(max_iter=3000, class_weight="balanced")
}

# %%
def run_one_fold(tr, te, X, y, fbcsp, classifiers, scale=True):
    mne.set_log_level("CRITICAL")

    fb = clone(fbcsp)
    Xtr = fb.fit_transform(X[tr], y[tr])
    Xte = fb.transform(X[te])

    if scale:
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(Xtr)
        Xte = scaler.transform(Xte)

    fold_results = {}

    for name, clf in classifiers.items():
        model = clone(clf)
        model.fit(Xtr, y[tr])
        yhat = model.predict(Xte)

        fold_results[name] = {
            "acc": accuracy_score(y[te], yhat),
            "bacc": balanced_accuracy_score(y[te], yhat),
            "f1": f1_score(y[te], yhat, average="macro"),
            "cm": confusion_matrix(y[te], yhat,labels=[0,1], normalize="true")
        }

    return fold_results


def summary_results(results, classifiers):
    summary = {}
    for name in classifiers.keys():
        accs  = np.array([fold[name]["acc"]  for fold in results], dtype=float)
        baccs = np.array([fold[name]["bacc"] for fold in results], dtype=float)
        f1s   = np.array([fold[name]["f1"]   for fold in results], dtype=float)
        cms   = [fold[name]["cm"] for fold in results]

        summary[name] = {
            "acc_mean": float(accs.mean()),
            "acc_std": float(accs.std()),
            "acc_folds": accs,
            "bacc_mean": float(baccs.mean()),
            "bacc_std": float(baccs.std()),
            "bacc_folds": baccs,
            "f1_mean": float(f1s.mean()),
            "f1_std": float(f1s.std()),
            "f1_folds": f1s,
            "cms": cms
        }
    return summary


def run_experiment(exp_name, X, y, n_classes, classifiers, bands, sfreq, cv, scale=True):
    print(f"\n{'='*20} {exp_name} {'='*20}")
    print("X shape:", X.shape)
    print("class counts:", Counter(y))

    fbcsp = FBCSP(
        sfreq=sfreq,
        bands=bands,
        n_classes=n_classes,
        n_components=4,
        reg="ledoit_wolf",
        log=True,
        norm_trace=False
    )

    splits = list(cv.split(X, y))

    results = Parallel(n_jobs=-1, prefer="processes")(
        delayed(run_one_fold)(tr, te, X, y, fbcsp, classifiers, scale)
        for tr, te in splits
    )

    summary = summary_results(results, classifiers)

    chance = 1 / n_classes
    print(f"Chance level = {chance:.3f}")

    for name, s in summary.items():
        print(
            f"{name:10s} "
            f"acc={s['acc_mean']:.3f}±{s['acc_std']:.3f}  "
            f"bacc={s['bacc_mean']:.3f}±{s['bacc_std']:.3f}  "
            f"macroF1={s['f1_mean']:.3f}±{s['f1_std']:.3f}  "
            f"folds={np.round(s['acc_folds'], 3)}"
        )

    return summary

def run_direct_3class(X, y, bands, sfreq, cv, scale=True):
    print(f"\n{'='*20} Direct 3-class {'='*20}")
    print("X shape:", X.shape)
    print("class counts:", Counter(y))

    fbcsp = FBCSP(
        sfreq=sfreq,
        bands=bands,
        n_classes=3,
        n_components=4,
        reg="ledoit_wolf",
        log=True,
        norm_trace=False
    )

    clf = clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    splits = list(cv.split(X, y))

    accs, baccs, f1s, cms = [], [], [], []

    for fold_i, (tr, te) in enumerate(splits, 1):
        fb = clone(fbcsp)
        Xtr = fb.fit_transform(X[tr], y[tr])
        Xte = fb.transform(X[te])

        if scale:
            scaler = StandardScaler()
            Xtr = scaler.fit_transform(Xtr)
            Xte = scaler.transform(Xte)
        k = min(10, Xtr.shape[1])
        # selector = SelectKBest(f_classif)
        # Xtr = selector.fit_transform(Xtr, y[tr])
        # Xte = selector.transform(Xte)

        model = clone(clf)
        model.fit(Xtr, y[tr])
        yhat = model.predict(Xte)

        accs.append(accuracy_score(y[te], yhat))
        baccs.append(balanced_accuracy_score(y[te], yhat))
        f1s.append(f1_score(y[te], yhat, average="macro"))
        cms.append(confusion_matrix(y[te], yhat, labels=[0, 1, 2], normalize="true"))

    accs = np.array(accs)
    baccs = np.array(baccs)
    f1s = np.array(f1s)

    print("Chance level = 0.333")
    print(
        f"KNN        "
        f"acc={accs.mean():.3f}±{accs.std():.3f}  "
        f"bacc={baccs.mean():.3f}±{baccs.std():.3f}  "
        f"macroF1={f1s.mean():.3f}±{f1s.std():.3f}  "
        f"folds={np.round(accs, 3)}"
    )

    cm_mean = np.mean(np.stack(cms, axis=0), axis=0)
    print("Mean confusion matrix (rows=true, cols=pred):")
    print(np.round(cm_mean, 2))

    return {
        "acc_mean": float(accs.mean()),
        "acc_std": float(accs.std()),
        "bacc_mean": float(baccs.mean()),
        "bacc_std": float(baccs.std()),
        "f1_mean": float(f1s.mean()),
        "f1_std": float(f1s.std()),
        "acc_folds": accs,
        "cms": cms,
    }

def run_pairwise_voting_3class(X, y, bands, sfreq, cv, scale=True):
    print(f"\n{'='*20} Pairwise Voting 3-class {'='*20}")
    print("X shape:", X.shape)
    print("class counts:", Counter(y))

    clf = KNeighborsClassifier()
    splits = list(cv.split(X, y))

    accs, baccs, f1s, cms = [], [], [], []

    for fold_i, (tr, te) in enumerate(splits, 1):
        X_train, X_test = X[tr], X[te]
        y_train, y_test = y[tr], y[te]

        # ----- 1) Left vs Right -----
        mask_lr_tr = y_train != 2
        X_lr_tr = X_train[mask_lr_tr]
        y_lr_tr = y_train[mask_lr_tr]   # 0 or 1

        fbcsp_lr = FBCSP(
            sfreq=sfreq,
            bands=bands,
            n_classes=2,
            n_components=4,
            reg="ledoit_wolf",
            log=True,
            norm_trace=False
        )

        Xtr_lr = fbcsp_lr.fit_transform(X_lr_tr, y_lr_tr)
        Xte_lr = fbcsp_lr.transform(X_test)

        if scale:
            scaler_lr = StandardScaler()
            Xtr_lr = scaler_lr.fit_transform(Xtr_lr)
            Xte_lr = scaler_lr.transform(Xte_lr)

        model_lr = clone(clf)
        model_lr.fit(Xtr_lr, y_lr_tr)
        pred_lr = model_lr.predict(Xte_lr)   # 0=Left, 1=Right

        # ----- 2) Left vs Blank -----
        mask_lb_tr = y_train != 1
        X_lb_tr = X_train[mask_lb_tr]
        y_lb_tr_raw = y_train[mask_lb_tr]    # 0 or 2
        y_lb_tr = (y_lb_tr_raw == 0).astype(int)   # Left=1, Blank=0

        fbcsp_lb = FBCSP(
            sfreq=sfreq,
            bands=bands,
            n_classes=2,
            n_components=4,
            reg="ledoit_wolf",
            log=True,
            norm_trace=False
        )

        Xtr_lb = fbcsp_lb.fit_transform(X_lb_tr, y_lb_tr)
        Xte_lb = fbcsp_lb.transform(X_test)

        if scale:
            scaler_lb = StandardScaler()
            Xtr_lb = scaler_lb.fit_transform(Xtr_lb)
            Xte_lb = scaler_lb.transform(Xte_lb)

        model_lb = clone(clf)
        model_lb.fit(Xtr_lb, y_lb_tr)
        pred_lb_bin = model_lb.predict(Xte_lb)   # 1=Left, 0=Blank
        pred_lb = np.where(pred_lb_bin == 1, 0, 2)

        # ----- 3) Right vs Blank -----
        mask_rb_tr = y_train != 0
        X_rb_tr = X_train[mask_rb_tr]
        y_rb_tr_raw = y_train[mask_rb_tr]    # 1 or 2
        y_rb_tr = (y_rb_tr_raw == 1).astype(int)   # Right=1, Blank=0

        fbcsp_rb = FBCSP(
            sfreq=sfreq,
            bands=bands,
            n_classes=2,
            n_components=4,
            reg="ledoit_wolf",
            log=True,
            norm_trace=False
        )

        Xtr_rb = fbcsp_rb.fit_transform(X_rb_tr, y_rb_tr)
        Xte_rb = fbcsp_rb.transform(X_test)

        if scale:
            scaler_rb = StandardScaler()
            Xtr_rb = scaler_rb.fit_transform(Xtr_rb)
            Xte_rb = scaler_rb.transform(Xte_rb)

        model_rb = clone(clf)
        model_rb.fit(Xtr_rb, y_rb_tr)
        pred_rb_bin = model_rb.predict(Xte_rb)   # 1=Right, 0=Blank
        pred_rb = np.where(pred_rb_bin == 1, 1, 2)

        # ----- Voting -----
        yhat = []
        for i in range(len(y_test)):
            votes = [pred_lr[i], pred_lb[i], pred_rb[i]]
            counts = np.bincount(votes, minlength=3)
            yhat.append(np.argmax(counts))
        yhat = np.array(yhat)

        accs.append(accuracy_score(y_test, yhat))
        baccs.append(balanced_accuracy_score(y_test, yhat))
        f1s.append(f1_score(y_test, yhat, average="macro"))
        cms.append(confusion_matrix(y_test, yhat, labels=[0, 1, 2], normalize="true"))

    accs = np.array(accs)
    baccs = np.array(baccs)
    f1s = np.array(f1s)

    print("Chance level = 0.333")
    print(
        f"LDA-vote   "
        f"acc={accs.mean():.3f}±{accs.std():.3f}  "
        f"bacc={baccs.mean():.3f}±{baccs.std():.3f}  "
        f"macroF1={f1s.mean():.3f}±{f1s.std():.3f}  "
        f"folds={np.round(accs, 3)}"
    )

    cm_mean = np.mean(np.stack(cms, axis=0), axis=0)
    print("Mean confusion matrix (rows=true, cols=pred):")
    print(np.round(cm_mean, 2))

    return {
        "acc_mean": float(accs.mean()),
        "acc_std": float(accs.std()),
        "bacc_mean": float(baccs.mean()),
        "bacc_std": float(baccs.std()),
        "f1_mean": float(f1s.mean()),
        "f1_std": float(f1s.std()),
        "acc_folds": accs,
        "cms": cms,
    }

def run_hierarchical_3class(X, y, bands, sfreq, cv, scale=True):
    print(f"\n{'='*20} Hierarchical 3-class (MI/Blank -> L/R) {'='*20}")
    print("X shape:", X.shape)
    print("class counts:", Counter(y))

    clf_stage1 = KNeighborsClassifier()
    clf_stage2 = KNeighborsClassifier()

    splits = list(cv.split(X, y))

    accs, baccs, f1s, cms = [], [], [], []

    for fold_i, (tr, te) in enumerate(splits, 1):
        X_train, X_test = X[tr], X[te]
        y_train, y_test = y[tr], y[te]

        # Stage 1: MI vs Blank
        y_train_mi_blank = np.where(y_train == 2, 0, 1)

        fbcsp_stage1 = FBCSP(
            sfreq=sfreq,
            bands=bands,
            n_classes=2,
            n_components=4,
            reg="ledoit_wolf",
            log=True,
            norm_trace=False
        )

        Xtr_s1 = fbcsp_stage1.fit_transform(X_train, y_train_mi_blank)
        Xte_s1 = fbcsp_stage1.transform(X_test)

        if scale:
            scaler_s1 = StandardScaler()
            Xtr_s1 = scaler_s1.fit_transform(Xtr_s1)
            Xte_s1 = scaler_s1.transform(Xte_s1)

        model_s1 = clone(clf_stage1)
        model_s1.fit(Xtr_s1, y_train_mi_blank)
        pred_stage1 = model_s1.predict(Xte_s1)   # 1=MI, 0=Blank

        # Stage 2: Left vs Right
        mask_lr_tr = y_train != 2
        X_lr_tr = X_train[mask_lr_tr]
        y_lr_tr = y_train[mask_lr_tr]   # 0=Left, 1=Right

        fbcsp_stage2 = FBCSP(
            sfreq=sfreq,
            bands=bands,
            n_classes=2,
            n_components=4,
            reg="ledoit_wolf",
            log=True,
            norm_trace=False
        )

        Xtr_s2 = fbcsp_stage2.fit_transform(X_lr_tr, y_lr_tr)
        Xte_s2 = fbcsp_stage2.transform(X_test)

        if scale:
            scaler_s2 = StandardScaler()
            Xtr_s2 = scaler_s2.fit_transform(Xtr_s2)
            Xte_s2 = scaler_s2.transform(Xte_s2)

        model_s2 = clone(clf_stage2)
        model_s2.fit(Xtr_s2, y_lr_tr)
        pred_stage2 = model_s2.predict(Xte_s2)   # 0=Left, 1=Right

        # Final decision
        yhat = np.where(pred_stage1 == 0, 2, pred_stage2)

        accs.append(accuracy_score(y_test, yhat))
        baccs.append(balanced_accuracy_score(y_test, yhat))
        f1s.append(f1_score(y_test, yhat, average="macro"))
        cms.append(confusion_matrix(y_test, yhat, labels=[0, 1, 2], normalize="true"))

    accs = np.array(accs)
    baccs = np.array(baccs)
    f1s = np.array(f1s)

    print("Chance level = 0.333")
    print(
        f"KNN-hier   "
        f"acc={accs.mean():.3f}±{accs.std():.3f}  "
        f"bacc={baccs.mean():.3f}±{baccs.std():.3f}  "
        f"macroF1={f1s.mean():.3f}±{f1s.std():.3f}  "
        f"folds={np.round(accs, 3)}"
    )

    cm_mean = np.mean(np.stack(cms, axis=0), axis=0)
    print("Mean confusion matrix (rows=true, cols=pred):")
    print(np.round(cm_mean, 2))

    return {
        "acc_mean": float(accs.mean()),
        "acc_std": float(accs.std()),
        "bacc_mean": float(baccs.mean()),
        "bacc_std": float(baccs.std()),
        "f1_mean": float(f1s.mean()),
        "f1_std": float(f1s.std()),
        "acc_folds": accs,
        "cms": cms,
    }



def run_csp_binary(exp_name, X, y, cv, fmin=8, fmax=40, n_components=4, scale=True):
    print(f"\n{'='*20} {exp_name} | CSP {fmin}-{fmax} Hz {'='*20}")
    print("X shape:", X.shape)
    print("class counts:", Counter(y))

    clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    splits = list(cv.split(X, y))

    accs, baccs, f1s, cms = [], [], [], []

    for fold_i, (tr, te) in enumerate(splits, 1):
        X_train = X[tr].copy()
        X_test = X[te].copy()
        y_train = y[tr]
        y_test = y[te]

        # band-pass inside each fold
        X_train_f = mne.filter.filter_data(X_train, sfreq, fmin, fmax, verbose=False)
        X_test_f = mne.filter.filter_data(X_test, sfreq, fmin, fmax, verbose=False)

        csp = CSP(
            n_components=n_components,
            reg="ledoit_wolf",
            log=True,
            norm_trace=False
        )

        Xtr_csp = csp.fit_transform(X_train_f, y_train)
        Xte_csp = csp.transform(X_test_f)

        if scale:
            scaler = StandardScaler()
            Xtr_csp = scaler.fit_transform(Xtr_csp)
            Xte_csp = scaler.transform(Xte_csp)

        model = clone(clf)
        model.fit(Xtr_csp, y_train)
        yhat = model.predict(Xte_csp)

        accs.append(accuracy_score(y_test, yhat))
        baccs.append(balanced_accuracy_score(y_test, yhat))
        f1s.append(f1_score(y_test, yhat, average="macro"))
        cms.append(confusion_matrix(y_test, yhat, labels=[0,1], normalize="true"))

    accs = np.array(accs)
    baccs = np.array(baccs)
    f1s = np.array(f1s)

    print("Chance level = 0.500")
    print(
        f"CSP+LDA    "
        f"acc={accs.mean():.3f}±{accs.std():.3f}  "
        f"bacc={baccs.mean():.3f}±{baccs.std():.3f}  "
        f"macroF1={f1s.mean():.3f}±{f1s.std():.3f}  "
        f"folds={np.round(accs, 3)}"
    )

    cm_mean = np.mean(np.stack(cms, axis=0), axis=0)
    print("Mean confusion matrix (rows=true, cols=pred):")
    print(np.round(cm_mean, 2))

    return {
        "acc_mean": float(accs.mean()),
        "acc_std": float(accs.std()),
        "bacc_mean": float(baccs.mean()),
        "bacc_std": float(baccs.std()),
        "f1_mean": float(f1s.mean()),
        "f1_std": float(f1s.std()),
        "acc_folds": accs,
        "cms": cms,
    }

def run_ovr_csp_3class(X, y, sfreq, cv, info,
                       fmin=23, fmax=35,
                       n_components=2, scale=True,
                       classifier_name="LDA",
                       plot_patterns=True):
    print(f"\n{'='*20} OvR CSP 3-class ({fmin}-{fmax} Hz) {'='*20}")
    print("X shape:", X.shape)
    print("class counts:", Counter(y))

    if classifier_name == "LDA":
        clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    elif classifier_name == "KNN":
        clf = KNeighborsClassifier()
    elif classifier_name == "LogReg":
        clf = LogisticRegression(max_iter=3000, class_weight="balanced")
    else:
        raise ValueError("classifier_name must be one of: LDA, KNN, LogReg")

    splits = list(cv.split(X, y))

    accs, baccs, f1s, cms = [], [], [], []

    for fold_i, (tr, te) in enumerate(splits, 1):
        X_train, X_test = X[tr], X[te]
        y_train, y_test = y[tr], y[te]

        ovr_csp = OvRCSP(
            sfreq=sfreq,
            fmin=fmin,
            fmax=fmax,
            n_classes=3,
            n_components=n_components,
            reg="ledoit_wolf",
            log=True,
            norm_trace=False,
            verbose=False
        )

        Xtr_feat = ovr_csp.fit_transform(X_train, y_train)
        Xte_feat = ovr_csp.transform(X_test)

        if scale:
            scaler = StandardScaler()
            Xtr_feat = scaler.fit_transform(Xtr_feat)
            Xte_feat = scaler.transform(Xte_feat)

        model = clone(clf)
        model.fit(Xtr_feat, y_train)
        yhat = model.predict(Xte_feat)

        accs.append(accuracy_score(y_test, yhat))
        baccs.append(balanced_accuracy_score(y_test, yhat))
        f1s.append(f1_score(y_test, yhat, average="macro"))
        cms.append(confusion_matrix(y_test, yhat, labels=[0, 1, 2], normalize="true"))

    accs = np.array(accs)
    baccs = np.array(baccs)
    f1s = np.array(f1s)

    print("Chance level = 0.333")
    print(
        f"{classifier_name:10s} "
        f"acc={accs.mean():.3f}±{accs.std():.3f}  "
        f"bacc={baccs.mean():.3f}±{baccs.std():.3f}  "
        f"macroF1={f1s.mean():.3f}±{f1s.std():.3f}  "
        f"folds={np.round(accs, 3)}"
    )

    cm_mean = np.mean(np.stack(cms, axis=0), axis=0)
    print("Mean confusion matrix (rows=true, cols=pred):")
    print(np.round(cm_mean, 2))

    # ---- fit on full data once for plotting patterns ----
    if plot_patterns:
        ovr_csp_full = OvRCSP(
            sfreq=sfreq,
            fmin=fmin,
            fmax=fmax,
            n_classes=3,
            n_components=n_components,
            reg="ledoit_wolf",
            log=True,
            norm_trace=False,
            verbose=False
        )
        ovr_csp_full.fit(X, y)

        class_names = {0: "Left", 1: "Right", 2: "Blank"}
        for k, csp in enumerate(ovr_csp_full._csps):
            fig = csp.plot_patterns(
                info,
                components=list(range(n_components)),
                ch_type="eeg",
                show=False,
                size=1.5,
                sphere=(0., 0., 0., 0.13)
            )
            fig.suptitle(f"OvR CSP patterns: {class_names[k]} vs Rest ({fmin}-{fmax} Hz)")
            plt.show()

    return {
        "acc_mean": float(accs.mean()),
        "acc_std": float(accs.std()),
        "bacc_mean": float(baccs.mean()),
        "bacc_std": float(baccs.std()),
        "f1_mean": float(f1s.mean()),
        "f1_std": float(f1s.std()),
        "acc_folds": accs,
        "cms": cms,
    }


# %%
def ar_features_1d(signal, order=6):

    x = np.asarray(signal, dtype=float)

    # remove mean to reduce intercept effect
    x = x - np.mean(x)

    n = len(x)
    if n <= order:
        return np.zeros(order, dtype=float)

    # Build regression matrix
    # target: x[order:]
    # predictors: [x[t-1], x[t-2], ..., x[t-order]]
    Y = x[order:]
    X = np.column_stack([x[order-k-1:n-k-1] for k in range(order)])

    try:
        coeffs, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
        return coeffs
    except Exception:
        return np.zeros(order, dtype=float)


def extract_ar_filterbank_features(epochs, picks, bands, order=6):

    features_all = []

    for fmin, fmax in bands:

        ep = epochs.copy().pick(picks).crop(0.5,3.0)
        ep = ep.filter(fmin, fmax, verbose=False)

        data = ep.get_data()
        data = data[:,:,::10]   # decimate

        n_trials, n_ch, _ = data.shape
        X_ar = []

        for i in range(n_trials):

            feat_trial = []

            for ch in range(n_ch):

                sig = data[i,ch,:]
                coeffs = ar_features_1d(sig, order)

                feat_trial.extend(coeffs)

            X_ar.append(feat_trial)

        features_all.append(np.array(X_ar))

    X = np.concatenate(features_all, axis=1)

    return X

def run_fbar_classifier(
    exp_name,
    epochs_a,
    epochs_b=None,
    epochs_c=None,
    bands=None,
    order=6,
    picks=None,
    cv=None,
    classifier_name="LDA",
    scale=True
):
    if picks is None:
        raise ValueError("Please provide picks (channel list).")
    if cv is None:
        raise ValueError("Please provide cv.")
    if bands is None:
        raise ValueError("Please provide bands.")
    if epochs_b is None:
        raise ValueError("epochs_b cannot be None.")

    X_a = extract_ar_filterbank_features(epochs_a, picks=picks, bands=bands, order=order)
    X_b = extract_ar_filterbank_features(epochs_b, picks=picks, bands=bands, order=order)

    if epochs_c is None:
        X = np.vstack([X_a, X_b])
        y = np.r_[np.zeros(len(X_a), dtype=int),
                  np.ones(len(X_b), dtype=int)]
        labels = [0, 1]
        chance = 0.5
    else:
        X_c = extract_ar_filterbank_features(epochs_c, picks=picks, bands=bands, order=order)
        X = np.vstack([X_a, X_b, X_c])
        y = np.r_[np.zeros(len(X_a), dtype=int),
                  np.ones(len(X_b), dtype=int),
                  2 * np.ones(len(X_c), dtype=int)]
        labels = [0, 1, 2]
        chance = 1 / 3

    print(f"\n{'='*20} {exp_name} | FBAR(order={order}) {'='*20}")
    print("Bands:", bands)
    print("X shape:", X.shape)
    print("class counts:", Counter(y))

    if classifier_name == "LDA":
        clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    elif classifier_name == "LogReg":
        clf = LogisticRegression(max_iter=3000, class_weight="balanced")
    elif classifier_name == "SVM":
        clf = SVC(kernel="linear", class_weight="balanced")
    else:
        raise ValueError("classifier_name must be one of: LDA, LogReg, SVM")

    splits = list(cv.split(X, y))

    accs, baccs, f1s, cms = [], [], [], []

    for tr, te in splits:
        Xtr = X[tr].copy()
        Xte = X[te].copy()
        ytr = y[tr]
        yte = y[te]

        if scale:
            scaler = StandardScaler()
            Xtr = scaler.fit_transform(Xtr)
            Xte = scaler.transform(Xte)

        model = clone(clf)
        model.fit(Xtr, ytr)
        yhat = model.predict(Xte)

        accs.append(accuracy_score(yte, yhat))
        baccs.append(balanced_accuracy_score(yte, yhat))
        f1s.append(f1_score(yte, yhat, average="macro"))
        cms.append(confusion_matrix(yte, yhat, labels=labels, normalize="true"))

    accs = np.array(accs)
    baccs = np.array(baccs)
    f1s = np.array(f1s)

    print(f"Chance level = {chance:.3f}")
    print(
        f"{classifier_name:10s} "
        f"acc={accs.mean():.3f}±{accs.std():.3f}  "
        f"bacc={baccs.mean():.3f}±{baccs.std():.3f}  "
        f"macroF1={f1s.mean():.3f}±{f1s.std():.3f}  "
        f"folds={np.round(accs, 3)}"
    )

    cm_mean = np.mean(np.stack(cms, axis=0), axis=0)
    print("Mean confusion matrix (rows=true, cols=pred):")
    print(np.round(cm_mean, 2))

    return {
        "acc_mean": float(accs.mean()),
        "acc_std": float(accs.std()),
        "bacc_mean": float(baccs.mean()),
        "bacc_std": float(baccs.std()),
        "f1_mean": float(f1s.mean()),
        "f1_std": float(f1s.std()),
        "acc_folds": accs,
        "cms": cms,
        "X_shape": X.shape
    }


# %%
# ===== Experiment 1: Right vs Blank =====
# Keep Right(1) and Blank(2)
mask_rb = y_3 != 0
X_rb = X_use[mask_rb]
y_rb_raw = y_3[mask_rb]
y_rb = (y_rb_raw == 1).astype(int)   # Right=1, Blank=0

summary_rb = run_experiment(
    exp_name="Right vs Blank",
    X=X_rb,
    y=y_rb,
    n_classes=2,
    classifiers=classifiers,
    bands=bands,
    sfreq=sfreq,
    cv=cv,
    scale=True
)
 # %%
summary_rb_csp = run_csp_binary(
    exp_name="Right vs Blank",
    X=X_rb,
    y=y_rb,
    cv=cv,
    fmin=23,
    fmax=35,
    n_components=4,
    scale=True
)

# %%
# ===== Experiment 2: MI vs Blank =====
# Left + Right -> MI(1), Blank -> 0
y_mi_blank = np.where(y_3 == 2, 0, 1)

summary_mi_blank = run_experiment(
    exp_name="MI vs Blank",
    X=X_use,
    y=y_mi_blank,
    n_classes=2,
    classifiers=classifiers,
    bands=bands,
    sfreq=sfreq,
    cv=cv,
    scale=True
)

# %%
# ===== Experiment 3: Left vs Right =====
# Keep Left(0) and Right(1)
mask_lr = y_3 != 2
X_lr = X_use[mask_lr]
y_lr = y_3[mask_lr]   # Left=0, Right=1

summary_lr = run_experiment(
    exp_name="Left vs Right",
    X=X_lr,
    y=y_lr,
    n_classes=2,
    classifiers=classifiers,
    bands=bands,
    sfreq=sfreq,
    cv=cv,
    scale=True
)

# %%
# ===== Experiment 4: Shuffled-label sanity check =====
# Use the same Right vs Blank task, but shuffle labels
rng = np.random.RandomState(0)
y_rb_shuf = rng.permutation(y_rb)

summary_rb_shuf = run_experiment(
    exp_name="Right vs Blank (SHUFFLED LABELS)",
    X=X_rb,
    y=y_rb_shuf,
    n_classes=2,
    classifiers=classifiers,
    bands=bands,
    sfreq=sfreq,
    cv=cv,
    scale=True
)

# %%
# ===== Experiment 5: Left vs Blank =====
# Keep Left(0) and Blank(2)
mask_lb = y_3 != 1
X_lb = X_use[mask_lb]