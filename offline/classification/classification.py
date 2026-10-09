# %%
import numpy as np
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
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.feature_selection import SelectKBest, mutual_info_classif,f_classif
import warnings
from FBCSP import FBCSP
from joblib import Parallel, delayed
import time
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
import os
import warnings
from collections import Counter

print(os.cpu_count())

start = time.perf_counter()

warnings.filterwarnings("ignore")
mne.set_log_level("ERROR")

# %%
bands = [(8,12), (12,16), (16,20), (20,24), (24,28), (28,32),(32,40)]

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
epochs = mne.Epochs(raw_all,events, event_id=event_id_use, tmin=-2.5,tmax=5.0,baseline=None,preload=True,reject_by_annotation=True)
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
epochs_3 = mne.concatenate_epochs([epochs_left, epochs_right, epochs_blank])
y_3 = np.r_[np.zeros(len(epochs_left),dtype = int),np.ones(len(epochs_right),dtype=int),2*np.ones(len(epochs_blank),dtype=int)]
# epochs_3 = epochs_3.apply_baseline((-2.5,0))
# epochs_3 = epochs_3.filter(25,35)
epochs_3 = epochs_3.crop(tmin=0.,tmax=4.5)

roi_chs = ['Fp1', 'Fz', 'F3', 'F7', 'FT9', 'FC5', 'FC1', 'C3', 'T7', 'TP9', 'CP5', 'CP1', 'Pz', 'TP10', 'CP6', 'CP2', 'Cz', 'C4', 'T8', 'FT10', 'FC6', 'FC2', 'F4', 'F8', 'Fp2', 'AF7', 'AF3', 'AFz', 'F1', 'F5', 'FT7', 'FC3', 'C1', 'C5', 'TP7', 'CP3', 'P1', 'P5', 'P6', 'P2', 'CPz', 'CP4', 'TP8', 'C6', 'C2', 'FC4', 'FT8', 'F6', 'AF8', 'AF4', 'F2']
motor_chs = ["FC3","FC1","FC2","FC4","C3","C1","Cz","C2","C4","CP3","CP1","CPz","CP2","CP4"]
epochs_roi = epochs_3.copy().pick_channels(roi_chs, ordered=True)
X_roi = epochs_roi.get_data()
sfreq = epochs_roi.info['sfreq']
print("X_roi",X_roi.shape ,"y_3:", y_3.shape, "class counts:", np.bincount(y_3))


fb_vis = FBCSP(sfreq=sfreq,bands=bands,n_classes=3,n_components=4,reg="ledoit_wolf",
    log=True,
    norm_trace=False
)

# -- Visualize FBCSP patterns ---
fb_vis.fit(X_roi, y_3)

outdir = "csp_patterns"
os.makedirs(outdir, exist_ok=True)
class_names = {0:"Left",1:"Right",2:"Blank"}

for bi, (l,h) in enumerate(bands):
    for k in [0,1,2]:
        csp = fb_vis._csp[bi][k]
        fig = csp.plot_patterns(
            epochs_roi.info,
            components=[0,1,2,3],
            ch_type="eeg",
            show=False,
            size=1.2,
            sphere=(0., 0., 0., 0.13)
        )
        fig.suptitle(f"{class_names[k]} vs Rest | {l}-{h} Hz")
        fname = f"{outdir}/band_{l}-{h}_class_{class_names[k]}.png"
        fig.savefig(fname, dpi=200, bbox_inches="tight")
        plt.close(fig)
print(f"Saved CSP pattern plots to: {outdir}/")


cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=1)

# Apply CSD and get data for classification
epochs_roi_csd = mne.preprocessing.compute_current_source_density(epochs_roi.copy())
X_use = epochs_roi_csd.get_data()

#CSP
# def run_csp(X,y,class_id, base_clf, n_components =4):
#     y_bin = (y==class_id).astype(int)
#     fold_scores = []
#     for train_index, test_index in cv.split(X,y_bin):
#         X_train, X_test = X[train_index], X[test_index]
#         y_train, y_test = y_bin[train_index], y_bin[test_index]
#
#         csp = CSP(n_components=n_components,reg = None, log = True, norm_trace=False)
#         clf = clone(base_clf)
#         #fit = learn parameters from training data
#         #transform = apply learned parameters to data
#         X_train_csp = csp.fit_transform(X_train,y_train)
#         X_test_csp = csp.transform(X_test)
#
#         clf.fit(X_train_csp,y_train)
#         y_pred = clf.predict(X_test_csp)
#         fold_scores.append(accuracy_score(y_test,y_pred))
#     return float(np.mean(fold_scores)),np.array(fold_scores)

# --- 3-class FBCSP + LDA/SVM/KNN/LogReg ---

FBCSP
fbcsp = FBCSP(sfreq=sfreq, bands=bands, n_classes=3, n_components=2, reg= "ledoit_wolf")

# pipe_3 = Pipeline([
#     ("fbcsp", fbcsp),
#     ("scaler", StandardScaler()),
#     ("select", SelectKBest(f_classif, k=30)), 
#     ("clf", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"))
# ])

# scores_real = cross_val_score(pipe_3, X_use, y_3, cv=cv)
# print("3-class REAL mean:", scores_real.mean(), "std:", scores_real.std())

# y_shuf = y_3.copy()
# np.random.RandomState(0).shuffle(y_shuf)
# scores_shuf = cross_val_score(pipe_3, X_use, y_shuf, cv=cv)
# print("3-class SHUF mean:", scores_shuf.mean(), "std:", scores_shuf.std())

# mask_lr = y_3 != 2
# X_lr = X_use[mask_lr]
# y_lr = y_3[mask_lr]  # 0/1

# fbcsp_lr = FBCSP(
#     sfreq=sfreq,
#     bands=bands,
#     n_classes=2,
#     n_components=6,
#     reg="ledoit_wolf",
#     log=True,
#     norm_trace=True
# )

# pipe_lr = Pipeline([
#     ("fbcsp", fbcsp_lr),
#     ("scaler", StandardScaler()),
#     ("select", SelectKBest(f_classif, k=20)),
#     ("clf", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"))
# ])

# scores_lr = cross_val_score(pipe_lr, X_lr, y_lr, cv=cv)
# print("LR REAL mean:", scores_lr.mean(), "std:", scores_lr.std())


classifiers = {
    "LDA" : LinearDiscriminantAnalysis(),
    "SVM" : SVC(kernel='linear'),
    "KNN" : KNeighborsClassifier(),
    "LogReg" : LogisticRegression(max_iter=3000)
}
def run_one_fold(tr, te, X, y, fbcsp, classifiers, scale):
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

        fold_results[name] = (
            accuracy_score(y[te], yhat),
            balanced_accuracy_score(y[te], yhat),
            f1_score(y[te], yhat, average="macro"),
        )
    return fold_results


def summary_results(results, classifiers):
    summary = {}
    for name in classifiers.keys():
        accs  = np.array([fold[name][0] for fold in results], dtype=float)
        baccs = np.array([fold[name][1] for fold in results], dtype=float)
        f1s   = np.array([fold[name][2] for fold in results], dtype=float)
        summary[name] = {
            "acc_mean": float(accs.mean()),
            "acc_folds": accs,
            "bacc_mean": float(baccs.mean()),
            "bacc_folds": baccs,
            "f1_mean": float(f1s.mean()),
            "f1_folds": f1s,
        }
    return summary

splits = list(cv.split(X_roi, y_3))

results = Parallel(n_jobs=-1, prefer="processes")(
    delayed(run_one_fold)(tr, te, X_use, y_3, fbcsp, classifiers, True)
    for tr, te in splits
)

summary = summary_results(results, classifiers)

print("\n=== 3-class Left/Right/Blank (chance=0.333) ===")
for name, s in summary.items():
    print(f"{name:10s} acc={s['acc_mean']:.3f} "
          f"bacc={s['bacc_mean']:.3f} macroF1={s['f1_mean']:.3f} "
          f"folds={np.round(s['acc_folds'],3)}")

end = time.perf_counter()

print(f"Sequential time: {end - start:.4f} seconds")

# --------------------

# OvR CSP
# class_names = {0: "Left", 1: "Right", 2: "Blank"}
# classifiers = {"LDA": lda}
# csp_n_components = 4
#
# for clf_name, base_clf in classifiers.items():
#     print(f"\n=== OvR CSP + {clf_name} ===")
#     for class_id in [0, 1, 2]:
#         mean_acc, folds = run_csp(X_roi, y_3, class_id=class_id,base_clf=base_clf, n_components=4)
#         chance = max(np.mean(y_3 == class_id), 1 - np.mean(y_3 == class_id))  # majority baseline for that binary task
#         print(f"{class_names[class_id]} vs Rest: {mean_acc:.3f}  folds={np.round(folds,3)}  majority={chance:.3f}")
#         y_bin = (y_3 == class_id).astype(int)
#         csp_vis = CSP(n_components=csp_n_components, reg=None, log=True, norm_trace=False)
#         csp_vis.fit(X_roi, y_bin)
#         csp_vis.plot_patterns(epochs_roi.info,ch_type="eeg",components=list(range(csp_n_components)),units="Patterns (AU)",size=1.5,show=True,sphere=(0., 0., 0., 0.13))
#         plt.suptitle(f"CSP patterns: {class_names[class_id]} vs Rest", y=0.98)


#  --- Riemannian Tangent Space + Logistic Regression ---
# #Riemannian Tangent Space
# riemann = Pipeline([
#     ("cov", Covariances(estimator="oas")),
#     ("ts", TangentSpace()),
#     ("scaler", StandardScaler()),
#     ("clf", LogisticRegression(max_iter=2000,class_weight="balanced"))
# ])

# scores_riemann = cross_val_score(riemann, X_roi, y_3, cv=cv, scoring = "balanced_accuracy", n_jobs=10)
# print("\nRiemannian Tangent Space + Logistic Regression")
# print("3-class REAL mean:", scores_riemann.mean(), "std:", scores_riemann.std())