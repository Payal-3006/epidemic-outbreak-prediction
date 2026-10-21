import numpy as np
import dill
import matplotlib.pyplot as plt
import itertools

from sklearn.preprocessing import Normalizer
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
from sklearn.ensemble import RandomForestClassifier

from imblearn.combine import SMOTETomek
from imblearn.pipeline import Pipeline

from numpy import interp

# -----------------------------
# PLOT CONFUSION MATRIX
# -----------------------------
def plot_confusion_matrix(cm, classes):
    plt.figure(figsize=(6,6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig("confusion_matrix.png")
    plt.close()


# -----------------------------
# ROC CURVE
# -----------------------------
def plot_roc(X, y, pipeline):
    cv = StratifiedKFold(n_splits=5, shuffle=True)
    tprs = []
    mean_fpr = np.linspace(0, 1, 100)

    for train, test in cv.split(X, y):
        probas_ = pipeline.fit(X[train], y[train]).predict_proba(X[test])
        fpr, tpr, _ = roc_curve(y[test], probas_[:, 1])
        tprs.append(interp(mean_fpr, fpr, tpr))

    mean_tpr = np.mean(tprs, axis=0)
    mean_auc = auc(mean_fpr, mean_tpr)

    plt.plot(mean_fpr, mean_tpr, label=f"AUC = {mean_auc:.2f}")
    plt.plot([0,1],[0,1],'--')
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC Curve")
    plt.legend()
    plt.savefig("roc_curve.png")
    plt.close()


# -----------------------------
# MAIN
# -----------------------------
def main():
    print("🚀 Loading data...")

    with open('./pkl/X.pkl', 'rb') as f:
        X = dill.load(f)

    with open('./pkl/y.pkl', 'rb') as f:
        y = dill.load(f)

    print("✅ Data loaded:", X.shape)

    # Pipeline (NO XGBOOST 🚫)
    pipeline = Pipeline([
        ('scaler', Normalizer()),
        ('smt', SMOTETomek()),
        ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    # Train-test split
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)

    for train_idx, test_idx in sss.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

    print("🚀 Training model...")
    pipeline.fit(X_train, y_train)

    print("✅ Model trained")

    # Predictions
    y_pred = pipeline.predict(X_test)

    # Evaluation
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, ["No Outbreak", "Outbreak"])

    print("\n📊 Classification Report:\n")
    print(classification_report(y_test, y_pred))

    # ROC
    plot_roc(X, y, pipeline)

    print("🎉 Done! Check confusion_matrix.png and roc_curve.png")


# -----------------------------
if __name__ == "__main__":
    main()