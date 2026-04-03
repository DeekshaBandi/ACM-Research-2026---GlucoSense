import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut
from imblearn.over_sampling import SMOTE


def load_dataset(path: Path) -> pd.DataFrame:
    """
    Load and merge CGM + Wearable features.
    If the dataset doesn't have CGM metrics, auto-enhance it.
    """
    df = pd.read_csv(path)
    
    # Check if CGM metrics are already present
    cgm_indicators = ['sd_glucose', 'mage', 'tir_percent', 'tar_percent', 'tbr_percent']
    has_cgm = any(col in df.columns for col in cgm_indicators)
    
    if not has_cgm:
        print("[INFO] CGM metrics not found. Auto-merging CGM + Wearable data...")
        # Load raw sources
        cgm = pd.read_csv("output/official_participant_day_outcomes.csv")
        wearable_raw = pd.read_csv("output/AllParticipants.csv")
        
        # Prepare CGM
        cgm["subject_id"] = cgm["subject_id"].astype(str).str.zfill(3)
        cgm["date"] = pd.to_datetime(cgm["date"], errors="coerce").dt.date
        
        # Prepare wearable
        wearable_raw = wearable_raw.rename(columns={"Subject_ID": "subject_id", "Date": "date"})
        wearable_raw["subject_id"] = wearable_raw["subject_id"].astype(str).str.zfill(3)
        wearable_raw["date"] = pd.to_datetime(wearable_raw["date"], errors="coerce").dt.date
        wearable_raw = wearable_raw.drop_duplicates(subset=["subject_id", "date"]).copy()
        
        # Keep CGM metrics (drop only target + metadata)
        cgm_keep = cgm.drop(columns=["cv_percentage", "cv_threshold_used", "label_method_used"], errors="ignore")
        
        # Merge
        df = cgm_keep.merge(wearable_raw, on=["subject_id", "date"], how="inner", validate="one_to_one")
        df = df.sort_values(["subject_id", "date"]).reset_index(drop=True)
        print(f"[OK] Merged: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Verify labels exist
    required = ['subject_id', 'date', 'label_cv_numeric']
    if not all(col in df.columns for col in required):
        raise ValueError(f"dataset at {path} missing required columns {required}")
    return df


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, np.ndarray]:
    # Drop meta columns and label columns; keep feature columns
    groups = df['subject_id'].values
    X = df.drop(columns=['subject_id', 'date', 'label_cv_threshold', 'label_cv_numeric'], errors='ignore')
    y = df['label_cv_numeric']
    # If object-valued features exist, convert via get_dummies
    X = pd.get_dummies(X, drop_first=True)
    return X, y, groups


def main():
    parser = argparse.ArgumentParser(description='Train Random Forest on model-ready daily dataset with regularization & SMOTE')
    parser.add_argument('--input', default='output/model_ready_daily_dataset.csv', help='Path to model-ready dataset CSV')
    parser.add_argument('--output', default='output/rf_model.joblib', help='Path to save trained model')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--test-size', type=float, default=0.2, help='Test split fraction')
    parser.add_argument('--val-size', type=float, default=0.2, help='Validation split fraction')
    parser.add_argument('--tune', action='store_true', help='Run GridSearchCV hyperparameter tuning')
    parser.add_argument('--n-features', type=int, default=None, help='Keep top N features by importance (None=all)')
    args = parser.parse_args()

    path = Path(args.input)
    if not path.exists():
        raise FileNotFoundError(f"Cannot find input file: {path}")

    df = load_dataset(path)
    X, y, groups = build_features(df)

    print(f'Dataset: {X.shape[0]} samples, {X.shape[1]} features, {len(np.unique(groups))} subjects')
    print('Class distribution:')
    print(y.value_counts(normalize=True))

    def make_clf():
        return RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_split=5,
            max_features='sqrt',
            class_weight='balanced',
            random_state=args.seed,
            n_jobs=-1,
        )

    # Hyperparameter tuning with GridSearchCV using LOSOCV
    if args.tune:
        print('\nRunning GridSearchCV with LOSOCV for hyperparameter tuning...')
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [5, 8, 10, 15],
            'min_samples_split': [2, 5, 10],
            'max_features': ['sqrt', 'log2'],
        }
        logo = LeaveOneGroupOut()
        grid = GridSearchCV(make_clf(), param_grid, cv=logo, scoring='f1_weighted', n_jobs=-1, verbose=1)
        grid.fit(X, y, groups=groups)
        print(f'\nBest params: {grid.best_params_}')
        print(f'Best LOSOCV F1 score: {grid.best_score_:.4f}')
        best_params = grid.best_params_
    else:
        best_params = None

    # === Leave-One-Subject-Out Cross-Validation ===
    print('\n=== Leave-One-Subject-Out Cross-Validation ===')
    logo = LeaveOneGroupOut()
    smote = SMOTE(random_state=args.seed, k_neighbors=3)

    fold_results = []
    all_y_true = []
    all_y_pred = []
    all_y_proba = []

    X_arr = X.values
    y_arr = y.values

    for train_idx, test_idx in logo.split(X_arr, y_arr, groups):
        subject_out = np.unique(groups[test_idx])[0]
        X_tr, X_te = X_arr[train_idx], X_arr[test_idx]
        y_tr, y_te = y_arr[train_idx], y_arr[test_idx]

        # Apply SMOTE to training fold only
        X_tr_bal, y_tr_bal = smote.fit_resample(X_tr, y_tr)

        if best_params:
            fold_clf = RandomForestClassifier(
                **best_params,
                class_weight='balanced',
                random_state=args.seed,
                n_jobs=-1,
            )
        else:
            fold_clf = make_clf()

        fold_clf.fit(X_tr_bal, y_tr_bal)
        y_pred = fold_clf.predict(X_te)
        y_proba = fold_clf.predict_proba(X_te)[:, 1]

        fold_f1 = f1_score(y_te, y_pred, zero_division=0)
        fold_acc = accuracy_score(y_te, y_pred)
        fold_results.append({'subject': subject_out, 'f1': fold_f1, 'accuracy': fold_acc, 'n_test': len(y_te)})
        all_y_true.extend(y_te)
        all_y_pred.extend(y_pred)
        all_y_proba.extend(y_proba)
        print(f'  Subject {subject_out}: F1={fold_f1:.4f}, Acc={fold_acc:.4f}, n={len(y_te)}')

    results_df = pd.DataFrame(fold_results)
    print(f'\nLOSOCV F1  mean={results_df["f1"].mean():.4f}  std={results_df["f1"].std():.4f}')
    print(f'LOSOCV Acc mean={results_df["accuracy"].mean():.4f}  std={results_df["accuracy"].std():.4f}')

    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)
    all_y_proba = np.array(all_y_proba)

    print('\n=== Aggregated LOSOCV Performance ===')
    print('accuracy ', f'{accuracy_score(all_y_true, all_y_pred):.4f}')
    print('precision', f'{precision_score(all_y_true, all_y_pred, zero_division=0):.4f}')
    print('recall   ', f'{recall_score(all_y_true, all_y_pred, zero_division=0):.4f}')
    print('f1       ', f'{f1_score(all_y_true, all_y_pred, zero_division=0):.4f}')
    print('roc_auc  ', f'{roc_auc_score(all_y_true, all_y_proba):.4f}')
    print('confusion matrix')
    print(confusion_matrix(all_y_true, all_y_pred))
    print('classification report')
    print(classification_report(all_y_true, all_y_pred, zero_division=0))

    # Train final model on all data with SMOTE
    print('\n=== Training final model on all data ===')
    X_bal, y_bal = smote.fit_resample(X_arr, y_arr)
    if best_params:
        clf = RandomForestClassifier(
            **best_params,
            class_weight='balanced',
            random_state=args.seed,
            n_jobs=-1,
        )
    else:
        clf = make_clf()
    clf.fit(X_bal, y_bal)

    # Feature importance from final model
    print('\n=== Feature Importance (final model) ===')
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': clf.feature_importances_
    }).sort_values('importance', ascending=False)
    print(feature_importance.head(20))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, out_path)

    print(f'\nRandomForest model saved to {out_path}')


if __name__ == '__main__':
    main()
