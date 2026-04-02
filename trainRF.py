import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
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


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    # Drop meta columns and label columns; keep feature columns
    X = df.drop(columns=['subject_id', 'date', 'label_cv_threshold', 'label_cv_numeric'], errors='ignore')
    y = df['label_cv_numeric']
    # If object-valued features exist, convert via get_dummies
    X = pd.get_dummies(X, drop_first=True)
    return X, y


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
    X, y = build_features(df)

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y,
        test_size=args.test_size,
        stratify=y,
        random_state=args.seed,
    )

    val_fraction = args.val_size / (1 - args.test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=val_fraction,
        stratify=y_trainval,
        random_state=args.seed,
    )

    print('Data shapes:')
    print('  X_train', X_train.shape, 'y_train', y_train.shape)
    print('  X_val', X_val.shape, 'y_val', y_val.shape)
    print('  X_test', X_test.shape, 'y_test', y_test.shape)

    print('Class distribution (train):')
    print(y_train.value_counts(normalize=True))

    # Apply SMOTE to training data only
    print('\nApplying SMOTE to training data...')
    smote = SMOTE(random_state=args.seed, k_neighbors=3)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    print(f'  After SMOTE: {len(X_train_balanced)} samples')
    print('  Class distribution after SMOTE:')
    print(pd.Series(y_train_balanced).value_counts(normalize=True))

    # Hyperparameter tuning with GridSearchCV
    if args.tune:
        print('\nRunning GridSearchCV for hyperparameter tuning...')
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [5, 8, 10, 15],
            'min_samples_split': [2, 5, 10],
            'max_features': ['sqrt', 'log2'],
        }
        base_clf = RandomForestClassifier(
            class_weight='balanced',
            random_state=args.seed,
            n_jobs=-1,
        )
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
        grid = GridSearchCV(base_clf, param_grid, cv=skf, scoring='f1_weighted', n_jobs=-1, verbose=1)
        grid.fit(X_train_balanced, y_train_balanced)
        print(f'\nBest params: {grid.best_params_}')
        print(f'Best CV F1 score: {grid.best_score_:.4f}')
        clf = grid.best_estimator_
    else:
        # Use reasonable defaults with regularization
        clf = RandomForestClassifier(
            n_estimators=200,
            max_depth=8,  # Reduced from None to prevent overfitting
            min_samples_split=5,
            max_features='sqrt',
            class_weight='balanced',
            random_state=args.seed,
            n_jobs=-1,
        )
        clf.fit(X_train_balanced, y_train_balanced)

    # Feature importance analysis
    print('\n=== Feature Importance ===')
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': clf.feature_importances_
    }).sort_values('importance', ascending=False)
    print(feature_importance.head(20))

    # Feature selection: keep top N features if requested
    if args.n_features is not None and args.n_features < len(X_train.columns):
        print(f'\nSelecting top {args.n_features} features by importance...')
        top_features = feature_importance.head(args.n_features)['feature'].tolist()
        X_train = X_train[top_features]
        X_val = X_val[top_features]
        X_test = X_test[top_features]
        X_train_balanced = X_train_balanced[top_features]

        # Retrain with reduced features
        clf = RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_split=5,
            max_features='sqrt',
            class_weight='balanced',
            random_state=args.seed,
            n_jobs=-1,
        )
        clf.fit(X_train_balanced, y_train_balanced)

    # Cross-validation on training set
    print('\n=== 5-Fold Cross-Validation on Training Data ===')
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    cv_scores = cross_val_score(clf, X_train_balanced, y_train_balanced, cv=skf, scoring='f1_weighted')
    print(f'F1 CV scores: {cv_scores}')
    print(f'F1 CV mean: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})')

    def show_scores(split_name, X_split, y_split):
        y_pred = clf.predict(X_split)
        y_proba = clf.predict_proba(X_split)[:, 1]
        print(f"\n== {split_name} performance ==")
        print('accuracy', f'{accuracy_score(y_split, y_pred):.4f}')
        print('precision', f'{precision_score(y_split, y_pred, zero_division=0):.4f}')
        print('recall', f'{recall_score(y_split, y_pred, zero_division=0):.4f}')
        print('f1', f'{f1_score(y_split, y_pred, zero_division=0):.4f}')
        print('roc_auc', f'{roc_auc_score(y_split, y_proba):.4f}')
        print('confusion matrix')
        print(confusion_matrix(y_split, y_pred))
        print('classification report')
        print(classification_report(y_split, y_pred, zero_division=0))

    show_scores('train (balanced)', X_train_balanced, y_train_balanced)
    show_scores('val', X_val, y_val)
    show_scores('test', X_test, y_test)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, out_path)

    print(f'\nRandomForest model saved to {out_path}')


if __name__ == '__main__':
    main()
