"""
Phase 1.3: Machine Learning Model Training & Evaluation
========================================================
This script implements the complete ML pipeline:
1. Data loading and preprocessing
2. Feature engineering and selection
3. Train-test split with stratification
4. Random Forest model training with hyperparameter tuning
5. Model evaluation and performance metrics
6. Model serialization for production deployment
7. Visualization of results

Author: Flood Sentinel Team
Date: 2025
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10

# Constants
INPUT_FILE = Path("data/final_training_dataset.csv")
MODEL_OUTPUT_DIR = Path("models")
RESULTS_OUTPUT_DIR = Path("results")
MODEL_FILE = MODEL_OUTPUT_DIR / "flood_risk_model.pkl"
SCALER_FILE = MODEL_OUTPUT_DIR / "feature_scaler.pkl"

# Model parameters
RANDOM_STATE = 42
TEST_SIZE = 0.2


def setup_directories():
    """Create necessary output directories."""
    MODEL_OUTPUT_DIR.mkdir(exist_ok=True)
    RESULTS_OUTPUT_DIR.mkdir(exist_ok=True)
    print(f"✓ Created output directories")


def load_and_explore_data():
    """Load the enriched dataset and perform initial exploration."""
    print("\n[1/7] Loading enriched training dataset...")
    
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Training dataset not found: {INPUT_FILE}\n"
            f"Please run '2_enrich_dataset.py' first."
        )
    
    df = pd.read_csv(INPUT_FILE)
    print(f"   ✓ Loaded {len(df)} data points from {INPUT_FILE}")
    print(f"   ✓ Features: {len(df.columns) - 1}")
    print(f"   ✓ Target variable: historical_flood")
    
    # Check for missing values
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(f"\n   ⚠ Missing values detected:")
        print(missing[missing > 0])
    else:
        print(f"   ✓ No missing values")
    
    # Class distribution
    class_dist = df['historical_flood'].value_counts()
    print(f"\n   Target Distribution:")
    print(f"     - Not Flooded (0): {class_dist[0]} ({class_dist[0]/len(df)*100:.1f}%)")
    print(f"     - Flooded (1): {class_dist[1]} ({class_dist[1]/len(df)*100:.1f}%)")
    
    return df


def prepare_features(df: pd.DataFrame):
    """
    Prepare features for model training.
    
    Args:
        df: Input DataFrame with all features
        
    Returns:
        X: Feature matrix
        y: Target vector
        feature_names: List of feature names
    """
    print("\n[2/7] Preparing features for model training...")
    
    # Select numeric features for modeling
    feature_columns = [
        'elevation_m',
        'slope_percent',
        'distance_to_water_m',
        'precipitation_mm_event',
        'soil_permeability_mm_hr',
        'land_cover'  # Numeric NLCD code
    ]
    
    X = df[feature_columns].copy()
    y = df['historical_flood'].copy()
    
    print(f"   ✓ Selected {len(feature_columns)} features:")
    for feat in feature_columns:
        print(f"     - {feat}")
    
    # Feature engineering: Create interaction features
    print(f"\n   Creating engineered features...")
    
    # Interaction: Low elevation + high precipitation = higher risk
    X['elevation_precip_interaction'] = X['elevation_m'] * X['precipitation_mm_event']
    
    # Interaction: Distance to water + low permeability = higher risk
    X['water_soil_interaction'] = X['distance_to_water_m'] * X['soil_permeability_mm_hr']
    
    # Ratio: Precipitation to permeability (runoff potential)
    X['runoff_potential'] = X['precipitation_mm_event'] / (X['soil_permeability_mm_hr'] + 1)
    
    feature_columns.extend([
        'elevation_precip_interaction',
        'water_soil_interaction',
        'runoff_potential'
    ])
    
    print(f"   ✓ Total features (including engineered): {len(feature_columns)}")
    
    return X, y, feature_columns


def split_and_scale_data(X, y):
    """
    Split data into train/test sets and apply feature scaling.
    
    Args:
        X: Feature matrix
        y: Target vector
        
    Returns:
        X_train, X_test, y_train, y_test: Split datasets
        scaler: Fitted StandardScaler object
    """
    print("\n[3/7] Splitting and scaling data...")
    
    # Stratified train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )
    
    print(f"   ✓ Train set: {len(X_train)} samples")
    print(f"   ✓ Test set: {len(X_test)} samples")
    print(f"   ✓ Train class distribution: {y_train.value_counts().to_dict()}")
    print(f"   ✓ Test class distribution: {y_test.value_counts().to_dict()}")
    
    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrame to preserve column names
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
    
    print(f"   ✓ Applied StandardScaler to features")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def train_model(X_train, y_train):
    """
    Train Random Forest classifier with optimized hyperparameters.
    
    Args:
        X_train: Training features
        y_train: Training labels
        
    Returns:
        Trained model
    """
    print("\n[4/7] Training Random Forest model...")
    
    # Initialize Random Forest with tuned hyperparameters
    model = RandomForestClassifier(
        n_estimators=200,           # Number of trees
        max_depth=15,                # Maximum tree depth
        min_samples_split=10,        # Minimum samples to split a node
        min_samples_leaf=5,          # Minimum samples at leaf node
        max_features='sqrt',         # Number of features for best split
        class_weight='balanced',     # Handle class imbalance
        random_state=RANDOM_STATE,
        n_jobs=-1,                   # Use all CPU cores
        verbose=0
    )
    
    print(f"   Model configuration:")
    print(f"     - Algorithm: Random Forest")
    print(f"     - Trees: {model.n_estimators}")
    print(f"     - Max depth: {model.max_depth}")
    print(f"     - Class weight: {model.class_weight}")
    
    # Train the model
    print(f"\n   Training in progress...")
    model.fit(X_train, y_train)
    print(f"   ✓ Model training complete")
    
    # Cross-validation score
    print(f"\n   Performing 5-fold cross-validation...")
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy', n_jobs=-1)
    print(f"   ✓ CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    return model


def evaluate_model(model, X_train, X_test, y_train, y_test, feature_names):
    """
    Comprehensive model evaluation with multiple metrics.
    
    Args:
        model: Trained model
        X_train, X_test: Feature matrices
        y_train, y_test: Target vectors
        feature_names: List of feature names
        
    Returns:
        Dictionary of evaluation metrics
    """
    print("\n[5/7] Evaluating model performance...")
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Prediction probabilities
    y_train_proba = model.predict_proba(X_train)[:, 1]
    y_test_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics = {
        'train_accuracy': accuracy_score(y_train, y_train_pred),
        'test_accuracy': accuracy_score(y_test, y_test_pred),
        'train_precision': precision_score(y_train, y_train_pred),
        'test_precision': precision_score(y_test, y_test_pred),
        'train_recall': recall_score(y_train, y_train_pred),
        'test_recall': recall_score(y_test, y_test_pred),
        'train_f1': f1_score(y_train, y_train_pred),
        'test_f1': f1_score(y_test, y_test_pred),
        'train_auc': roc_auc_score(y_train, y_train_proba),
        'test_auc': roc_auc_score(y_test, y_test_proba)
    }
    
    # Print metrics
    print(f"\n   {'Metric':<20} {'Train':<12} {'Test':<12}")
    print(f"   {'-'*44}")
    print(f"   {'Accuracy':<20} {metrics['train_accuracy']:<12.4f} {metrics['test_accuracy']:<12.4f}")
    print(f"   {'Precision':<20} {metrics['train_precision']:<12.4f} {metrics['test_precision']:<12.4f}")
    print(f"   {'Recall':<20} {metrics['train_recall']:<12.4f} {metrics['test_recall']:<12.4f}")
    print(f"   {'F1-Score':<20} {metrics['train_f1']:<12.4f} {metrics['test_f1']:<12.4f}")
    print(f"   {'ROC-AUC':<20} {metrics['train_auc']:<12.4f} {metrics['test_auc']:<12.4f}")
    
    # Classification report
    print(f"\n   Detailed Classification Report (Test Set):")
    print(f"   {'-'*44}")
    print(classification_report(y_test, y_test_pred, target_names=['Not Flooded', 'Flooded']))
    
    return metrics, y_test_pred, y_test_proba


def visualize_results(model, y_test, y_test_pred, y_test_proba, feature_names):
    """
    Create and save visualization plots.
    
    Args:
        model: Trained model
        y_test: True test labels
        y_test_pred: Predicted test labels
        y_test_proba: Prediction probabilities
        feature_names: List of feature names
    """
    print("\n[6/7] Generating visualizations...")
    
    # 1. Confusion Matrix
    fig, ax = plt.subplots(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_test_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Not Flooded', 'Flooded'],
                yticklabels=['Not Flooded', 'Flooded'],
                ax=ax)
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax.set_title('Confusion Matrix - Flood Risk Prediction', fontsize=14, fontweight='bold', pad=20)
    
    # Add percentage annotations
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j + 0.5, i + 0.7, f'({cm_percent[i, j]*100:.1f}%)',
                   ha='center', va='center', fontsize=10, color='gray')
    
    plt.tight_layout()
    confusion_matrix_file = RESULTS_OUTPUT_DIR / "confusion_matrix.png"
    plt.savefig(confusion_matrix_file, dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved confusion matrix: {confusion_matrix_file}")
    plt.close()
    
    # 2. Feature Importance
    fig, ax = plt.subplots(figsize=(10, 8))
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=True)
    
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(feature_importance)))
    ax.barh(feature_importance['feature'], feature_importance['importance'], color=colors)
    ax.set_xlabel('Importance Score', fontsize=12, fontweight='bold')
    ax.set_ylabel('Feature', fontsize=12, fontweight='bold')
    ax.set_title('Feature Importance - Random Forest Model', fontsize=14, fontweight='bold', pad=20)
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    feature_importance_file = RESULTS_OUTPUT_DIR / "feature_importance.png"
    plt.savefig(feature_importance_file, dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved feature importance: {feature_importance_file}")
    plt.close()
    
    # 3. ROC Curve
    fig, ax = plt.subplots(figsize=(8, 6))
    fpr, tpr, _ = roc_curve(y_test, y_test_proba)
    auc_score = roc_auc_score(y_test, y_test_proba)
    
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_score:.3f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    ax.set_title('ROC Curve - Flood Risk Prediction', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    roc_curve_file = RESULTS_OUTPUT_DIR / "roc_curve.png"
    plt.savefig(roc_curve_file, dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved ROC curve: {roc_curve_file}")
    plt.close()
    
    # 4. Probability Distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Separate probabilities by actual class
    proba_not_flooded = y_test_proba[y_test == 0]
    proba_flooded = y_test_proba[y_test == 1]
    
    ax.hist(proba_not_flooded, bins=30, alpha=0.6, label='Actually Not Flooded', color='green', edgecolor='black')
    ax.hist(proba_flooded, bins=30, alpha=0.6, label='Actually Flooded', color='red', edgecolor='black')
    ax.axvline(x=0.5, color='black', linestyle='--', linewidth=2, label='Decision Threshold')
    ax.set_xlabel('Predicted Flood Probability', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax.set_title('Distribution of Predicted Probabilities', fontsize=14, fontweight='bold', pad=20)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    probability_dist_file = RESULTS_OUTPUT_DIR / "probability_distribution.png"
    plt.savefig(probability_dist_file, dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved probability distribution: {probability_dist_file}")
    plt.close()


def save_model_artifacts(model, scaler):
    """
    Serialize and save model and scaler for production use.
    
    Args:
        model: Trained model
        scaler: Fitted scaler
    """
    print("\n[7/7] Saving model artifacts...")
    
    # Save model
    with open(MODEL_FILE, 'wb') as f:
        pickle.dump(model, f)
    print(f"   ✓ Saved model: {MODEL_FILE}")
    print(f"     File size: {MODEL_FILE.stat().st_size / 1024:.2f} KB")
    
    # Save scaler
    with open(SCALER_FILE, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"   ✓ Saved scaler: {SCALER_FILE}")
    print(f"     File size: {SCALER_FILE.stat().st_size / 1024:.2f} KB")


def main():
    """Main execution pipeline."""
    print("=" * 70)
    print("PHASE 1.3: MODEL TRAINING & EVALUATION")
    print("=" * 70)
    
    # Setup
    setup_directories()
    
    # Load data
    df = load_and_explore_data()
    
    # Prepare features
    X, y, feature_names = prepare_features(df)
    
    # Split and scale
    X_train, X_test, y_train, y_test, scaler = split_and_scale_data(X, y)
    
    # Train model
    model = train_model(X_train, y_train)
    
    # Evaluate
    metrics, y_test_pred, y_test_proba = evaluate_model(
        model, X_train, X_test, y_train, y_test, feature_names
    )
    
    # Visualize
    visualize_results(model, y_test, y_test_pred, y_test_proba, feature_names)
    
    # Save artifacts
    save_model_artifacts(model, scaler)
    
    print("\n" + "=" * 70)
    print("✓ PHASE 1 COMPLETE: MODEL TRAINING & EVALUATION")
    print("=" * 70)
    print(f"\nModel Performance Summary:")
    print(f"  - Test Accuracy: {metrics['test_accuracy']:.4f}")
    print(f"  - Test F1-Score: {metrics['test_f1']:.4f}")
    print(f"  - Test ROC-AUC: {metrics['test_auc']:.4f}")
    print(f"\nModel artifacts saved to: {MODEL_OUTPUT_DIR}")
    print(f"Visualizations saved to: {RESULTS_OUTPUT_DIR}")
    print(f"\n{'='*70}")
    print("NEXT STEPS:")
    print("  Phase 1 is now complete. Await confirmation before proceeding to Phase 2.")
    print("  Phase 2 will focus on building the Flask backend API.")
    print("=" * 70)


if __name__ == "__main__":
    main()
