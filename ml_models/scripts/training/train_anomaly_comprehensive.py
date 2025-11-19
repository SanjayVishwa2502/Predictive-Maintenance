"""
COMPREHENSIVE ANOMALY DETECTION PIPELINE
=======================================
Implements multiple anomaly detection approaches:
1. Statistical Methods (Z-score, IQR, Modified Z-score)
2. Distance-Based (LOF, KNN)
3. Density-Based (DBSCAN)
4. Isolation-Based (Isolation Forest)
5. Boundary-Based (One-Class SVM, Elliptic Envelope)
6. Deep Learning (Autoencoder)
7. Ensemble Methods (Voting, Weighted Average)

With comprehensive validation and detailed reporting.
"""

import numpy as np
import pandas as pd
import joblib
import json
import time
import warnings
from pathlib import Path
from datetime import datetime
import sys

# ML imports
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, 
    precision_recall_curve, roc_curve, average_precision_score
)

# Deep Learning
try:
    from tensorflow import keras
    from tensorflow.keras import layers, models
    KERAS_AVAILABLE = True
except ImportError:
    KERAS_AVAILABLE = False
    print("⚠️ TensorFlow not available - Autoencoder will be skipped")

# MLflow
import mlflow

# Custom imports
sys.path.append(str(Path(__file__).parent.parent.parent))
from scripts.data_preparation.feature_engineering import prepare_ml_data

warnings.filterwarnings('ignore')


class StatisticalAnomalyDetector:
    """Statistical anomaly detection using Z-score, IQR, and Modified Z-score"""
    
    def __init__(self, method='zscore', threshold=3.0):
        self.method = method
        self.threshold = threshold
        self.stats = {}
        
    def fit(self, X):
        """Compute statistics from normal data"""
        if self.method == 'zscore':
            self.stats['mean'] = np.mean(X, axis=0)
            self.stats['std'] = np.std(X, axis=0)
            
        elif self.method == 'iqr':
            self.stats['q1'] = np.percentile(X, 25, axis=0)
            self.stats['q3'] = np.percentile(X, 75, axis=0)
            self.stats['iqr'] = self.stats['q3'] - self.stats['q1']
            
        elif self.method == 'modified_zscore':
            self.stats['median'] = np.median(X, axis=0)
            self.stats['mad'] = np.median(np.abs(X - self.stats['median']), axis=0)
            
        return self
        
    def predict(self, X):
        """Predict: 1 for normal, -1 for anomaly"""
        if self.method == 'zscore':
            z_scores = np.abs((X - self.stats['mean']) / (self.stats['std'] + 1e-10))
            anomaly_scores = np.max(z_scores, axis=1)
            
        elif self.method == 'iqr':
            lower_bound = self.stats['q1'] - self.threshold * self.stats['iqr']
            upper_bound = self.stats['q3'] + self.threshold * self.stats['iqr']
            outliers = (X < lower_bound) | (X > upper_bound)
            anomaly_scores = np.sum(outliers, axis=1) / X.shape[1]
            
        elif self.method == 'modified_zscore':
            modified_z = 0.6745 * np.abs(X - self.stats['median']) / (self.stats['mad'] + 1e-10)
            anomaly_scores = np.max(modified_z, axis=1)
            
        predictions = np.where(anomaly_scores > self.threshold, -1, 1)
        return predictions
    
    def decision_function(self, X):
        """Return anomaly scores (higher = more anomalous)"""
        if self.method == 'zscore':
            z_scores = np.abs((X - self.stats['mean']) / (self.stats['std'] + 1e-10))
            return np.max(z_scores, axis=1)
        elif self.method == 'iqr':
            lower_bound = self.stats['q1'] - self.threshold * self.stats['iqr']
            upper_bound = self.stats['q3'] + self.threshold * self.stats['iqr']
            outliers = (X < lower_bound) | (X > upper_bound)
            return np.sum(outliers, axis=1) / X.shape[1]
        elif self.method == 'modified_zscore':
            modified_z = 0.6745 * np.abs(X - self.stats['median']) / (self.stats['mad'] + 1e-10)
            return np.max(modified_z, axis=1)


class AutoencoderAnomalyDetector:
    """Deep learning autoencoder for anomaly detection"""
    
    def __init__(self, encoding_dim=32, epochs=50, batch_size=256, contamination=0.1):
        self.encoding_dim = encoding_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.contamination = contamination
        self.autoencoder = None
        self.threshold = None
        
    def fit(self, X):
        """Train autoencoder on normal data"""
        input_dim = X.shape[1]
        
        # Build autoencoder architecture
        encoder_input = layers.Input(shape=(input_dim,))
        encoded = layers.Dense(128, activation='relu')(encoder_input)
        encoded = layers.Dropout(0.2)(encoded)
        encoded = layers.Dense(64, activation='relu')(encoded)
        encoded = layers.Dense(self.encoding_dim, activation='relu', name='encoding')(encoded)
        
        decoded = layers.Dense(64, activation='relu')(encoded)
        decoded = layers.Dropout(0.2)(decoded)
        decoded = layers.Dense(128, activation='relu')(decoded)
        decoded = layers.Dense(input_dim, activation='linear')(decoded)
        
        self.autoencoder = models.Model(encoder_input, decoded)
        self.autoencoder.compile(optimizer='adam', loss='mse')
        
        # Train
        self.autoencoder.fit(
            X, X,
            epochs=self.epochs,
            batch_size=self.batch_size,
            shuffle=True,
            validation_split=0.1,
            verbose=0
        )
        
        # Compute reconstruction errors on training data to set threshold
        reconstructions = self.autoencoder.predict(X, verbose=0)
        mse = np.mean(np.power(X - reconstructions, 2), axis=1)
        self.threshold = np.percentile(mse, 100 * (1 - self.contamination))
        
        return self
        
    def predict(self, X):
        """Predict: 1 for normal, -1 for anomaly"""
        scores = self.decision_function(X)
        return np.where(scores > self.threshold, -1, 1)
    
    def decision_function(self, X):
        """Return reconstruction error (anomaly score)"""
        reconstructions = self.autoencoder.predict(X, verbose=0)
        mse = np.mean(np.power(X - reconstructions, 2), axis=1)
        return mse


class EnsembleAnomalyDetector:
    """Ensemble of multiple anomaly detectors with voting"""
    
    def __init__(self, detectors, method='majority_vote'):
        self.detectors = detectors
        self.method = method
        
    def fit(self, X):
        """Fit all detectors"""
        for name, detector in self.detectors.items():
            print(f"  Training {name}...")
            detector.fit(X)
        return self
        
    def predict(self, X):
        """Ensemble prediction"""
        predictions = []
        for detector in self.detectors.values():
            pred = detector.predict(X)
            # Convert to binary: -1 (anomaly) -> 1, 1 (normal) -> 0
            binary_pred = (pred == -1).astype(int)
            predictions.append(binary_pred)
            
        predictions = np.array(predictions)
        
        if self.method == 'majority_vote':
            # Majority vote: if more than half say anomaly
            votes = np.sum(predictions, axis=0)
            ensemble_pred = (votes > len(self.detectors) / 2).astype(int)
        elif self.method == 'unanimous':
            # All must agree it's anomaly
            ensemble_pred = np.all(predictions, axis=0).astype(int)
        elif self.method == 'any':
            # Any detector flags as anomaly
            ensemble_pred = np.any(predictions, axis=0).astype(int)
        
        # Convert back: 1 (anomaly) -> -1, 0 (normal) -> 1
        return np.where(ensemble_pred == 1, -1, 1)
    
    def decision_function(self, X):
        """Average anomaly scores from all detectors"""
        scores = []
        for detector in self.detectors.values():
            if hasattr(detector, 'decision_function'):
                score = detector.decision_function(X)
            else:
                score = detector.score_samples(X) * -1  # Convert to anomaly score
            scores.append(score)
        return np.mean(scores, axis=0)


def evaluate_detector(detector, X_test, y_true, detector_name):
    """Comprehensive evaluation of anomaly detector"""
    
    print(f"\n{'=' * 60}")
    print(f"Evaluating: {detector_name}")
    print(f"{'=' * 60}")
    
    # Predictions
    y_pred = detector.predict(X_test)
    y_pred_binary = (y_pred == -1).astype(int)  # 1 for anomaly
    
    # Basic metrics
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred_binary),
        'precision': precision_score(y_true, y_pred_binary, zero_division=0),
        'recall': recall_score(y_true, y_pred_binary, zero_division=0),
        'f1_score': f1_score(y_true, y_pred_binary, zero_division=0)
    }
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred_binary)
    tn, fp, fn, tp = cm.ravel()
    
    metrics['true_negatives'] = int(tn)
    metrics['false_positives'] = int(fp)
    metrics['false_negatives'] = int(fn)
    metrics['true_positives'] = int(tp)
    metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # ROC-AUC if decision_function available
    try:
        if hasattr(detector, 'decision_function'):
            scores = detector.decision_function(X_test)
        elif hasattr(detector, 'score_samples'):
            scores = -detector.score_samples(X_test)
        else:
            scores = y_pred_binary
            
        if len(np.unique(scores)) > 1:
            metrics['roc_auc'] = roc_auc_score(y_true, scores)
            metrics['avg_precision'] = average_precision_score(y_true, scores)
        else:
            metrics['roc_auc'] = 0.0
            metrics['avg_precision'] = 0.0
    except:
        metrics['roc_auc'] = 0.0
        metrics['avg_precision'] = 0.0
    
    # Print results
    print(f"  Accuracy:     {metrics['accuracy']:.4f}")
    print(f"  Precision:    {metrics['precision']:.4f}")
    print(f"  Recall:       {metrics['recall']:.4f}")
    print(f"  F1 Score:     {metrics['f1_score']:.4f}")
    print(f"  Specificity:  {metrics['specificity']:.4f}")
    print(f"  ROC-AUC:      {metrics['roc_auc']:.4f}")
    print(f"  Avg Precision: {metrics['avg_precision']:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"    TN={tn:6d}  FP={fp:6d}")
    print(f"    FN={fn:6d}  TP={tp:6d}")
    
    return metrics, cm


def train_comprehensive_anomaly_detection(machine_id, config):
    """
    Train comprehensive anomaly detection models with multiple algorithms
    
    Returns:
        Detailed report with all model performances
    """
    
    print(f"\n{'#' * 80}")
    print(f"#  COMPREHENSIVE ANOMALY DETECTION TRAINING")
    print(f"#  Machine: {machine_id}")
    print(f"#  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#' * 80}\n")
    
    # MLflow experiment
    mlflow.set_experiment(f"ML_Anomaly_Comprehensive_{machine_id}")
    
    with mlflow.start_run(run_name=f"{machine_id}_comprehensive_anomaly"):
        overall_start = time.time()
        
        # =====================================================================
        # STEP 1: DATA LOADING & PREPROCESSING
        # =====================================================================
        print("\n" + "=" * 80)
        print("STEP 1: DATA LOADING & PREPROCESSING")
        print("=" * 80)
        
        train_df, val_df, test_df = prepare_ml_data(machine_id, 'classification')
        train_data = pd.concat([train_df, val_df], ignore_index=True)
        
        # Filter to normal samples for training
        if 'failure_status' in train_data.columns:
            normal_data = train_data[train_data['failure_status'] == 0]
            print(f"  Total samples: {len(train_data)}")
            print(f"  Normal samples: {len(normal_data)} ({len(normal_data)/len(train_data)*100:.1f}%)")
            print(f"  Anomaly samples: {len(train_data) - len(normal_data)}")
        else:
            normal_data = train_data
            print(f"  No failure_status - using all {len(normal_data)} samples")
        
        # Feature selection
        exclude_cols = ['failure_status', 'rul', 'machine_id', 'timestamp']
        feature_cols = [col for col in normal_data.columns if col not in exclude_cols]
        numeric_cols = normal_data[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = numeric_cols
        
        print(f"  Features: {len(feature_cols)}")
        
        # Handle missing values
        imputer = SimpleImputer(strategy='mean')
        X_train_raw = normal_data[feature_cols].values
        X_test_raw = test_df[feature_cols].values
        
        X_train_imputed = imputer.fit_transform(X_train_raw)
        X_test_imputed = imputer.transform(X_test_raw)
        
        # Scaling (important for distance-based methods)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_imputed)
        X_test = scaler.transform(X_test_imputed)
        
        print(f"  Training shape: {X_train.shape}")
        print(f"  Test shape: {X_test.shape}")
        
        # Ground truth labels
        if 'failure_status' in test_df.columns:
            y_true = (test_df['failure_status'].values == 1).astype(int)
            print(f"  Test anomalies: {np.sum(y_true)} ({np.sum(y_true)/len(y_true)*100:.1f}%)")
        else:
            print("  ⚠️ No ground truth labels available")
            y_true = None
        
        # =====================================================================
        # STEP 2: TRAIN MULTIPLE ANOMALY DETECTORS
        # =====================================================================
        print("\n" + "=" * 80)
        print("STEP 2: TRAINING MULTIPLE ANOMALY DETECTION ALGORITHMS")
        print("=" * 80)
        
        detectors = {}
        training_times = {}
        contamination = config.get('contamination', 0.1)
        
        # 1. Isolation Forest
        print("\n[1/10] Isolation Forest")
        start = time.time()
        detectors['isolation_forest'] = IsolationForest(
            contamination=contamination,
            n_estimators=config.get('n_estimators', 100),
            max_samples='auto',
            random_state=42,
            n_jobs=-1
        )
        detectors['isolation_forest'].fit(X_train)
        training_times['isolation_forest'] = time.time() - start
        print(f"  ✅ Trained in {training_times['isolation_forest']:.2f}s")
        
        # 2. One-Class SVM
        print("\n[2/10] One-Class SVM")
        start = time.time()
        detectors['one_class_svm'] = OneClassSVM(
            gamma='auto',
            nu=contamination,
            kernel='rbf'
        )
        detectors['one_class_svm'].fit(X_train)
        training_times['one_class_svm'] = time.time() - start
        print(f"  ✅ Trained in {training_times['one_class_svm']:.2f}s")
        
        # 3. Local Outlier Factor
        print("\n[3/10] Local Outlier Factor (LOF)")
        start = time.time()
        detectors['lof'] = LocalOutlierFactor(
            n_neighbors=20,
            contamination=contamination,
            novelty=True
        )
        detectors['lof'].fit(X_train)
        training_times['lof'] = time.time() - start
        print(f"  ✅ Trained in {training_times['lof']:.2f}s")
        
        # 4. Elliptic Envelope
        print("\n[4/10] Elliptic Envelope (Robust Covariance)")
        start = time.time()
        try:
            detectors['elliptic_envelope'] = EllipticEnvelope(
                contamination=contamination,
                random_state=42
            )
            detectors['elliptic_envelope'].fit(X_train)
            training_times['elliptic_envelope'] = time.time() - start
            print(f"  ✅ Trained in {training_times['elliptic_envelope']:.2f}s")
        except Exception as e:
            print(f"  ⚠️ Failed: {e}")
            training_times['elliptic_envelope'] = 0
        
        # 5. Statistical: Z-Score
        print("\n[5/10] Statistical Method: Z-Score")
        start = time.time()
        detectors['zscore'] = StatisticalAnomalyDetector(method='zscore', threshold=3.0)
        detectors['zscore'].fit(X_train)
        training_times['zscore'] = time.time() - start
        print(f"  ✅ Trained in {training_times['zscore']:.2f}s")
        
        # 6. Statistical: IQR
        print("\n[6/10] Statistical Method: IQR (Interquartile Range)")
        start = time.time()
        detectors['iqr'] = StatisticalAnomalyDetector(method='iqr', threshold=1.5)
        detectors['iqr'].fit(X_train)
        training_times['iqr'] = time.time() - start
        print(f"  ✅ Trained in {training_times['iqr']:.2f}s")
        
        # 7. Statistical: Modified Z-Score
        print("\n[7/10] Statistical Method: Modified Z-Score")
        start = time.time()
        detectors['modified_zscore'] = StatisticalAnomalyDetector(method='modified_zscore', threshold=3.5)
        detectors['modified_zscore'].fit(X_train)
        training_times['modified_zscore'] = time.time() - start
        print(f"  ✅ Trained in {training_times['modified_zscore']:.2f}s")
        
        # 8. DBSCAN (Density-Based)
        print("\n[8/10] DBSCAN (Density-Based Clustering)")
        start = time.time()
        # DBSCAN doesn't have predict, so we'll fit on train and mark outliers
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        train_labels = dbscan.fit_predict(X_train)
        # Count outliers (label -1)
        n_outliers = np.sum(train_labels == -1)
        print(f"  Found {n_outliers} outliers in training data ({n_outliers/len(train_labels)*100:.2f}%)")
        training_times['dbscan'] = time.time() - start
        print(f"  ℹ️  DBSCAN trained in {training_times['dbscan']:.2f}s (not used for test prediction)")
        
        # 9. Autoencoder (Deep Learning)
        if KERAS_AVAILABLE and X_train.shape[0] > 1000:
            print("\n[9/10] Autoencoder (Deep Learning)")
            start = time.time()
            try:
                encoding_dim = min(32, X_train.shape[1] // 2)
                detectors['autoencoder'] = AutoencoderAnomalyDetector(
                    encoding_dim=encoding_dim,
                    epochs=50,
                    batch_size=256,
                    contamination=contamination
                )
                detectors['autoencoder'].fit(X_train)
                training_times['autoencoder'] = time.time() - start
                print(f"  ✅ Trained in {training_times['autoencoder']:.2f}s")
            except Exception as e:
                print(f"  ⚠️ Autoencoder failed: {e}")
                training_times['autoencoder'] = 0
        else:
            print("\n[9/10] Autoencoder - SKIPPED")
            print(f"  ℹ️  TensorFlow available: {KERAS_AVAILABLE}, Samples: {X_train.shape[0]}")
            training_times['autoencoder'] = 0
        
        # 10. Ensemble (Voting)
        print("\n[10/10] Ensemble Method (Majority Voting)")
        start = time.time()
        # Select top 3 detectors for ensemble
        ensemble_detectors = {
            'isolation_forest': detectors['isolation_forest'],
            'one_class_svm': detectors['one_class_svm'],
            'lof': detectors['lof']
        }
        detectors['ensemble_voting'] = EnsembleAnomalyDetector(
            ensemble_detectors,
            method='majority_vote'
        )
        # Already fitted during individual training
        training_times['ensemble_voting'] = time.time() - start
        print(f"  ✅ Ensemble ready in {training_times['ensemble_voting']:.2f}s")
        
        print(f"\n{'=' * 80}")
        print(f"All {len(detectors)} detectors trained!")
        print(f"{'=' * 80}")
        
        # =====================================================================
        # STEP 3: EVALUATE ALL DETECTORS
        # =====================================================================
        if y_true is not None:
            print("\n" + "=" * 80)
            print("STEP 3: COMPREHENSIVE EVALUATION")
            print("=" * 80)
            
            all_results = {}
            
            for detector_name, detector in detectors.items():
                if detector_name == 'dbscan':
                    continue  # Skip DBSCAN (no predict method)
                
                metrics, cm = evaluate_detector(detector, X_test, y_true, detector_name)
                metrics['training_time_seconds'] = training_times.get(detector_name, 0)
                all_results[detector_name] = metrics
                
                # Log to MLflow
                for metric_name, value in metrics.items():
                    if isinstance(value, (int, float)):
                        mlflow.log_metric(f'{detector_name}_{metric_name}', value)
            
            # =====================================================================
            # STEP 4: SELECT BEST MODEL
            # =====================================================================
            print("\n" + "=" * 80)
            print("STEP 4: MODEL SELECTION & RANKING")
            print("=" * 80)
            
            # Rank by F1 score
            ranked = sorted(all_results.items(), key=lambda x: x[1]['f1_score'], reverse=True)
            
            print("\n📊 MODEL PERFORMANCE RANKING (by F1 Score):")
            print(f"{'Rank':<6} {'Model':<25} {'F1':<8} {'Precision':<10} {'Recall':<8} {'ROC-AUC':<8}")
            print("-" * 80)
            
            for i, (name, metrics) in enumerate(ranked, 1):
                print(f"{i:<6} {name:<25} {metrics['f1_score']:<8.4f} "
                      f"{metrics['precision']:<10.4f} {metrics['recall']:<8.4f} "
                      f"{metrics.get('roc_auc', 0):<8.4f}")
            
            best_model_name = ranked[0][0]
            best_model = detectors[best_model_name]
            best_metrics = ranked[0][1]
            
            print(f"\n🏆 BEST MODEL: {best_model_name}")
            print(f"   F1 Score: {best_metrics['f1_score']:.4f}")
            print(f"   Precision: {best_metrics['precision']:.4f}")
            print(f"   Recall: {best_metrics['recall']:.4f}")
            
        else:
            print("\n⚠️ No ground truth labels - skipping evaluation")
            best_model_name = 'isolation_forest'
            best_model = detectors[best_model_name]
            best_metrics = {'training_time_seconds': training_times[best_model_name]}
            all_results = {name: {'training_time_seconds': t} for name, t in training_times.items()}
        
        # =====================================================================
        # STEP 5: SAVE BEST MODEL & ARTIFACTS
        # =====================================================================
        print("\n" + "=" * 80)
        print("STEP 5: SAVING MODELS & ARTIFACTS")
        print("=" * 80)
        
        # Use correct path: ml_models/models/anomaly/ (from project root)
        project_root = Path(__file__).parent.parent.parent.parent
        save_path = project_root / 'ml_models' / 'models' / 'anomaly' / machine_id
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save best model
        model_file = save_path / f'{best_model_name}.pkl'
        joblib.dump({
            'model': best_model,
            'scaler': scaler,
            'imputer': imputer
        }, model_file)
        
        model_size_mb = model_file.stat().st_size / (1024 * 1024)
        print(f"  ✅ Best model saved: {model_file}")
        print(f"     Size: {model_size_mb:.2f} MB")
        
        # Save all models (optional)
        all_models_file = save_path / 'all_detectors.pkl'
        joblib.dump({
            'detectors': {k: v for k, v in detectors.items() if k != 'dbscan'},
            'scaler': scaler,
            'imputer': imputer
        }, all_models_file)
        
        all_models_size = all_models_file.stat().st_size / (1024 * 1024)
        print(f"  ✅ All detectors saved: {all_models_file}")
        print(f"     Size: {all_models_size:.2f} MB")
        
        # Save feature names
        feature_file = save_path / 'features.json'
        with open(feature_file, 'w') as f:
            json.dump({'features': feature_cols}, f, indent=2)
        print(f"  ✅ Features saved: {len(feature_cols)} features")
        
        # Save preprocessing artifacts
        preprocessing_file = save_path / 'preprocessing.pkl'
        joblib.dump({
            'imputer': imputer,
            'scaler': scaler,
            'feature_cols': feature_cols
        }, preprocessing_file)
        print(f"  ✅ Preprocessing saved: {preprocessing_file}")
        
        # =====================================================================
        # STEP 6: GENERATE COMPREHENSIVE REPORT
        # =====================================================================
        print("\n" + "=" * 80)
        print("STEP 6: GENERATING COMPREHENSIVE REPORT")
        print("=" * 80)
        
        total_time = time.time() - overall_start
        
        report = {
            'machine_id': machine_id,
            'task_type': 'comprehensive_anomaly_detection',
            'timestamp': datetime.now().isoformat(),
            'training_time_total_minutes': total_time / 60,
            
            # Best model info
            'best_model': {
                'name': best_model_name,
                'metrics': best_metrics,
                'model_path': str(model_file),
                'model_size_mb': model_size_mb
            },
            
            # All models performance
            'all_models': all_results if y_true is not None else {},
            
            # Training times
            'training_times': training_times,
            
            # Data info
            'data_info': {
                'n_train_samples': len(X_train),
                'n_test_samples': len(X_test),
                'n_features': len(feature_cols),
                'n_anomalies_test': int(np.sum(y_true)) if y_true is not None else None
            },
            
            # Model rankings
            'model_rankings': [
                {'rank': i+1, 'model': name, 'f1_score': metrics['f1_score']}
                for i, (name, metrics) in enumerate(ranked)
            ] if y_true is not None else [],
            
            # Files saved
            'artifacts': {
                'best_model': str(model_file),
                'all_models': str(all_models_file),
                'features': str(feature_file),
                'preprocessing': str(preprocessing_file),
                'total_size_mb': model_size_mb + all_models_size
            }
        }
        
        # Save report
        report_path = project_root / 'ml_models' / 'reports' / 'performance_metrics' / f'{machine_id}_comprehensive_anomaly_report.json'
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"  ✅ Report saved: {report_path}")
        
        # MLflow logging
        mlflow.log_param('best_model', best_model_name)
        mlflow.log_metric('total_training_time_minutes', total_time / 60)
        mlflow.log_metric('model_size_mb', model_size_mb)
        mlflow.log_metric('n_detectors_trained', len(detectors))
        
        # =====================================================================
        # COMPLETION SUMMARY
        # =====================================================================
        print("\n" + "#" * 80)
        print("#  COMPREHENSIVE ANOMALY DETECTION - COMPLETED")
        print("#" * 80)
        print(f"\n  Machine: {machine_id}")
        print(f"  Total Time: {total_time/60:.2f} minutes")
        print(f"  Detectors Trained: {len(detectors)}")
        print(f"  Best Model: {best_model_name} (F1={best_metrics.get('f1_score', 0):.4f})")
        print(f"  Model Size: {model_size_mb:.2f} MB")
        print(f"  Report: {report_path}")
        print(f"\n" + "#" * 80 + "\n")
        
        return report


if __name__ == "__main__":
    import argparse
    from config.model_config import AUTOGLUON_CONFIG
    
    parser = argparse.ArgumentParser(description='Comprehensive Anomaly Detection Training')
    parser.add_argument('--machine_id', required=True, help='Machine ID')
    parser.add_argument('--contamination', type=float, default=0.1, help='Expected contamination rate')
    args = parser.parse_args()
    
    config = AUTOGLUON_CONFIG['anomaly'].copy()
    config['contamination'] = args.contamination
    
    train_comprehensive_anomaly_detection(args.machine_id, config)
