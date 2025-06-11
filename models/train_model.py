# models/train_model.py - Train diabetes prediction model
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
import pickle
import os
import joblib
from datetime import datetime

class DiabetesPredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = [
            'age', 'gender', 'bmi', 'bp_systolic', 'bp_diastolic', 
            'glucose', 'insulin', 'family_history', 'activity_level', 'smoking'
        ]
        self.model_metadata = {}
    
    def load_and_prepare_data(self, data_path='data/train_data.csv'):
        """Load and prepare training data"""
        print(f"📂 Loading training data from {data_path}...")
        
        # Load data
        data = pd.read_csv(data_path)
        
        # Handle missing values
        data['insulin'].fillna(data['insulin'].median(), inplace=True)
        
        # Prepare features and target
        X = data[self.feature_names]
        y = data['has_diabetes']
        
        print(f"✅ Loaded {len(data)} training samples")
        print(f"   Features: {len(self.feature_names)}")
        print(f"   Positive cases: {y.sum()} ({y.mean():.1%})")
        
        return X, y
    
    def train_model(self, X, y):
        """Train the diabetes prediction model"""
        print(f"\n🤖 Training diabetes prediction model...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train Random Forest model
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            class_weight='balanced'  # Handle imbalanced data
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        auc_score = roc_auc_score(y_test, y_pred_proba)
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=5, scoring='roc_auc')
        
        print(f"✅ Model training completed!")
        print(f"   Accuracy: {accuracy:.3f}")
        print(f"   AUC Score: {auc_score:.3f}")
        print(f"   Cross-validation AUC: {cv_scores.mean():.3f} (±{cv_scores.std():.3f})")
        
        # Detailed classification report
        print(f"\n📋 Classification Report:")
        print(classification_report(y_test, y_pred, target_names=['No Diabetes', 'Diabetes']))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print(f"\n🔢 Confusion Matrix:")
        print(f"   True Negatives: {cm[0,0]}, False Positives: {cm[0,1]}")
        print(f"   False Negatives: {cm[1,0]}, True Positives: {cm[1,1]}")
        
        # Calculate medical metrics
        sensitivity = cm[1,1] / (cm[1,1] + cm[1,0])  # Recall for diabetes
        specificity = cm[0,0] / (cm[0,0] + cm[0,1])  # Recall for no diabetes
        ppv = cm[1,1] / (cm[1,1] + cm[0,1])  # Precision for diabetes
        npv = cm[0,0] / (cm[0,0] + cm[1,0])  # Precision for no diabetes
        
        print(f"\n🏥 Medical Performance Metrics:")
        print(f"   Sensitivity (Recall): {sensitivity:.3f}")
        print(f"   Specificity: {specificity:.3f}")
        print(f"   Positive Predictive Value: {ppv:.3f}")
        print(f"   Negative Predictive Value: {npv:.3f}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\n🔍 Feature Importance:")
        for _, row in feature_importance.iterrows():
            print(f"   {row['feature']}: {row['importance']:.3f}")
        
        # Store metadata
        self.model_metadata = {
            'training_date': datetime.now().isoformat(),
            'accuracy': accuracy,
            'auc_score': auc_score,
            'cv_auc_mean': cv_scores.mean(),
            'cv_auc_std': cv_scores.std(),
            'sensitivity': sensitivity,
            'specificity': specificity,
            'ppv': ppv,
            'npv': npv,
            'n_training_samples': len(X_train),
            'n_test_samples': len(X_test),
            'feature_importance': feature_importance.to_dict('records')
        }
        
        return self.model
    
    def save_model(self, model_path='models/diabetes_model.pkl'):
        """Save the trained model and scaler"""
        os.makedirs('models', exist_ok=True)
        
        # Save model and scaler together
        model_package = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'metadata': self.model_metadata
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_package, f)
        
        print(f"💾 Model saved to {model_path}")
        
        # Also save metadata as JSON for easy reading
        import json
        metadata_path = model_path.replace('.pkl', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(self.model_metadata, f, indent=2)
        
        print(f"📄 Model metadata saved to {metadata_path}")
    
    def predict(self, patient_features):
        """Make prediction for a single patient"""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        # Ensure features are in correct order
        features_array = np.array([patient_features[name] for name in self.feature_names]).reshape(1, -1)
        
        # Scale features
        features_scaled = self.scaler.transform(features_array)
        
        # Make prediction
        prediction = self.model.predict(features_scaled)[0]
        probability = self.model.predict_proba(features_scaled)[0]
        
        return {
            'has_diabetes': bool(prediction),
            'diabetes_probability': float(probability[1]),
            'no_diabetes_probability': float(probability[0])
        }

def test_model_predictions():
    """Test model with sample patients"""
    print(f"\n🧪 Testing model with sample patients...")
    
    # Load model
    with open('models/diabetes_model.pkl', 'rb') as f:
        model_package = pickle.load(f)
    
    predictor = DiabetesPredictor()
    predictor.model = model_package['model']
    predictor.scaler = model_package['scaler']
    predictor.feature_names = model_package['feature_names']
    
    # Test cases
    test_patients = [
        {
            'name': 'High Risk Patient',
            'age': 65, 'gender': 1, 'bmi': 32.5, 'bp_systolic': 150, 'bp_diastolic': 95,
            'glucose': 180, 'insulin': 45.0, 'family_history': 1, 'activity_level': 1, 'smoking': 2
        },
        {
            'name': 'Low Risk Patient', 
            'age': 25, 'gender': 0, 'bmi': 22.0, 'bp_systolic': 110, 'bp_diastolic': 70,
            'glucose': 85, 'insulin': 8.0, 'family_history': 0, 'activity_level': 4, 'smoking': 0
        },
        {
            'name': 'Medium Risk Patient',
            'age': 45, 'gender': 1, 'bmi': 27.5, 'bp_systolic': 130, 'bp_diastolic': 85,
            'glucose': 110, 'insulin': 20.0, 'family_history': 1, 'activity_level': 2, 'smoking': 1
        }
    ]
    
    for patient in test_patients:
        name = patient.pop('name')
        result = predictor.predict(patient)
        
        print(f"\n👤 {name}:")
        print(f"   Risk: {'HIGH RISK' if result['has_diabetes'] else 'LOW RISK'}")
        print(f"   Diabetes Probability: {result['diabetes_probability']:.1%}")
        
        if result['diabetes_probability'] > 0.7:
            print(f"   🚨 Recommendation: Immediate diabetes screening and lifestyle intervention")
        elif result['diabetes_probability'] > 0.3:
            print(f"   ⚠️  Recommendation: Regular monitoring and preventive care")
        else:
            print(f"   ✅ Recommendation: Continue routine health checkups")

if __name__ == "__main__":
    # Initialize predictor
    predictor = DiabetesPredictor()
    
    # Load and prepare data
    X, y = predictor.load_and_prepare_data()
    
    # Train model
    predictor.train_model(X, y)
    
    # Save model
    predictor.save_model()
    
    # Test predictions
    test_model_predictions()
    
    print(f"\n🎉 Model training complete! Ready for API deployment.")