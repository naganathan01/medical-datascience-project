# api/app.py - Production Flask API for diabetes prediction
from flask import Flask, request, jsonify
import pickle
import numpy as np
import logging
from datetime import datetime
import os
import traceback
from prometheus_client import Counter, Histogram, generate_latest
from flask import Response
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Prometheus metrics
prediction_counter = Counter('diabetes_predictions_total', 'Total diabetes predictions made')
prediction_latency = Histogram('diabetes_prediction_duration_seconds', 'Diabetes prediction latency')
high_risk_predictions = Counter('high_risk_diabetes_predictions_total', 'High risk diabetes predictions')
error_counter = Counter('diabetes_prediction_errors_total', 'Diabetes prediction errors')

app = Flask(__name__)

class DiabetesPredictionAPI:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.metadata = None
        self.load_model()
    
    def load_model(self):
        """Load the trained diabetes prediction model"""
        try:
            model_path = os.path.join('..', 'models', 'diabetes_model.pkl')
            if not os.path.exists(model_path):
                model_path = 'models/diabetes_model.pkl'  # Alternative path
            
            with open(model_path, 'rb') as f:
                model_package = pickle.load(f)
            
            self.model = model_package['model']
            self.scaler = model_package['scaler']
            self.feature_names = model_package['feature_names']
            self.metadata = model_package['metadata']
            
            logger.info("✅ Diabetes prediction model loaded successfully")
            logger.info(f"   Model accuracy: {self.metadata.get('accuracy', 'unknown'):.3f}")
            logger.info(f"   Model AUC: {self.metadata.get('auc_score', 'unknown'):.3f}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load model: {str(e)}")
            raise
    
    def validate_input(self, data):
        """Validate input data"""
        required_fields = self.feature_names
        errors = []
        
        for field in required_fields:
            if field not in data:
                errors.append(f"Missing required field: {field}")
        
        # Validate data types and ranges
        validations = {
            'age': (18, 100),
            'gender': (0, 1),
            'bmi': (15, 50),
            'bp_systolic': (90, 200),
            'bp_diastolic': (60, 120),
            'glucose': (70, 300),
            'insulin': (2, 100),
            'family_history': (0, 1),
            'activity_level': (1, 5),
            'smoking': (0, 2)
        }
        
        for field, (min_val, max_val) in validations.items():
            if field in data:
                try:
                    value = float(data[field])
                    if not (min_val <= value <= max_val):
                        errors.append(f"{field} must be between {min_val} and {max_val}")
                except (ValueError, TypeError):
                    errors.append(f"{field} must be a number")
        
        return errors
    
    def predict(self, patient_data):
        """Make diabetes risk prediction"""
        start_time = time.time()
        prediction_counter.inc()
        
        try:
            # Validate input
            validation_errors = self.validate_input(patient_data)
            if validation_errors:
                error_counter.inc()
                return {'error': 'Invalid input', 'details': validation_errors}, 400
            
            # Prepare features
            features = [float(patient_data[name]) for name in self.feature_names]
            features_array = np.array(features).reshape(1, -1)
            
            # Scale features
            features_scaled = self.scaler.transform(features_array)
            
            # Make prediction
            prediction = self.model.predict(features_scaled)[0]
            probabilities = self.model.predict_proba(features_scaled)[0]
            
            # Calculate risk level
            diabetes_prob = float(probabilities[1])
            
            if diabetes_prob >= 0.7:
                risk_level = "HIGH"
                recommendation = "Immediate medical consultation recommended. Consider diabetes screening tests (HbA1c, fasting glucose)."
            elif diabetes_prob >= 0.3:
                risk_level = "MEDIUM"
                recommendation = "Regular monitoring advised. Lifestyle modifications recommended. Follow-up in 6 months."
            else:
                risk_level = "LOW"
                recommendation = "Continue healthy lifestyle. Routine screening as per guidelines."
            
            # Track high-risk predictions
            if diabetes_prob >= 0.7:
                high_risk_predictions.inc()
            
            # Record latency
            prediction_latency.observe(time.time() - start_time)
            
            result = {
                'prediction': {
                    'has_diabetes_risk': bool(prediction),
                    'diabetes_probability': round(diabetes_prob, 3),
                    'no_diabetes_probability': round(float(probabilities[0]), 3),
                    'risk_level': risk_level
                },
                'recommendations': {
                    'clinical_action': recommendation,
                    'lifestyle_factors': self._get_lifestyle_recommendations(patient_data, diabetes_prob)
                },
                'model_info': {
                    'model_accuracy': round(self.metadata.get('accuracy', 0), 3),
                    'model_auc': round(self.metadata.get('auc_score', 0), 3),
                    'prediction_timestamp': datetime.utcnow().isoformat()
                }
            }
            
            logger.info(f"Prediction made: Risk={risk_level}, Probability={diabetes_prob:.3f}")
            return result, 200
            
        except Exception as e:
            error_counter.inc()
            logger.error(f"Prediction error: {str(e)}")
            logger.error(traceback.format_exc())
            return {'error': 'Internal prediction error'}, 500
    
    def _get_lifestyle_recommendations(self, patient_data, diabetes_prob):
        """Generate personalized lifestyle recommendations"""
        recommendations = []
        
        # BMI recommendations
        bmi = float(patient_data.get('bmi', 25))
        if bmi > 30:
            recommendations.append("Weight management: Consider structured weight loss program")
        elif bmi > 25:
            recommendations.append("Weight management: Aim for gradual weight reduction")
        
        # Activity recommendations
        activity = int(patient_data.get('activity_level', 3))
        if activity < 3:
            recommendations.append("Physical activity: Increase to at least 150 minutes moderate exercise per week")
        
        # Smoking recommendations
        smoking = int(patient_data.get('smoking', 0))
        if smoking > 0:
            recommendations.append("Smoking cessation: Strongly recommended - increases diabetes risk significantly")
        
        # Blood pressure recommendations
        bp_systolic = float(patient_data.get('bp_systolic', 120))
        if bp_systolic > 140:
            recommendations.append("Blood pressure management: Monitor and manage hypertension")
        
        # Glucose recommendations
        glucose = float(patient_data.get('glucose', 90))
        if glucose > 125:
            recommendations.append("Glucose management: Monitor blood sugar levels regularly")
        
        return recommendations

# Initialize API
api = DiabetesPredictionAPI()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'diabetes-prediction-api',
        'version': '1.0.0',
        'model_loaded': api.model is not None,
        'timestamp': datetime.utcnow().isoformat()
    }), 200

@app.route('/predict', methods=['POST'])
def predict_diabetes():
    """Diabetes risk prediction endpoint"""
    try:
        # Get JSON data
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        # Make prediction
        result, status_code = api.predict(data)
        return jsonify(result), status_code
        
    except Exception as e:
        logger.error(f"API error: {str(e)}")
        error_counter.inc()
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/model/info', methods=['GET'])
def model_info():
    """Get model information and metadata"""
    if api.metadata:
        return jsonify({
            'model_metadata': api.metadata,
            'feature_names': api.feature_names,
            'model_type': 'Random Forest Classifier',
            'use_case': 'Diabetes Risk Assessment'
        }), 200
    else:
        return jsonify({'error': 'Model metadata not available'}), 404

@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    """Batch prediction endpoint"""
    try:
        data = request.get_json()
        
        if not data or 'patients' not in data:
            return jsonify({'error': 'Expected JSON with "patients" array'}), 400
        
        patients = data['patients']
        results = []
        
        for i, patient in enumerate(patients):
            result, status_code = api.predict(patient)
            results.append({
                'patient_index': i,
                'prediction': result,
                'status': 'success' if status_code == 200 else 'error'
            })
        
        return jsonify({
            'batch_results': results,
            'total_patients': len(patients),
            'timestamp': datetime.utcnow().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        return jsonify({'error': 'Batch prediction failed'}), 500

@app.route('/metrics', methods=['GET'])
def metrics():
    """Prometheus metrics endpoint"""
    return Response(generate_latest(), mimetype='text/plain')

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    logger.info("🚀 Starting Diabetes Prediction API...")
    logger.info(f"   Health check: http://localhost:5000/health")
    logger.info(f"   Prediction: http://localhost:5000/predict")
    logger.info(f"   Metrics: http://localhost:5000/metrics")
    app.run(host='0.0.0.0', port=5000, debug=False)