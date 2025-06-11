# api/app.py - Complete CORS-enabled Flask API with Fixed Dashboard Routes
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
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

# 🔧 CORS Configuration
CORS(app, origins=["*"], methods=["GET", "POST", "OPTIONS"], 
     allow_headers=["Content-Type", "Authorization"])

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
            # Try multiple possible paths for the model
            possible_paths = [
                os.path.join('..', 'models', 'diabetes_model.pkl'),
                'models/diabetes_model.pkl',
                os.path.join('models', 'diabetes_model.pkl'),
                'diabetes_model.pkl'
            ]
            
            model_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    model_path = path
                    break
            
            if model_path is None:
                raise FileNotFoundError("Model file not found in any expected location")
            
            with open(model_path, 'rb') as f:
                model_package = pickle.load(f)
            
            self.model = model_package['model']
            self.scaler = model_package['scaler']
            self.feature_names = model_package['feature_names']
            self.metadata = model_package['metadata']
            
            logger.info("✅ Diabetes prediction model loaded successfully")
            logger.info(f"   Model path: {model_path}")
            logger.info(f"   Model accuracy: {self.metadata.get('accuracy', 'unknown'):.3f}")
            logger.info(f"   Model AUC: {self.metadata.get('auc_score', 'unknown'):.3f}")
            logger.info(f"   Features: {self.feature_names}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load model: {str(e)}")
            logger.error(f"   Current working directory: {os.getcwd()}")
            logger.error(f"   Files in current directory: {os.listdir('.')}")
            raise
    
    def validate_input(self, data):
        """Validate input data"""
        if not self.feature_names:
            return ["Model not properly loaded"]
            
        required_fields = self.feature_names
        errors = []
        
        # Check for missing fields
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
            
            # Prepare features in the correct order
            features = [float(patient_data[name]) for name in self.feature_names]
            features_array = np.array(features).reshape(1, -1)
            
            # Scale features
            features_scaled = self.scaler.transform(features_array)
            
            # Make prediction
            prediction = self.model.predict(features_scaled)[0]
            probabilities = self.model.predict_proba(features_scaled)[0]
            
            # Calculate risk level
            diabetes_prob = float(probabilities[1])
            no_diabetes_prob = float(probabilities[0])
            
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
                    'no_diabetes_probability': round(no_diabetes_prob, 3),
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
                },
                'input_data': patient_data
            }
            
            logger.info(f"Prediction made: Risk={risk_level}, Probability={diabetes_prob:.3f}")
            return result, 200
            
        except Exception as e:
            error_counter.inc()
            logger.error(f"Prediction error: {str(e)}")
            logger.error(traceback.format_exc())
            return {'error': 'Internal prediction error', 'details': str(e)}, 500
    
    def _get_lifestyle_recommendations(self, patient_data, diabetes_prob):
        """Generate personalized lifestyle recommendations"""
        recommendations = []
        
        try:
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
            
            # General recommendations based on risk level
            if diabetes_prob > 0.5:
                recommendations.append("Regular health checkups: Schedule quarterly medical consultations")
            
        except Exception as e:
            logger.warning(f"Error generating recommendations: {str(e)}")
            recommendations.append("Consult healthcare provider for personalized recommendations")
        
        return recommendations if recommendations else ["Maintain current healthy lifestyle"]

# Initialize API
try:
    api = DiabetesPredictionAPI()
    logger.info("✅ API initialized successfully")
except Exception as e:
    logger.error(f"❌ Failed to initialize API: {str(e)}")
    api = None

# 🎯 FIXED DASHBOARD ROUTES - These are the corrected routes
@app.route('/')
def home():
    """Home route - serve dashboard from current directory"""
    try:
        logger.info(f"🔍 Looking for dashboard.html in: {os.getcwd()}")
        logger.info(f"   Files in current directory: {os.listdir('.')}")
        
        # Check if dashboard.html exists in current directory (api folder)
        if os.path.exists('dashboard.html'):
            logger.info("✅ Found dashboard.html in current directory")
            return send_from_directory('.', 'dashboard.html')
        
        # If not found, return API info
        logger.warning("⚠️  dashboard.html not found in current directory")
        return jsonify({
            'message': 'Diabetes Prediction API',
            'version': '1.0.0',
            'status': 'running',
            'endpoints': {
                'health': '/health',
                'predict': '/predict', 
                'model_info': '/model/info',
                'batch_predict': '/predict/batch',
                'metrics': '/metrics',
                'debug': '/debug/files'
            },
            'current_directory': os.getcwd(),
            'files_in_current_dir': os.listdir('.'),
            'dashboard_found': False,
            'note': 'Place dashboard.html in the api/ folder (same as app.py)'
        })
        
    except Exception as e:
        logger.error(f"Error serving home route: {str(e)}")
        return jsonify({'error': f'Unable to serve dashboard: {str(e)}'}), 500

@app.route('/dashboard')
def dashboard_route():
    """Alternative dashboard route"""
    logger.info("🔍 Dashboard route accessed")
    return home()

@app.route('/debug/files')
def debug_files():
    """Debug endpoint to see file structure"""
    try:
        current_dir = os.getcwd()
        
        debug_info = {
            'current_directory': current_dir,
            'files_in_current_dir': os.listdir('.'),
            'dashboard_exists': os.path.exists('dashboard.html'),
            'dashboard_path': os.path.abspath('dashboard.html') if os.path.exists('dashboard.html') else None,
            'parent_directory': os.path.dirname(current_dir),
            'files_in_parent': os.listdir('..') if os.path.exists('..') else []
        }
        
        # Check for dashboard in parent directory too
        parent_dashboard = os.path.join('..', 'dashboard.html')
        if os.path.exists(parent_dashboard):
            debug_info['dashboard_in_parent'] = True
            debug_info['parent_dashboard_path'] = os.path.abspath(parent_dashboard)
        else:
            debug_info['dashboard_in_parent'] = False
        
        return jsonify(debug_info)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'diabetes-prediction-api',
        'version': '1.0.0',
        'model_loaded': api is not None and api.model is not None,
        'cors_enabled': True,
        'timestamp': datetime.utcnow().isoformat(),
        'current_directory': os.getcwd(),
        'dashboard_exists': os.path.exists('dashboard.html'),
        'endpoints': ['/health', '/predict', '/model/info', '/predict/batch', '/metrics', '/debug/files']
    }), 200

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict_diabetes():
    """Diabetes risk prediction endpoint"""
    # Handle preflight requests
    if request.method == 'OPTIONS':
        return '', 200
    
    if api is None or api.model is None:
        return jsonify({'error': 'Model not loaded properly'}), 503
    
    try:
        # Get JSON data
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        logger.info(f"Received prediction request: {data}")
        
        # Make prediction
        result, status_code = api.predict(data)
        return jsonify(result), status_code
        
    except Exception as e:
        logger.error(f"API error: {str(e)}")
        logger.error(traceback.format_exc())
        error_counter.inc()
        return jsonify({'error': 'Internal server error', 'details': str(e)}), 500

@app.route('/model/info', methods=['GET'])
def model_info():
    """Get model information and metadata"""
    if api is None or api.metadata is None:
        return jsonify({'error': 'Model metadata not available'}), 404
    
    return jsonify({
        'model_metadata': api.metadata,
        'feature_names': api.feature_names,
        'model_type': 'Random Forest Classifier',
        'use_case': 'Diabetes Risk Assessment',
        'feature_count': len(api.feature_names) if api.feature_names else 0,
        'model_loaded': api.model is not None
    }), 200

@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    """Batch prediction endpoint"""
    if api is None or api.model is None:
        return jsonify({'error': 'Model not loaded properly'}), 503
    
    try:
        data = request.get_json()
        
        if not data or 'patients' not in data:
            return jsonify({'error': 'Expected JSON with "patients" array'}), 400
        
        patients = data['patients']
        if not isinstance(patients, list):
            return jsonify({'error': '"patients" must be an array'}), 400
        
        results = []
        successful_predictions = 0
        
        for i, patient in enumerate(patients):
            try:
                result, status_code = api.predict(patient)
                results.append({
                    'patient_index': i,
                    'prediction': result,
                    'status': 'success' if status_code == 200 else 'error'
                })
                if status_code == 200:
                    successful_predictions += 1
            except Exception as e:
                results.append({
                    'patient_index': i,
                    'prediction': {'error': str(e)},
                    'status': 'error'
                })
        
        return jsonify({
            'batch_results': results,
            'total_patients': len(patients),
            'successful_predictions': successful_predictions,
            'failed_predictions': len(patients) - successful_predictions,
            'timestamp': datetime.utcnow().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        return jsonify({'error': 'Batch prediction failed', 'details': str(e)}), 500

@app.route('/metrics', methods=['GET'])
def metrics():
    """Prometheus metrics endpoint"""
    try:
        return Response(generate_latest(), mimetype='text/plain')
    except Exception as e:
        logger.error(f"Metrics error: {str(e)}")
        return jsonify({'error': 'Metrics not available'}), 500

# Error handlers
@app.errorhandler(404)
def not_found(error):
    logger.warning(f"404 error: {request.url}")
    return jsonify({
        'error': 'Endpoint not found',
        'requested_url': request.url,
        'available_endpoints': ['/health', '/predict', '/model/info', '/predict/batch', '/metrics', '/debug/files'],
        'dashboard_status': 'dashboard.html exists' if os.path.exists('dashboard.html') else 'dashboard.html not found'
    }), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {str(error)}")
    return jsonify({'error': 'Internal server error'}), 500

@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({'error': 'Method not allowed'}), 405

if __name__ == '__main__':
    logger.info("🚀 Starting Diabetes Prediction API with CORS enabled...")
    logger.info(f"   Current directory: {os.getcwd()}")
    logger.info(f"   Files in current directory: {os.listdir('.')}")
    logger.info(f"   Dashboard exists: {os.path.exists('dashboard.html')}")
    logger.info(f"   Health check: http://localhost:5000/health")
    logger.info(f"   Prediction: http://localhost:5000/predict")
    logger.info(f"   Model info: http://localhost:5000/model/info")
    logger.info(f"   Debug info: http://localhost:5000/debug/files")
    logger.info(f"   Dashboard: http://localhost:5000/ or http://localhost:5000/dashboard")
    logger.info(f"   🔧 CORS enabled for all origins (development mode)")
    
    if api is None:
        logger.warning("⚠️  API starting without model - check model file location")
    
    app.run(host='0.0.0.0', port=5000, debug=False)