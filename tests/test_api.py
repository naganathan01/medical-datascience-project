# tests/test_api.py - Comprehensive API tests
import requests
import json
import pytest
import time

# Test configuration
API_BASE_URL = "http://localhost:5000"

class TestDiabetesAPI:
    
    def test_health_check(self):
        """Test health check endpoint"""
        response = requests.get(f"{API_BASE_URL}/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data['status'] == 'healthy'
        assert 'timestamp' in data
        print("✅ Health check passed")
    
    def test_single_prediction_low_risk(self):
        """Test prediction for low-risk patient"""
        patient_data = {
            "age": 25,
            "gender": 0,
            "bmi": 22.0,
            "bp_systolic": 110,
            "bp_diastolic": 70,
            "glucose": 85,
            "insulin": 8.0,
            "family_history": 0,
            "activity_level": 4,
            "smoking": 0
        }
        
        response = requests.post(
            f"{API_BASE_URL}/predict",
            json=patient_data,
            headers={'Content-Type': 'application/json'}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert 'prediction' in data
        assert 'recommendations' in data
        assert data['prediction']['risk_level'] in ['LOW', 'MEDIUM', 'HIGH']
        print(f"✅ Low-risk prediction: {data['prediction']['risk_level']}")
    
    def test_single_prediction_high_risk(self):
        """Test prediction for high-risk patient"""
        patient_data = {
            "age": 65,
            "gender": 1,
            "bmi": 32.5,
            "bp_systolic": 150,
            "bp_diastolic": 95,
            "glucose": 180,
            "insulin": 45.0,
            "family_history": 1,
            "activity_level": 1,
            "smoking": 2
        }
        
        response = requests.post(
            f"{API_BASE_URL}/predict",
            json=patient_data,
            headers={'Content-Type': 'application/json'}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data['prediction']['risk_level'] in ['MEDIUM', 'HIGH']
        print(f"✅ High-risk prediction: {data['prediction']['risk_level']}")
    
    def test_batch_prediction(self):
        """Test batch prediction endpoint"""
        batch_data = {
            "patients": [
                {
                    "age": 30, "gender": 0, "bmi": 23.0, "bp_systolic": 115,
                    "bp_diastolic": 75, "glucose": 90, "insulin": 10.0,
                    "family_history": 0, "activity_level": 3, "smoking": 0
                },
                {
                    "age": 55, "gender": 1, "bmi": 28.0, "bp_systolic": 140,
                    "bp_diastolic": 90, "glucose": 130, "insulin": 25.0,
                    "family_history": 1, "activity_level": 2, "smoking": 1
                }
            ]
        }
        
        response = requests.post(
            f"{API_BASE_URL}/predict/batch",
            json=batch_data,
            headers={'Content-Type': 'application/json'}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert 'batch_results' in data
        assert len(data['batch_results']) == 2
        print("✅ Batch prediction passed")
    
    def test_model_info(self):
        """Test model information endpoint"""
        response = requests.get(f"{API_BASE_URL}/model/info")
        assert response.status_code == 200
        
        data = response.json()
        assert 'model_metadata' in data
        assert 'feature_names' in data
        print("✅ Model info endpoint passed")
    
    def test_invalid_input(self):
        """Test API with invalid input"""
        invalid_data = {
            "age": "invalid",  # Should be number
            "bmi": 999  # Out of range
        }
        
        response = requests.post(
            f"{API_BASE_URL}/predict",
            json=invalid_data,
            headers={'Content-Type': 'application/json'}
        )
        
        assert response.status_code == 400
        data = response.json()
        assert 'error' in data
        print("✅ Invalid input handling passed")
    
    def test_metrics_endpoint(self):
        """Test Prometheus metrics endpoint"""
        response = requests.get(f"{API_BASE_URL}/metrics")
        assert response.status_code == 200
        assert 'diabetes_predictions_total' in response.text
        print("✅ Metrics endpoint passed")

def run_load_test():
    """Simple load test"""
    print("\n🔥 Running load test...")
    
    patient_data = {
        "age": 45, "gender": 1, "bmi": 27.0, "bp_systolic": 130,
        "bp_diastolic": 85, "glucose": 105, "insulin": 15.0,
        "family_history": 0, "activity_level": 3, "smoking": 0
    }
    
    start_time = time.time()
    successful_requests = 0
    
    for i in range(100):
        try:
            response = requests.post(
                f"{API_BASE_URL}/predict",
                json=patient_data,
                headers={'Content-Type': 'application/json'},
                timeout=5
            )
            if response.status_code == 200:
                successful_requests += 1
        except Exception as e:
            print(f"Request {i} failed: {e}")
    
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"✅ Load test completed:")
    print(f"   Total requests: 100")
    print(f"   Successful: {successful_requests}")
    print(f"   Failed: {100 - successful_requests}")
    print(f"   Duration: {duration:.2f} seconds")
    print(f"   Requests/second: {100/duration:.2f}")

if __name__ == "__main__":
    print("🧪 Testing Diabetes Prediction API...")
    
    # Wait for API to be ready
    time.sleep(2)
    
    # Run tests
    test_api = TestDiabetesAPI()
    
    try:
        test_api.test_health_check()
        test_api.test_single_prediction_low_risk()
        test_api.test_single_prediction_high_risk()
        test_api.test_batch_prediction()
        test_api.test_model_info()
        test_api.test_invalid_input()
        test_api.test_metrics_endpoint()
        
        # Run load test
        run_load_test()
        
        print("\n🎉 All tests passed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")