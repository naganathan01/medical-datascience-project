# data/generate_data.py - Generate realistic patient data
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def generate_patient_data(n_patients=5000):
    """Generate realistic diabetic risk assessment data"""
    
    print(f"🏥 Generating {n_patients} synthetic patient records...")
    
    np.random.seed(42)  # For reproducible results
    
    # Patient demographics
    ages = np.random.normal(45, 15, n_patients).clip(18, 90)
    
    # Gender (0=Female, 1=Male)
    gender = np.random.binomial(1, 0.5, n_patients)
    
    # BMI distribution (realistic for diabetic risk)
    bmi = np.random.normal(27, 6, n_patients).clip(15, 50)
    
    # Blood pressure (systolic)
    bp_systolic = np.random.normal(125, 20, n_patients).clip(90, 200)
    bp_diastolic = bp_systolic * 0.6 + np.random.normal(0, 5, n_patients)
    bp_diastolic = bp_diastolic.clip(60, 120)
    
    # Glucose levels (mg/dL)
    glucose = np.random.normal(95, 25, n_patients).clip(70, 300)
    
    # Insulin levels (μU/mL)
    insulin = np.random.exponential(15, n_patients).clip(2, 100)
    
    # Family history of diabetes (0=No, 1=Yes)
    family_history = np.random.binomial(1, 0.3, n_patients)
    
    # Physical activity level (1-5 scale)
    activity_level = np.random.poisson(3, n_patients).clip(1, 5)
    
    # Smoking status (0=Never, 1=Former, 2=Current)
    smoking = np.random.choice([0, 1, 2], n_patients, p=[0.6, 0.25, 0.15])
    
    # Realistic diabetic risk calculation
    risk_score = (
        (ages - 30) * 0.02 +                    # Age factor
        (bmi - 25) * 0.15 +                     # BMI factor
        (glucose - 90) * 0.05 +                 # Glucose factor
        (bp_systolic - 120) * 0.02 +            # Blood pressure factor
        np.log(insulin) * 0.3 +                 # Insulin factor
        family_history * 1.5 +                  # Family history
        (5 - activity_level) * 0.3 +            # Physical activity
        smoking * 0.4 +                         # Smoking
        gender * 0.2 +                          # Gender (males slightly higher risk)
        np.random.normal(0, 1, n_patients)      # Random variation
    )
    
    # Convert to probability and binary outcome
    risk_probability = 1 / (1 + np.exp(-risk_score + 3))  # Sigmoid function
    has_diabetes = (risk_probability > 0.5).astype(int)
    
    # Create DataFrame
    patient_data = pd.DataFrame({
        'patient_id': [f'P{i:06d}' for i in range(1, n_patients + 1)],
        'age': ages.round(0).astype(int),
        'gender': gender,  # 0=Female, 1=Male
        'bmi': bmi.round(1),
        'bp_systolic': bp_systolic.round(0).astype(int),
        'bp_diastolic': bp_diastolic.round(0).astype(int),
        'glucose': glucose.round(0).astype(int),
        'insulin': insulin.round(1),
        'family_history': family_history,
        'activity_level': activity_level,
        'smoking': smoking,
        'risk_probability': risk_probability.round(3),
        'has_diabetes': has_diabetes,
        'created_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    })
    
    # Add some data quality variations (realistic)
    # Some missing values
    missing_mask = np.random.random(n_patients) < 0.02  # 2% missing insulin
    patient_data.loc[missing_mask, 'insulin'] = np.nan
    
    return patient_data

def save_sample_datasets(data):
    """Save different datasets for different purposes"""
    
    os.makedirs('data', exist_ok=True)
    
    # Full dataset
    data.to_csv('data/patient_data.csv', index=False)
    print(f"✅ Saved full dataset: data/patient_data.csv ({len(data)} records)")
    
    # Training dataset (80%)
    train_data = data.sample(frac=0.8, random_state=42)
    train_data.to_csv('data/train_data.csv', index=False)
    print(f"✅ Saved training dataset: data/train_data.csv ({len(train_data)} records)")
    
    # Test dataset (20%)
    test_data = data.drop(train_data.index)
    test_data.to_csv('data/test_data.csv', index=False)
    print(f"✅ Saved test dataset: data/test_data.csv ({len(test_data)} records)")
    
    # Sample for API testing
    api_samples = data.sample(10, random_state=123)
    api_samples.to_csv('data/api_test_samples.csv', index=False)
    print(f"✅ Saved API test samples: data/api_test_samples.csv (10 records)")

def print_data_summary(data):
    """Print summary statistics"""
    print(f"\n📊 Dataset Summary:")
    print(f"   Total patients: {len(data)}")
    print(f"   Diabetic patients: {data['has_diabetes'].sum()} ({data['has_diabetes'].mean():.1%})")
    print(f"   Age range: {data['age'].min():.0f} - {data['age'].max():.0f} years")
    print(f"   BMI range: {data['bmi'].min():.1f} - {data['bmi'].max():.1f}")
    print(f"   Glucose range: {data['glucose'].min():.0f} - {data['glucose'].max():.0f} mg/dL")
    
    print(f"\n📈 Risk Distribution:")
    print(f"   Low risk (<30%): {(data['risk_probability'] < 0.3).sum()} patients")
    print(f"   Medium risk (30-70%): {((data['risk_probability'] >= 0.3) & (data['risk_probability'] < 0.7)).sum()} patients")
    print(f"   High risk (>70%): {(data['risk_probability'] >= 0.7).sum()} patients")

if __name__ == "__main__":
    # Generate realistic patient data
    patient_data = generate_patient_data(5000)
    
    # Save datasets
    save_sample_datasets(patient_data)
    
    # Print summary
    print_data_summary(patient_data)
    
    print(f"\n🎯 Sample patient records:")
    print(patient_data.head(3).to_string())
    
    print(f"\n✅ Data generation complete! Ready for model training.")