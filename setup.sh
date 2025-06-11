#!/bin/bash
# setup.sh - Complete project setup script

echo "🏥 Setting up Diabetes Risk Assessment System..."

# Create project structure
echo "📁 Creating project structure..."
mkdir -p data models api tests deployment monitoring

# Generate sample data
echo "📊 Generating synthetic patient data..."
cd data
python generate_data.py
cd ..

# Train model
echo "🤖 Training diabetes prediction model..."
cd models
python train_model.py
cd ..

# Copy model to API directory
echo "📦 Preparing API deployment..."
cp models/diabetes_model.pkl api/models/
cp models/diabetes_model_metadata.json api/models/

echo "✅ Setup complete!"
echo ""
echo "🚀 Next steps:"
echo "1. Start the API: cd api && python app.py"
echo "2. Test the API: cd tests && python test_api.py"
echo "3. Deploy with Docker: docker-compose -f deployment/docker-compose.yml up"
echo ""
echo "📍 API Endpoints:"
echo "   Health: http://localhost:5000/health"
echo "   Predict: http://localhost:5000/predict"
echo "   Metrics: http://localhost:5000/metrics"