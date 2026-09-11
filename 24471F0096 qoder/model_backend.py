import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import pyswarms as ps
import warnings
import joblib
import json
from flask import Flask, request, jsonify

warnings.filterwarnings('ignore')

class CervicalCancerPredictor:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_selector = None
        self.label_encoder = None
        self.selected_features = None
        self.feature_names = None
        
    def load_and_preprocess_data(self, data_path):
        """Load and preprocess the cervical cancer dataset"""
        df = pd.read_csv(data_path)
        
        # Clean column names by stripping whitespace
        df.columns = df.columns.str.strip()
        
        # Replace '?' with NaN
        df.replace('?', np.nan, inplace=True)
        
        # Drop 'Unnamed: 0' column if it exists
        if 'Unnamed: 0' in df.columns:
            df.drop('Unnamed: 0', axis=1, inplace=True)
        
        # Encode the target variable 'Category'
        self.label_encoder = LabelEncoder()
        df['Category_Encoded'] = self.label_encoder.fit_transform(df['Category'])
        y = df['Category_Encoded']
        
        # Prepare features
        categorical_cols_to_encode_X = ['Sex']
        categorical_cols_to_encode_X = [col for col in categorical_cols_to_encode_X if col in df.columns]
        
        cols_to_drop_from_X = ['Category', 'Category_Encoded']
        cols_to_drop_from_X = [col for col in cols_to_drop_from_X if col in df.columns]
        
        if categorical_cols_to_encode_X:
            X = pd.get_dummies(df.drop(columns=cols_to_drop_from_X, errors='ignore'), 
                              columns=categorical_cols_to_encode_X, drop_first=True)
        else:
            X = df.drop(columns=cols_to_drop_from_X, errors='ignore')
        
        # Convert all remaining columns to numeric
        X = X.apply(pd.to_numeric, errors='coerce')
        
        # Fill missing values with median
        X.fillna(X.median(), inplace=True)
        
        self.feature_names = X.columns.tolist()
        
        return X, y
    
    def train_model(self, data_path):
        """Train the cervical cancer prediction model"""
        # Load and preprocess data
        X, y = self.load_and_preprocess_data(data_path)
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Apply SMOTE for handling imbalanced data
        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        
        # Scale the features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Feature selection using Particle Swarm Optimization (PSO)
        def pso_objective_function(particles):
            scores = []
            for particle in particles:
                mask = particle > 0.5
                if np.sum(mask) == 0:
                    scores.append(1)
                    continue

                X_selected = X_train_scaled[:, mask]

                model = RandomForestClassifier(
                    n_estimators=200,
                    max_depth=10,
                    random_state=42,
                    n_jobs=-1
                )
                model.fit(X_selected, y_train)
                acc = model.score(X_selected, y_train)

                scores.append(1 - acc)
            return np.array(scores)

        options = {'c1': 1.5, 'c2': 1.5, 'w': 0.7}
        optimizer = ps.single.GlobalBestPSO(
            n_particles=20,
            dimensions=X_train_scaled.shape[1],
            options=options
        )

        cost, best_pos = optimizer.optimize(pso_objective_function, iters=30)
        self.selected_features = best_pos > 0.5

        X_train_pso = X_train_scaled[:, self.selected_features]
        X_test_pso = X_test_scaled[:, self.selected_features]

        print("Number of Selected Features:", np.sum(self.selected_features))
        
        # Train the final model
        self.model = RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            random_state=42,
            n_jobs=-1
        )
        
        self.model.fit(X_train_pso, y_train)
        
        # Evaluate the model
        y_pred = self.model.predict(X_test_pso)
        accuracy = accuracy_score(y_test, y_pred)
        print("Final Accuracy:", accuracy * 100)
        print("\nClassification Report:\n")
        print(classification_report(y_test, y_pred, 
                                  target_names=self.label_encoder.classes_))
        print("\nConfusion Matrix:\n")
        print(confusion_matrix(y_test, y_pred))
        
        return accuracy
    
    def predict(self, patient_data):
        """Predict the cervical cancer risk category for a patient"""
        if self.model is None or self.scaler is None:
            raise ValueError("Model not trained. Call train_model() first.")
        
        # Convert patient data to DataFrame
        patient_df = pd.DataFrame([patient_data])
        
        # Ensure all required features are present
        for feature in self.feature_names:
            if feature not in patient_df.columns:
                patient_df[feature] = 0  # Default value for missing features
        
        # Reorder columns to match training data
        patient_df = patient_df[self.feature_names]
        
        # Scale the features
        patient_scaled = self.scaler.transform(patient_df)
        
        # Select features using the same selection as training
        patient_selected = patient_scaled[:, self.selected_features]
        
        # Make prediction
        prediction = self.model.predict(patient_selected)[0]
        prediction_proba = self.model.predict_proba(patient_selected)[0]
        
        # Convert prediction back to original label
        predicted_label = self.label_encoder.inverse_transform([prediction])[0]
        
        return {
            'predicted_category': predicted_label,
            'predicted_encoded': int(prediction),
            'probabilities': {label: float(prob) 
                            for label, prob in zip(self.label_encoder.classes_, prediction_proba)}
        }
    
    def save_model(self, filepath):
        """Save the trained model to a file"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'selected_features': self.selected_features,
            'label_encoder': self.label_encoder,
            'feature_names': self.feature_names
        }
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a trained model from a file"""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.selected_features = model_data['selected_features']
        self.label_encoder = model_data['label_encoder']
        self.feature_names = model_data['feature_names']
        print(f"Model loaded from {filepath}")


# Flask API for the model
app = Flask(__name__)
predictor = CervicalCancerPredictor()

# You can load a pre-trained model if available, or train a new one
# For now, we'll create a placeholder for training
MODEL_PATH = 'cervical_cancer_model.pkl'

@app.route('/predict', methods=['POST'])
def predict_risk():
    try:
        # Get patient data from request
        patient_data = request.json
        
        # Validate required fields
        required_fields = ['Age', 'Sex', 'ALB', 'ALP', 'ALT', 'AST', 'BIL', 'CHE', 'CHOL', 'CREA', 'GGT', 'PROT']
        for field in required_fields:
            if field not in patient_data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Make prediction
        result = predictor.predict(patient_data)
        
        return jsonify({
            'success': True,
            'prediction': result
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/train', methods=['POST'])
def train_model():
    try:
        # Get data path from request
        data = request.json
        data_path = data.get('data_path', 'Cervical_Cancer.csv')
        
        # Train the model
        accuracy = predictor.train_model(data_path)
        
        # Save the trained model
        predictor.save_model(MODEL_PATH)
        
        return jsonify({
            'success': True,
            'message': 'Model trained successfully',
            'accuracy': accuracy
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'model_loaded': predictor.model is not None})

if __name__ == '__main__':
    # For testing purposes, you can train the model when running this script
    # Uncomment the following lines if you want to train the model immediately
    # predictor.train_model('Cervical_Cancer.csv')
    # predictor.save_model('cervical_cancer_model.pkl')
    
    print("Starting Cervical Cancer Prediction API...")
    print("Available endpoints:")
    print("  POST /train - Train the model with CSV data")
    print("  POST /predict - Make a prediction")
    print("  GET  /health - Health check")
    
    app.run(debug=True, host='0.0.0.0', port=5000)