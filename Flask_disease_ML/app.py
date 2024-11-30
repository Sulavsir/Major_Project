import matplotlib
# Use Agg backend to avoid GUI-related issues on macOS
matplotlib.use('Agg')

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import bcrypt
import pandas as pd
import numpy as np
from pymongo import MongoClient
import os
from datetime import datetime, timedelta
import re
from io import BytesIO
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import seaborn as sns
import io
import base64
import plotly.graph_objs as go
import jwt
from flask_jwt_extended import JWTManager, jwt_required, create_access_token, get_jwt_identity


app = Flask(__name__)
CORS(app)  

SECRET_KEY = 'fksdjhfagaldgfasjhfdg'

# MongoDB connection string
MONGO_URI = "mongodb+srv://admin:admin@newapis.olpnfrw.mongodb.net/DiseasePredictions"
client = MongoClient(MONGO_URI)
db = client["DiseasePredictions"]
users_collection = db["users"]
predictions_collection = db['predicts']
suggestions_collection = db["suggestions"]

# Initialize JWT manager
app.config['JWT_SECRET_KEY'] = SECRET_KEY
jwt = JWTManager(app)

# Load models (SVM, RF, LR) and scaler
working_dir = os.path.dirname(os.path.abspath(__file__))
try:
    heart_disease_lr_model = joblib.load(open(f'{working_dir}/saved_models/logistic_regression_model.joblib', 'rb'))
    heart_disease_svm_model = joblib.load(open(f'{working_dir}/saved_models/svm_model.joblib', 'rb'))
    heart_disease_rf_model = joblib.load(open(f'{working_dir}/saved_models/rf_model.joblib', 'rb'))
    saved_columns = joblib.load(open(f'{working_dir}/saved_models/saved_columns.joblib', 'rb'))
    scaler = joblib.load(open(f'{working_dir}/saved_models/scaler.joblib', 'rb'))
except Exception as e:
    print(f"Error loading models or scaler: {e}")
    exit("Model or scaler files are missing. Please check the files.")

# Load the dataset
df = pd.read_csv(f'{working_dir}/dataset/heart.csv')  
X = df.drop('target', axis=1)  # Features (exclude 'target' column)
y = df['target']  # Target (the 'target' column)

# Function to train and generate the evaluation metrics for the model
def generate_model_report(model_type):
    categorical_columns = ['cp', 'restecg', 'exang', 'slope', 'ca', 'thal', 'fbs']
    continuous_columns = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

    # One-hot encode categorical variables
    X_encoded = pd.get_dummies(X, columns=categorical_columns, drop_first=True)

    # Save the column names for consistency
    saved_columns = X_encoded.columns.tolist()

    # Initialize the scaler
    scaler = StandardScaler()
    X_encoded[continuous_columns] = scaler.fit_transform(X_encoded[continuous_columns])

    # Train-test split (for evaluation)
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

    # Select the appropriate model
    if model_type == 'LR':
        model = heart_disease_lr_model
    elif model_type == 'SVM':
        model = heart_disease_svm_model
    elif model_type == 'RF':
        model = heart_disease_rf_model
    else:
        return None, "Model type not supported"

    print(f"Model type: {type(model)}")

    # Train the model
    model.fit(X_train, y_train)

    # Predictions for the test set
    predictions = model.predict(X_test)

    # Check if the model supports `predict_proba`
    if hasattr(model, "predict_proba"):
        pred_probs = model.predict_proba(X_test)[:, 1]  # Probability for positive class
    else:
        # If predict_proba is not available, use predictions as an alternative
        pred_probs = predictions  # Probability for positive class

    # Generate ROC curve
    fpr, tpr, _ = roc_curve(y_test, pred_probs)
    roc_auc = auc(fpr, tpr)

    # Create ROC curve plot
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic (ROC)')
    ax.legend(loc='lower right')

    # Save ROC curve to base64
    buf = BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    roc_image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()

    # Generate confusion matrix
    cm = confusion_matrix(y_test, predictions)
    fig, ax = plt.subplots()
    cax = ax.matshow(cm, cmap='Blues')
    fig.colorbar(cax)
    ax.set_xlabel('Predicted', color='white')
    ax.set_ylabel('Actual', color='white')
    ax.set_title('Confusion Matrix', color='white')

    # Adding numerical values inside the matrix cells
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            text_color = 'white' if cm[i, j] > cm.max() / 2 else 'black'
            ax.text(j, i, str(cm[i, j]), ha='center', va='center', color=text_color, fontsize=14)

    # Save confusion matrix to base64
    buf = BytesIO()
    fig.savefig(buf, format='png', facecolor='#2e2e2e')  # Save with dark background
    buf.seek(0)
    cm_image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()

    return {
        "roc_image": roc_image_base64,
        "cm_image": cm_image_base64,
        "roc_auc": roc_auc
    }, None

def preprocess_new_data(data, scaler, saved_columns):
    # List of categorical columns (update these based on your dataset)
    categorical_columns = ['cp', 'restecg', 'exang', 'slope', 'ca', 'thal', 'fbs']
    
    # One-hot encode the categorical features
    data_encoded = pd.get_dummies(data, columns=categorical_columns, drop_first=True)

    # Add missing columns (those that are in saved_columns but not in the input data)
    missing_cols = set(saved_columns) - set(data_encoded.columns)
    for col in missing_cols:
        data_encoded[col] = 0  # Add missing columns with value 0
    print("Missing Columns:", set(saved_columns) - set(data_encoded.columns))
    print("Extra Columns:", set(data_encoded.columns) - set(saved_columns))


    # Reorder columns to match the saved_columns order
    data_encoded = data_encoded[saved_columns]

    # List of continuous columns that need to be scaled (ensure this matches your training columns)
    continuous_columns = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    
    # Apply the scaler to the continuous columns
    data_encoded[continuous_columns] = scaler.transform(data_encoded[continuous_columns])



    return data_encoded

@app.route("/results", methods=['GET'])
def viewResults():
    try:
        # Fetch prediction history from MongoDB
        prediction_history = list(predictions_collection.find().sort('prediction_time', -1))  # Get most recent predictions
        
        # Format results to be more readable
        formatted_results = []
        for record in prediction_history:
            formatted_results.append({
                "input_data": record.get("input_data"),
                "prediction": record.get("prediction"),
                "model_used": record.get("model_used"),
                "prediction_time": record.get("prediction_time").strftime("%Y-%m-%d %H:%M:%S"),
                "accuracy": record.get("accuracy")
            })

        return jsonify({"prediction_history": formatted_results}), 200

    except Exception as e:
        print(f"Error processing /results: {e}")
        return jsonify({"message": "An error occurred while processing the request"}), 500


@app.route('/generate_reports', methods=['POST'])
def generate_reports():
    try:
        # Get the model type from the request
        data = request.get_json()
        model_type = data.get('model_type', 'LR')  # Default to Logistic Regression
        print(f"Model type selected: {model_type}")

        # Generate the report based on the selected model
        report, error = generate_model_report(model_type)

        if error:
            return jsonify({"error": error}), 400

        # Return the generated report
        report_data = {
            "roc_images": {
                model_type: report["roc_image"],
            },
            "confusion_matrices": {
                model_type: report["cm_image"],
            },
            "comparative_auc_chart": None,
            "roc_auc": report["roc_auc"]
        }

        return jsonify(report_data)

    except Exception as e:
        print(f"Error occurred: {e}")

        return jsonify({"error": str(e)}), 500
    
@app.route('/submit_suggestion', methods=['POST'])
def submit_suggestion():
    data = request.json
    print(f"Received suggestion: {data}")  
    name = data.get('name')
    email = data.get('email')
    suggestion = data.get('suggestion')

    if not name or not email or not suggestion:
        return jsonify({"message": "Name, email, and suggestion are required!"}), 400

    try:
        suggestion_record = {
            "name": name,
            "email": email,
            "suggestion": suggestion,
            "submitted_at": datetime.now()
        }
        suggestions_collection.insert_one(suggestion_record)
    except Exception as e:
        return jsonify({"message": f"Error saving suggestion: {str(e)}"}), 500

    return jsonify({"message": "Thank you for your suggestion!"}), 200


@app.route('/signup', methods=['POST'])
def signup_user():
    data = request.json
    print("signed up",data)
    username = data.get('username')
    password = data.get('password')
    email = data.get('email')

    if not username or not password or not email:
        return jsonify({"message": "Username, email, and password are required!"}), 400

    # Email validation
    if not re.match(r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$", email):
        return jsonify({"message": "Invalid email format!"}), 400
    
    # Password validation
    if len(password) < 8:
        return jsonify({"message": "Password must be at least 8 characters long, with at least one uppercase letter and one number."}), 400

    # Check if the username or email already exists in the database
    if users_collection.find_one({"username": username}):
        return jsonify({"message": "Username already exists!"}), 400
    if users_collection.find_one({"email": email}):
        return jsonify({"message": "Email already registered!"}), 400

    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

    try:
        users_collection.insert_one({
            "username": username,
            "email": email,
            "password": hashed_password
        })
    except Exception as e:
        return jsonify({"message": f"Error during registration: {str(e)}"}), 500

    return jsonify({"message": "User registered successfully!"}), 200

@app.route('/login', methods=['POST'])
def login_user():
    data = request.json
    email = data.get('email')
    password = data.get('password')

    if not email or not password:
        return jsonify({"message": "Email and password are required"}), 400

    user = users_collection.find_one({"email": email})
    
    if not user:
        return jsonify({"message": "User not found"}), 404

    if bcrypt.checkpw(password.encode('utf-8'), user['password']):
        token = create_access_token(identity=email, expires_delta=timedelta(days=1))

        return jsonify({"token": token}), 200

    return jsonify({"message": "Invalid credentials"}), 400

@app.route('/predict', methods=['POST'])
@jwt_required()
def predict_heart_disease():

    current_user = get_jwt_identity()  
    print(f"Request from user: {current_user}")  # For debugging

    data = request.get_json()  # Get the input data from the request

    required_fields = [
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 
        'exang', 'oldpeak', 'slope', 'ca', 'thal', 'model_type'
    ]
    
    # Check if all required fields are in the input data
    for field in required_fields:
        if field not in data:
            return jsonify({"message": f"{field} is required"}), 400

    # Validate numeric fields (age, trestbps, chol, thalach, oldpeak)
    try:
        age = float(data['age'])
        trestbps = float(data['trestbps'])
        chol = float(data['chol'])
        thalach = float(data['thalach'])
        oldpeak = float(data['oldpeak'])
        
        # Basic validation for numeric fields
        if age < 0 or age > 100:
            return jsonify({"message": "Age must be between 0 and 100"}), 400
        if thalach < 50 or thalach > 220:
            return jsonify({"message": "Thalach must be between 50 and 220"}), 400

    except ValueError as e:
        return jsonify({"message": f"Invalid input: {str(e)}"}), 400

    # Extract model_type
    model_type = data.get('model_type', "")

    # Prepare the features list (make sure to extract the correct values)
    features = [
        age, data['sex'], data['cp'], trestbps, chol, data['fbs'], data['restecg'], 
        thalach, data['exang'], oldpeak, data['slope'], data['ca'], data['thal']
    ]

    # Create the DataFrame for the model input
    try:
        input_data = pd.DataFrame([features], columns=required_fields[:13])
        processed_data = preprocess_new_data(input_data, scaler, saved_columns)  # Preprocess input data
        print("Processed Data Shape:", processed_data.shape)
          # Log the processed DataFrame
    except Exception as e:
        return jsonify({"message": f"Error during preprocessing: {str(e)}"}), 500

    # Model selection based on the user's input model_type
    model_type = model_type.upper() if model_type else None

    # Determine which model to use
    if model_type == 'LR':
        model = heart_disease_lr_model
    elif model_type == 'SVM':
        model = heart_disease_svm_model
    elif model_type == 'RF':
        model = heart_disease_rf_model
    else:
        return jsonify({"message": "Invalid model type selected"}), 400

    # Make the prediction using the selected model
    try:
        
        if model_type == 'RF' and hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(processed_data)
           
            print("Probabilities for RF", probabilities)

            prediction = 'Heart Disease predicted' if probabilities[0][1] >= 0.4 else 'No Heart Disease predicted'
        elif model_type == 'LR':
            probabilities = model.predict_proba(processed_data)
            print("Probabilities for LR", probabilities)
            prediction = 'Heart Disease predicted' if probabilities[0][1] >= 0.36 else 'No Heart Disease predicted'
        else:
            # For models without `predict_proba`, use `predict`
            prediction_class = model.predict(processed_data)[0]
            prediction = 'Heart Disease predicted' if prediction_class == 1 else 'No Heart Disease predicted'
    except Exception as e:
        print(f"Error during prediction: {e}")  # Log the error to the console
        return jsonify({"message": f"Prediction error: {str(e)}"}), 500

    # Calculate the model's accuracy (if possible)
    accuracy = None
    try:
        full_data = pd.read_csv(f'{working_dir}/dataset/heart.csv')  
        X = full_data.drop(columns=['target'])  
        y = full_data['target']  

        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Preprocess the test data (same preprocessing as for the input data)
        processed_X_test = preprocess_new_data(X_test, scaler, saved_columns)

        # Use the pre-trained model to predict on the test data
        test_predictions = model.predict(processed_X_test)

        # Calculate accuracy
        accuracy = accuracy_score(y_test, test_predictions)
        print(f"Model accuracy on test data: {accuracy * 100:.2f}%")
    except Exception as e:
        print(f"Error during accuracy calculation: {str(e)}")

    # Save prediction in the database (MongoDB)
    prediction_record = {
        "input_data": data,
        "prediction": prediction,  # The prediction result string
        "model_used": model_type,
        "prediction_time": datetime.now(),
        "accuracy": f"{accuracy * 100:.2f}%" if accuracy is not None else "N/A",
    }

    try:
        predictions_collection.insert_one(prediction_record)
        print(f"Prediction saved: {prediction}")  # Log for debugging
    except Exception as e:
        print(f"Error storing prediction: {e}")
        return jsonify({"message": f"Error storing prediction: {str(e)}"}), 500

    # Return the prediction and accuracy
    response_data = {'model_type':model_type,'result': prediction, 'prediction': 1 if 'Heart Disease predicted' in prediction else 0}
    if accuracy is not None:
        response_data['accuracy'] = f"{accuracy * 100:.2f}%"

    return jsonify(response_data)

if __name__ == '__main__':
    app.run(debug=True)
