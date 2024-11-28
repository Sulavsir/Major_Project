from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import bcrypt
import pandas as pd
from pymongo import MongoClient
import os
from datetime import datetime, timedelta
import re
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score
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

def preprocess_new_data(data, scaler, saved_columns):
    # List of categorical columns (update these based on your dataset)
    categorical_columns = ['cp', 'restecg', 'exang', 'slope', 'ca', 'thal', 'fbs']
    
    # One-hot encode the categorical features
    data_encoded = pd.get_dummies(data, columns=categorical_columns, drop_first=True)

    # Add missing columns (those that are in saved_columns but not in the input data)
    missing_cols = set(saved_columns) - set(data_encoded.columns)
    for col in missing_cols:
        data_encoded[col] = 0  # Add missing columns with value 0

    # Reorder columns to match the saved_columns order
    data_encoded = data_encoded[saved_columns]

    # List of continuous columns that need to be scaled (ensure this matches your training columns)
    continuous_columns = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    
    # Apply the scaler to the continuous columns
    data_encoded[continuous_columns] = scaler.transform(data_encoded[continuous_columns])



    return data_encoded




# Function to generate ROC curve and AUC
def generate_roc_curve(model, X_test, y_test, model_name):
    fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8,6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic (ROC) for {model_name}')
    plt.legend(loc='lower right')

    # Save the plot to a BytesIO object
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    
    # Convert the image to base64 string for frontend use
    return base64.b64encode(img.getvalue()).decode('utf8')

# Function to generate confusion matrix and return as a base64 string
def generate_confusion_matrix(model, X_test, y_test):
    cm = confusion_matrix(y_test, model.predict(X_test))
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    
    # Save to BytesIO
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    
    return base64.b64encode(img.getvalue()).decode('utf8')

# Function to generate comparative analysis chart (AUC comparison)
def generate_comparative_auc_chart(auc_scores):
    # Create a plot comparing AUC scores
    models = list(auc_scores.keys())
    scores = list(auc_scores.values())
    
    fig = go.Figure([go.Bar(x=models, y=scores)])
    fig.update_layout(title="Comparative AUC of Different Models", xaxis_title="Model", yaxis_title="AUC Score")
    
    # Return the chart in JSON format for Plotly to render
    return fig.to_json()


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


# Route to generate the report for all models
@app.route('/generate_reports', methods=['GET'])
def generate_reports():
    # Example X_test, y_test (replace with actual data)
    X_test = pd.DataFrame()  # Add your test features here
    y_test = pd.Series()  # Add your test labels here
    
    auc_scores = {}
    
    # Calculate AUC for each model
    for model, model_name in zip([heart_disease_lr_model, heart_disease_svm_model, heart_disease_rf_model], 
                                 ['Logistic Regression', 'Support Vector Machine', 'Random Forest']):
        
        auc_scores[model_name] = auc(*roc_curve(y_test, model.predict_proba(X_test)[:, 1])[:2])
    
    # Generate ROC curves and confusion matrices for each model
    roc_images = {}
    cm_images = {}
    
    for model, model_name in zip([heart_disease_lr_model, heart_disease_svm_model, heart_disease_rf_model], 
                                 ['Logistic Regression', 'Support Vector Machine', 'Random Forest']):
        roc_images[model_name] = generate_roc_curve(model, X_test, y_test, model_name)
        cm_images[model_name] = generate_confusion_matrix(model, X_test, y_test)
    
    # Generate comparative AUC chart
    comparative_auc_chart = generate_comparative_auc_chart(auc_scores)
    
    # Fetch test data from the database or another source
    user_data = list(predictions_collection.find().sort('prediction_time', -1))[:10]  # Just for illustration
    
    # Return the data to frontend
    return jsonify({
        "roc_images": roc_images,
        "confusion_matrices": cm_images,
        "comparative_auc_chart": comparative_auc_chart,
        "prediction_history": user_data
    })

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
    print("Incoming data:")
    print(request.json)

    current_user = get_jwt_identity()  # Get user info from JWT
    print(f"Request from user: {current_user}")  # For debugging

    data = request.get_json()  # Get the input data from the request

    # Required fields to be validated in the request
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
        print("Processed DataFrame after preprocessing:")
        print(processed_data)  # Log the processed DataFrame
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
        prediction = model.predict(processed_data)
    except Exception as e:
        print(f"Error during prediction: {e}")  # Log the error to the console
        return jsonify({"message": f"Prediction error: {str(e)}"}), 500

    # Determine the prediction result (1 = Disease, 0 = No Disease)
    result = 'Heart Disease predicted' if prediction[0] == 1 else 'No Heart Disease predicted'

    # Evaluate the model's accuracy without retraining:
    try:
        try:
            full_data = pd.read_csv(f'{working_dir}/dataset/heart.csv')  
        except Exception as e:
            print(f"An error occurred: {e}")  

        # Prepare X and y (features and labels)
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
        accuracy = None
        print(f"Error during accuracy calculation: {str(e)}")

    # Record the prediction in the database (MongoDB or your preferred DB)
    prediction_record = {
        "input_data": data,
        "prediction": result,
        "model_used": model_type,
        "prediction_time": datetime.now(),
        "accuracy": f"{accuracy * 100:.2f}%" if accuracy is not None else "N/A"
    }

    # Insert the prediction record into the database (ensure database connection is set up)
    try:
        predictions_collection.insert_one(prediction_record)
    except Exception as e:
        print(f"Error storing prediction: {e}")  # Log the error to the console
        return jsonify({"message": f"Error storing prediction: {str(e)}"}), 500

    # Return the result, the prediction, and the accuracy (if available)
    response_data = {'result': result, 'prediction': int(prediction[0])}
    if accuracy is not None:
        response_data['accuracy'] = f"{accuracy * 100:.2f}%"

    return jsonify(response_data)

if __name__ == '__main__':
    app.run(debug=True)
