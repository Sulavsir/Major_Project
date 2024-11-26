
# # Endpoint for Diabetes Prediction
# @app.route('/diabetes', methods=['POST'])
# def predict_diabetes():
#     data = request.get_json()

#     # Validate required fields
#     required_fields = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
#     for field in required_fields:
#         if field not in data:
#             return jsonify({"message": f"{field} is required"}), 400

#     features = [
#         data['Pregnancies'],
#         data['Glucose'],
#         data['BloodPressure'],
#         data['SkinThickness'],
#         data['Insulin'],
#         data['BMI'],
#         data['DiabetesPedigreeFunction'],
#         data['Age']
#     ]
    
#     prediction = diabetes_model.predict([np.array(features)])
#     return jsonify({'result': 'The person is diabetic' if prediction[0] == 1 else 'The person is not diabetic'})
