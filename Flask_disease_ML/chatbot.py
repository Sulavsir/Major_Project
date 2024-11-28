# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity


# predefined_questions = [
#     "What is heart disease?",
#     "How does this prediction system work?",
#     "What data do I need to provide for a prediction?",
#     "How can I calculate the risk of heart disease?",
#     "What factors affect the prediction results?",
#     "Can I download my prediction report?",
#     "What is BMI, and does it affect the predictions?",
#     "Does smoking influence the prediction results?",
#     "How accurate is the heart disease prediction?",
#     "What does this chatbot do?",
#     "How do I reset my password?",
#     "Do you store my personal data?",
#     "Can I update my profile information?",
#     "What is the minimum age for using the prediction system?",
#     "How do I contact support?",
#     "How does the system use my data?",
#     "Is the prediction system free to use?",
#     "What happens if I enter incorrect details?",
#     "Is this chatbot available 24/7?",
#     "Can I trust AI predictions for my health?",
#     "What should I do if my prediction shows a high risk?",
#     "Can quitting smoking improve my heart health?",
#     "How does smoking affect heart disease risk?",
#     "What are the main risk factors for heart disease?",
#     "Can I compare predictions from different models?",
# ]

# predefined_responses = [
#     "Heart disease refers to various conditions affecting the heart, including coronary artery disease, heart attacks, and arrhythmias.",
#     "This system uses machine learning models (Logistic Regression, SVM, Random Forest) to predict the likelihood of heart disease based on your health data.",
#     "You need to provide information like age, sex, cholesterol levels, blood pressure, chest pain type, and more to generate a prediction.",
#     "The system calculates your risk using your input data and analyzes it through trained machine learning models.",
#     "Factors like age, cholesterol, blood pressure, exercise habits, and smoking status significantly affect prediction results.",
#     "Yes, you can download your prediction report as a PDF or view it directly in your dashboard.",
#     "BMI (Body Mass Index) is a measure of body fat based on your height and weight. It can indirectly affect predictions through related health conditions.",
#     "Yes, smoking is considered. It significantly increases the risk of heart disease and affects the prediction results.",
#     "Our models are trained on reliable datasets and optimized for accuracy, but no prediction system is 100% accurate. Always consult a doctor for confirmation.",
#     "This chatbot assists you in using the prediction system, answering questions, and guiding you through the app.",
#     "You can reset your password by clicking on 'Forgot Password' and following the instructions sent to your email.",
#     "Your personal data is securely stored and used only for prediction purposes. We comply with data protection regulations.",
#     "Yes, you can update your profile information through the 'Profile' section in the app.",
#     "The system can be used by individuals aged 18 and above. For younger users, consult a healthcare provider.",
#     "You can contact support through the 'Contact Us' section in the app or by emailing our support team.",
#     "Your data is used solely for generating predictions and improving the system. It is never shared without your consent.",
#     "Yes, the basic features of the prediction system are free to use. Additional services may incur a cost.",
#     "If you enter incorrect details, the predictions may be inaccurate. Double-check your data before submitting.",
#     "Yes, the chatbot is available 24/7 to answer your questions and assist with the system.",
#     "AI predictions are reliable but not a substitute for medical advice. Always consult a healthcare professional.",
#     "If your prediction shows a high risk, consult a doctor immediately and consider lifestyle changes to improve your heart health.",
#     "Yes, quitting smoking can improve heart health and reduce your risk of heart disease significantly.",
#     "Smoking contributes to plaque buildup in arteries, increasing the risk of heart attacks and other conditions.",
#     "Common risk factors include high blood pressure, high cholesterol, diabetes, obesity, smoking, and a sedentary lifestyle.",
#     "Yes, you can compare results from Logistic Regression, SVM, and Random Forest models to understand their predictions better.",
# ]

# vectorizer = TfidfVectorizer(stop_words='english').fit(predefined_questions)

# @app.route('/chatbot', methods=['POST'])
# def chatbot_response():
#     user_input = request.json.get('query') 

#     if not user_input:
#         return jsonify({"error": "No query provided"}), 400

#     # Add user query to predefined questions
#     queries = predefined_questions + [user_input]

#     # Transform to vectors
#     vectors = vectorizer.transform(queries)

#     # Compute cosine similarity
#     similarity_matrix = cosine_similarity(vectors[-1], vectors[:-1])
#     max_similarity = similarity_matrix.max()
#     most_similar_index = similarity_matrix.argmax()

#     # Define similarity threshold
#     similarity_threshold = 0.7

#     if max_similarity < similarity_threshold:
#         return jsonify({"response": "I'm sorry, I didn't understand that. Can you rephrase your question?"})
    
#     # Return the most similar predefined response
#     return jsonify({"response": predefined_responses[most_similar_index]}), 200
