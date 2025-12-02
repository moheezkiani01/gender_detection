Gender Detection Using Machine Learning (Random Forest)

This project is a Machine Learning based Gender Detection System that predicts gender using different numerical features. The model is trained using the Random Forest Algorithm and achieves high accuracy after proper data preprocessing and balancing.


---

📁 Project Structure

Gender-Detection-ML/
│
├── data.csv                   # Dataset file
├── gender_detection.ipynb     # Jupyter Notebook (Main Project File)
├── gender_rf_model.pkl        # Saved Trained Model
├── README.md                  # Project Documentation


---

🛠 Technologies & Libraries Used

Python

Pandas

NumPy

Scikit-learn

Imbalanced-learn (SMOTE)

Joblib

Jupyter Notebook



---

🎯 Project Objective

The main goal of this project is to:

Predict gender using machine learning.

Apply data preprocessing, feature scaling, and class balancing (SMOTE).

Train a powerful Random Forest Classifier.

Save the trained model for future use.



---

📊 Dataset Description

The dataset is stored in data.csv.

It contains numerical features related to gender.

The target column is Gender.

Label Encoding is used to convert gender into numeric form:

Male → 1

Female → 0




---

⚙️ Data Preprocessing Steps

1. Loading Data using Pandas.


2. Exploration using:

head(), tail()

info(), describe()



3. Label Encoding to convert text labels.


4. Feature Scaling using StandardScaler.


5. Train-Test Split (80% training, 20% testing).


6. Class Balancing using SMOTE to handle imbalanced data.




---

🤖 Machine Learning Model

Random Forest Classifier is used with:

300 decision trees

Entropy-based splitting

Automatic feature selection

Full CPU utilization for fast training



---

📈 Model Training & Evaluation

The model is trained using:

rf.fit(X_train, y_train)

Accuracy is calculated using:

accuracy_score(y_test, y_pred)

The model achieved high accuracy on test data.



---

💾 Model Saving

The trained model is saved using Joblib:

joblib.dump(rf, "gender_rf_model.pkl")

This model can be reused without retraining.


---

▶️ How to Run the Project

1. Install required libraries:



pip install pandas numpy scikit-learn imbalanced-learn joblib

2. Open the notebook:



jupyter notebook gender_detection.ipynb

3. Run all cells step by step.




---

✅ Why Random Forest?

High accuracy

Handles overfitting well

Works on small & large datasets

Easy to explain in academic projects

No complex parameter tuning required



---

👨‍🎓 Author

Moheez kiani – Artificial Intelligence
Developed as part of academic learning.


---

📌 Note

This project is for educational purposes only.


---
