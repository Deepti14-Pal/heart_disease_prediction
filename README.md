# heart_disease_prediction
A Machine Learning &amp; Deep Learning based project to predict heart disease using clinical features.

This project aims to predict the presence of heart disease in patients using various machine learning and deep learning models. It was developed as part of the IBITF Sponsored PRAYAS Internship task under ABV-IIITM, Gwalior.

## 📌 Project Objective

To build a predictive model that classifies whether a person is likely to have heart disease based on clinical parameters.

## 📂 Dataset

- Dataset: [UCI Heart Disease Dataset](https://archive.ics.uci.edu/ml/datasets/heart+Disease)
- Records: 303 (in actual dataset, sample used here)
- Features: Age, Sex, Chest Pain Type, Cholesterol, Resting BP, Max Heart Rate, etc.
- Target: 0 (No Disease), 1 (Disease)

## 🛠️ Technologies Used

 **Python 3.10**
 **Libraries**: Pandas, NumPy, Seaborn, Matplotlib, Scikit-learn, XGBoost, TensorFlow
 **IDE**: PyCharm

## 📊 Models Implemented

1. Logistic Regression  
2. Random Forest Classifier  
3. XGBoost Classifier  
4. Neural Network (TensorFlow Keras)

## 📈 Model Accuracy Comparison

| Model               | Accuracy |
|--------------------|----------|
| Logistic Regression| 85%      |
| Random Forest      | 88% ✅   |
| XGBoost            | 86%      |
| Neural Network     | 87%      |

🔹 **Random Forest** was selected as the final model due to high accuracy and low overfitting.

## 📸 Screenshots

 Age Distribution Graph
 Model Training History (Neural Network)
 Accuracy Comparison (Bar Chart)
 Feature Importance (Random Forest)

## 🚀 How to Run

```bash
# Clone the repository
git clone https://github.com/Deepti14-Pal/heart_disease_prediction.git

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # On Windows

# Install dependencies
pip install -r requirements.txt

# Run the project
python heart_disease_pridiction.py

🙋‍♀️ Author
Deepti Pal
B.Tech CSE (2021–2025)
VITM, Gwalior
📧 deeptipal612@gmail.com

📄 License
This project is part of an academic assignment under PRAYAS Internship Scheme. Not intended for commercial use.
