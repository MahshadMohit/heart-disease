# 🫀 Heart Disease Prediction using Machine Learning

This project is focused on building multiple machine learning models to **predict the likelihood of heart disease** using the famous [UCI Heart Disease Dataset](https://archive.ics.uci.edu/ml/datasets/heart+Disease). The goal is to identify patients at risk and assist in early diagnosis using explainable, interpretable models.

---

## 📊 Dataset: UCI Heart Disease

- 🔗 Source: UCI Machine Learning Repository  
- 📦 Features: 13 numerical and categorical clinical attributes  
- 🎯 Target: Binary (0 = No heart disease, 1 = Heart disease present)  
- 💡 Shape: `303 rows × 14 columns`

---

## 🔬 Project Workflow

1. **Data Preprocessing**
   - Standardization using `StandardScaler`
   - Train/Test Split (80% / 20%)
   - Feature selection on numeric columns

2. **Model Training**
   - Multiple models were trained and evaluated:
     - ✅ Logistic Regression
     - ✅ Support Vector Machine (SVM)
     - ✅ K-Nearest Neighbors (KNN)
     - ✅ Multi-layer Perceptron (MLPClassifier)
     - ✅ Decision Tree
     - ✅ Random Forest
     - ✅ Extra Trees
     - ✅ Naive Bayes
     - ✅ XGBoost

3. **Evaluation Metrics**
   - Accuracy
   - Precision, Recall, F1-Score
   - Confusion Matrix
   - Cross-Validation (5-Fold)

---

## 📈 Model Evaluation Summary

| Model               | Accuracy | Precision (1) | Recall (1) | F1-Score (1) | CV Mean Acc. |
|---------------------|----------|----------------|-------------|--------------|---------------|
| **SVM (poly kernel)**     | **0.9016** | 0.91          | 0.91        | **0.91**     | **0.7686**     |
| **KNN**                  | **0.9016** | 0.93          | 0.88        | 0.90         | 0.7686         |
| **MLPClassifier**        | **0.9016** | 0.88          | **0.94**    | 0.91         | 0.7686         |
| Naive Bayes         | 0.8688   | 0.90          | 0.84        | 0.87         | 0.7686         |
| Logistic Regression | 0.8525   | 0.87          | 0.84        | 0.86         | 0.7686         |
| XGBoost             | 0.8525   | 0.87          | 0.84        | 0.86         | 0.7686         |
| Extra Trees         | 0.8525   | 0.87          | 0.84        | 0.86         | 0.7686         |
| Random Forest       | 0.8360   | 0.84          | 0.84        | 0.84         | 0.7686         |
| Decision Tree       | 0.8360   | 0.89          | 0.78        | 0.83         | 0.7686         |

✅ All models performed above 83% accuracy  
💡 **SVM (poly kernel)** and **MLP** achieved the best overall performance

---

## 💾 Final Model

The selected final model is:

```python
SVC(kernel='poly', probability=True)
