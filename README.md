# 🚢 Titanic Survival Engine: Predictive Analysis
**`Machine Learning | Classification | Pipeline Architecture`**

## ⚡ The Solution
This project is an end-to-end Machine Learning pipeline designed to predict passenger survival with high reliability. Instead of running a single model, this engine performs a **comparative analysis** across multiple architectures—Logistic Regression, Random Forest, and XGBoost—to identify the most robust solution for real-world deployment.

### 🧪 Key Insights
After rigorous testing, **Random Forest** was selected as the final production model. It achieved an Accuracy of **82.7%**, outperforming Logistic Regression and providing a more stable architecture than XGBoost for this specific dataset.

---

## 🛠️ Tech Stack
![Python 3.13](https://img.shields.io/badge/Python-3.13-blue?logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Latest-orange?logo=scikitlearn)
![Pandas](https://img.shields.io/badge/Pandas-Data_Intelligence-150458?logo=pandas)

- **Core:** Python 3.13
- **ML Frameworks:** Scikit-Learn, XGBoost
- **Data Handling:** Pandas, NumPy
- **Architecture:** Object-Oriented Programming (OOP)

---

## 📊 Model Performance Comparison
The following metrics were evaluated on a strictly isolated test set of 179 samples:

| Model | Accuracy | Recall (Survivors) | F1-Score (Weighted) |
| :--- | :--- | :--- | :--- |
| **Random Forest** | **82.68%** | **78.38%** | **0.83** |
| XGBoost | 82.12% | 77.03% | 0.82 |
| Logistic Regression | 82.12% | 74.32% | 0.82 |

**Engineering Decision:** We selected **Random Forest** as the primary model. While XGBoost offered similar performance, Random Forest provided the highest overall accuracy and a superior balance between model complexity and variance.

---

## 🚀 Architectural Features
- **Data Leakage Prevention:** Engineered a custom Object-Oriented `AgeImputer` class that calculates medians strictly from the training data to prevent test-set leakage.
- **Automated Preprocessing:** Modular pipeline handling one-hot encoding, dynamic feature scaling (`StandardScaler`), and missing value imputation.
- **Detailed Evaluation:** Utilized Confusion Matrices and Classification Reports to measure precision-recall trade-offs effectively.

---

## 📂 Project Structure & Usage
1. **Clone the repo:**
   ```bash
   git clone [https://github.com/NaramCharan/Titanic-Survival-Engine-Predictive-Analysis.git](https://github.com/NaramCharan/Titanic-Survival-Engine-Predictive-Analysis.git)
Install dependencies:

Bash
pip install -r requirements.txt

3. **Run Analysis:** Execute `Decision Trees.ipynb` to view the full pipeline, feature engineering, and model evaluations.

---

## 🗺️ Roadmap
- [ ] **Hyperparameter Tuning:** Implementing `GridSearchCV` to aggressively optimize depth, estimators, and learning rates.
- [ ] **Neural Network Integration:** Implementing a Multi-Layer Perceptron (MLP) using TensorFlow/Keras to benchmark against tree-based models.
- [ ] **API Deployment:** Wrapping the trained Random Forest model in a FastAPI backend for real-time JSON predictions.

---

## 📬 Contact
- **Naram Charan** - [LinkedIn](https://www.linkedin.com/in/naramcharan/) | [Email](mailto:charannaram1710@gmail.com)
