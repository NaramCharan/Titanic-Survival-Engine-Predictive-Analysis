# 🚢 Titanic Survival Engine: Predictive Analysis
**`Machine Learning | Classification | Pipeline Architecture`**

## ⚡ The Solution
This project is an end-to-end Machine Learning pipeline designed to predict passenger survival with high reliability. Instead of just running a single model, this engine performs a **comparative analysis** across multiple architectures to identify the most robust solution for real-world deployment.

### 🧪 Key Insights
After rigorous testing, **Random Forest** was selected as the final production model. It achieved an Accuracy of **85.5%**, outperforming Logistic Regression and providing a more stable, less overfit architecture than XGBoost for this specific dataset size.

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
The following metrics were evaluated on a strictly isolated, unseen Test Set:

| Model | Accuracy | Recall |
| :--- | :--- | :--- |
| **Random Forest** | **85.5%** | 81.1% |
| XGBoost | 83.2% | **82.4%** |
| Logistic Regression | 82.1% | 78.4% |

**Engineering Decision:** We selected **Random Forest** as the primary model. While XGBoost offered a slightly higher recall, Random Forest provided the highest overall accuracy and a superior balance between model complexity and variance, making it the optimal "production-ready" choice.

---

## 🚀 Architectural Features
- **Data Leakage Prevention:** Engineered a custom Object-Oriented `AgeImputer` class that calculates medians strictly from the training data to prevent test-set leakage.
- **Automated Preprocessing:** Modular pipeline handling one-hot encoding, dynamic feature scaling (`StandardScaler`), and missing value imputation.
- **Hyperparameter Tuning:** Utilized `GridSearchCV` across both tree-based models to aggressively optimize depth, estimators, and learning rates.

---

## 📂 Project Structure & Usage
1. **Clone the repo:** `git clone [YOUR_URL]`
2. **Install dependencies:** `pip install -r requirements.txt`
3. **Run Analysis:** Execute `Decision Trees.ipynb` to view the full pipeline, grid search, and evaluations.

---

## 🗺️ Roadmap
- [ ] **Neural Network Integration:** Implementing a Multi-Layer Perceptron (MLP) using TensorFlow/Keras to benchmark against tree-based models.
- [ ] **API Deployment:** Wrapping the trained Random Forest model in a FastAPI backend for real-time JSON predictions.
- [ ] **MLOps Tracking:** Integrating MLflow to track model experiments and metric drift over time.

---

## 📬 Contact
- **Naram Charan** - [LinkedIn](https://www.linkedin.com/in/naramcharan/) | [Email](mailto:charannaram1710@gmail.com)
