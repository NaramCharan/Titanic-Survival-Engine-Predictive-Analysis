# 🚢 Titanic Survival Engine: Predictive Analysis
**`Machine Learning | Classification | Production-Oriented Pipeline Architecture`**

A production-style Machine Learning system built to predict passenger survival on the Titanic dataset using modern preprocessing pipelines, intelligent hyperparameter optimization, and comparative model benchmarking.

---

## ⚡ Project Overview

This project is an **end-to-end Machine Learning classification engine** designed to predict whether a Titanic passenger survived or not based on passenger demographics and travel information.

Rather than relying on a single baseline model, this system evaluates and benchmarks multiple machine learning algorithms to determine the most reliable deployment candidate.

The project emphasizes:

✅ **Reproducibility**  
✅ **Data Leakage Prevention**  
✅ **Pipeline-Based Architecture**  
✅ **Cross-Validated Evaluation**  
✅ **Hyperparameter Optimization**  
✅ **Clean Modular Code Structure**

---

## 🎯 Objective

The objective of this project is to build a **reliable survival prediction system** while following **production-oriented machine learning practices**.

Instead of training models directly on raw data, the system uses:

- Automated preprocessing pipelines
- Modular feature engineering
- Intelligent missing value handling
- Stratified cross-validation
- Hyperparameter optimization

to improve reliability and maintainability.

---

## 🧠 Machine Learning Architecture

This project follows a **modular ML engineering architecture** rather than a notebook-only workflow.

The entire pipeline is separated into reusable components for better maintainability and scalability.

### 🔹 Feature Engineering Module (`preprocessing.py`)

A dedicated preprocessing module is responsible for:

- Feature engineering
- Data cleaning
- Dataset preparation
- Target extraction

This separation improves:

- Code readability
- Reusability
- Maintainability
- Experimentation speed

---

### 🔹 Scikit-Learn Pipeline Architecture

The project uses **Scikit-Learn Pipelines** to automate preprocessing and model execution.

This prevents:

- Manual preprocessing errors
- Train-test inconsistencies
- Data leakage issues

The pipeline ensures that transformations applied during training are consistently applied during testing.

---

### 🔹 ColumnTransformer Integration

Implemented **`ColumnTransformer`** for selective preprocessing of feature types.

#### Categorical Features
Applied preprocessing to:

- `Sex`
- `Title`
- `Embarked`
- `Cabin`

Pipeline:

```python
SimpleImputer(strategy="most_frequent")
→ OneHotEncoder(handle_unknown="ignore")
```

This ensures missing categorical values are handled safely before encoding.

---

### 🔹 Missing Value Handling Strategy

Instead of using simple statistical filling everywhere, this project uses a **hybrid imputation strategy**.

#### 1. SimpleImputer
Used for categorical columns:

- Strategy: `most_frequent`

Best suited for categorical missing values.

#### 2. KNNImputer
Used for numerical missing values:

```python
KNNImputer(n_neighbors=10)
```

This estimates missing values using neighboring observations, often improving prediction quality compared to mean/median imputation.

---

### 🔹 Feature Scaling

For **Logistic Regression**, the project uses:

```python
RobustScaler()
```

Why?

Because `RobustScaler` is less sensitive to outliers compared to `StandardScaler`, making Logistic Regression more stable.

---

### 🔹 Cross Validation Strategy

Implemented:

```python
StratifiedKFold(n_splits=10)
```

Why Stratified CV?

- Maintains class balance across folds
- Produces more reliable validation scores
- Reduces evaluation bias
- Better reflects real-world performance

This provides stronger generalization estimates than a simple train-test split.

---

### 🔹 Hyperparameter Optimization using Optuna

Instead of brute-force tuning with `GridSearchCV`, this project uses **Optuna** for smarter hyperparameter search.

Benefits:

✅ Faster optimization  
✅ Smarter parameter exploration  
✅ Computational efficiency  
✅ Better scalability

---

## 🔬 Models Evaluated

The following classification models were benchmarked:

### 1️⃣ Logistic Regression
- RobustScaler applied
- Stratified Cross Validation
- High interpretability
- Strong balance between metrics

### 2️⃣ Random Forest
Hyperparameters optimized using Optuna:

- `n_estimators`
- `max_depth`
- `min_samples_split`
- `min_samples_leaf`
- `max_features`

### 3️⃣ XGBoost
Hyperparameters optimized using Optuna:

- `n_estimators`
- `max_depth`
- `learning_rate`
- `subsample`
- `gamma`

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Latest-orange?logo=scikitlearn)
![XGBoost](https://img.shields.io/badge/XGBoost-Gradient_Boosting-green)
![Optuna](https://img.shields.io/badge/Optuna-Hyperparameter_Optimization-blueviolet)
![Pandas](https://img.shields.io/badge/Pandas-Data_Intelligence-150458?logo=pandas)


## 📊 Model Performance Comparison

| Model | Accuracy | Precision | Recall |
|--------|-----------|------------|---------|
| **Logistic Regression** | **85.47%** | **84.51%** | **80.00%** |
| Random Forest | 83.24% | 82.61% | 76.00% |
| XGBoost | **86.59%** | **89.23%** | 77.33% |

---

## 🏆 Final Model Selection

### Selected Model: Logistic Regression

Although **XGBoost achieved the highest Accuracy and Precision**, **Logistic Regression was selected as the preferred production model**.

### Why Logistic Regression?

The decision was based on **balanced performance rather than highest raw accuracy**.

Logistic Regression offered:

✅ Strong accuracy (**85.47%**)  
✅ Better recall (**80%**)  
✅ More interpretable predictions  
✅ Lower computational complexity  
✅ Reduced overfitting risk  
✅ Easier deployment and maintenance

### Engineering Decision

In real-world ML systems, selecting the model with the **most balanced performance** is often more valuable than selecting the model with the highest single metric.

For this dataset, Logistic Regression produced the best trade-off between:

**Accuracy + Recall + Simplicity + Reliability**

---

## 📂 Project Structure

```bash
Titanic-Survival-Engine-Predictive-Analysis/
│── dataset/
│   └── Titanic-Dataset.csv
│
│── src/
│   ├── preprocessing.py
│   └── Titanic Survival Engine.ipynb
│
│── requirements.txt
│── README.md
```

---

## 🚀 Installation & Usage

### 1️⃣ Clone Repository

```bash
git clone https://github.com/NaramCharan/Titanic-Survival-Engine-Predictive-Analysis.git
```

### 2️⃣ Navigate Into Project

```bash
cd Titanic-Survival-Engine-Predictive-Analysis
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Navigate to Source Directory

```bash
cd src
```

### 5️⃣ Run the Project

Open:

```bash
Titanic Survival Engine.ipynb
```

The notebook walks through:

- Data preprocessing
- Feature engineering
- Pipeline construction
- Missing value imputation
- Hyperparameter optimization
- Model training
- Cross validation
- Performance comparison

---

## 📈 Future Improvements

- [ ] FastAPI deployment
- [ ] Streamlit prediction interface
- [ ] Model persistence using Joblib
- [ ] SHAP model explainability
- [ ] Docker containerization
- [ ] CI/CD integration

---

## 📬 Contact

### 👨‍💻 Naram Charan

**LinkedIn:**  
https://www.linkedin.com/in/naramcharan/

**Email:**  
charannaram1710@gmail.com

---

## ⭐ Support

If you found this project useful or interesting, consider giving it a **star** on GitHub!
