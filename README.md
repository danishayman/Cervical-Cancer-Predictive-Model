# Cervical Cancer Predictive Model 📊

This project implements a machine learning pipeline to predict the risk factors associated with cervical cancer using data preprocessing, feature selection, and various classification algorithms. The project is implemented in a Jupyter Notebook (.ipynb) environment using Python.

---

## 🎯 Overview

The objective of this project is to build a predictive model that can classify the presence of cervical cancer risk factors. The dataset includes clinical and demographic information, along with several test results. The project includes data preprocessing, feature engineering, and model evaluation using multiple machine learning algorithms.

---

## 🌟 Features

- **Data Cleaning**: Handling missing values and standardizing the dataset.
- **Feature Scaling**: Using `StandardScaler` for normalization.
- **Feature Selection**: Identifying the most significant features based on importance metrics.
- **Model Training**: Implementing models such as:
  - Random Forest Classifier
  - Decision Tree Classifier with GridSearchCV
  - Support Vector Machine (SVM) with GridSearchCV
  - Neural Networks with Hyperparameter Tuning
  - Evolutionary Neural Network System
- **Model Evaluation**: Using metrics like accuracy, confusion matrix, precision, recall, and F1-score.
- **Hyperparameter Tuning**: Optimizing model performance using Grid Search and Evolutionary Algorithms.
- **Advanced Techniques**: SMOTE for handling imbalanced data, class weights, and genetic algorithms.

---

## ⚙️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/danishayman/Cervical-Cancer-Predictive-Model.git

   cd Cervical-Cancer-Predictive-Model
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Jupyter Notebooks:
   ```bash
   jupyter notebook Cervical_Cancer_Predictive_Model.ipynb
   jupyter notebook "Project Part 2 Cancer2.ipynb"
   ```

---

## 📂 Dataset

The dataset used in this project includes clinical features related to cervical cancer risk factors. It contains boolean-like and numerical data.

- **Source**: [Provide a link if publicly available, or describe briefly if not accessible.]

**Target Variables:**
- Hinselmann
- Schiller
- Citology
- Biopsy

---

## 🧠 Models and Methods

### Part 1: Traditional Machine Learning Approach

#### 1. Data Preprocessing
- Handled missing values by dropping rows and columns with excessive NaNs.
- Transformed boolean-like columns to proper numeric or boolean types.
- Standardized features using `StandardScaler`.

#### 2. Feature Selection
- Selected top 5 features using feature importance metrics from the Random Forest model:
  - `Age`
  - `First sexual intercourse`
  - `Hormonal Contraceptives (years)`
  - `Number of sexual partners`
  - `Num of pregnancies`

#### 3. Model Training
- **Random Forest Classifier**: Trained on the full feature set.
- **Decision Tree Classifier**: Tuned with hyperparameters like `max_depth`, `min_samples_split`, and `min_samples_leaf`.
- **Support Vector Machine (SVM)**: Explored kernels like linear and sigmoid with varying regularization parameters.

#### 4. Model Evaluation
- Metrics:
  - Accuracy
  - Precision, Recall, F1-score
  - Confusion Matrix
- Performed validation using a holdout validation set.

### Part 2: Advanced Machine Learning Techniques

#### 1. Neural Network Implementation
- Built feedforward neural networks with TensorFlow/Keras
- Implemented early stopping to prevent overfitting
- Used SMOTE for handling imbalance in the dataset
- Applied class weights to address class imbalance

#### 2. Evolutionary Neural Network Optimization
- Implemented genetic algorithms using DEAP library
- Evolved neural network architectures to find optimal hyperparameters:
  - Learning rate
  - Dropout rate
  - Batch size
  - Number of hidden layers
  - Number of neurons per layer

#### 3. Model Comparison and Evaluation
- Compared traditional models with neural network approaches
- Visualized training and validation performance
- Evaluated models using accuracy, precision, recall, F1-score, and confusion matrices
- Demonstrated improved performance with the evolutionary approach

---

## 🏆 Results

### Part 1: Traditional Machine Learning
- **Decision Tree Accuracy**: 74.63%
- **SVM Accuracy**: 64.18%.

### Part 2: Advanced Neural Networks
- **Basic Neural Network Accuracy**: 67.66%
- **Evolutionary Neural Network Accuracy**: 75.25%
- **Evolutionary Neural Network Precision**: 76.30%
- **Evolutionary Neural Network F1-Score**: 77.88%

Detailed confusion matrices and classification reports are provided in the notebooks.

---

## 🚀 How to Use

1. **Load Dataset**: Replace `risk_factors.csv` with your dataset file.
2. **Preprocess Data**: Run the preprocessing cells to clean and prepare the data.
3. **Train Models**: Use the provided pipeline to train and evaluate models.
4. **Hyperparameter Tuning**: Modify parameters in the GridSearchCV section as needed.
5. **Apply Advanced Techniques**: Run the evolutionary algorithm to optimize neural network architecture.
6. **Export Results**: Save the processed dataset or model outputs for future use.

---

## 🛠️ Dependencies

- Python 3.8+
- pandas
- numpy
- scikit-learn
- TensorFlow
- Keras
- DEAP (for evolutionary algorithms)
- imbalanced-learn (for SMOTE)
- matplotlib
- Jupyter Notebook

---

## 🙏 Acknowledgements
- **Libraries Used**: pandas, scikit-learn, numpy, TensorFlow, Keras, DEAP, imbalanced-learn

---
