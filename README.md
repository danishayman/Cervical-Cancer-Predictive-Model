# 🔬 Cervical Cancer Predictive Model


## 📋 Project Overview

This project develops machine learning models to predict cervical cancer based on various risk factors. Our goal is to create accurate and reliable predictive models that can help in early diagnosis and improve treatment outcomes.

## 🧠 Key Features

- **Data Analysis & Preprocessing**: Comprehensive preprocessing including handling missing values, feature selection, and data normalization
- **Multiple Model Development**: Implementation of several machine learning algorithms including:
  - Decision Trees
  - Support Vector Machines (SVM)
  - Neural Networks
  - Evolutionary Neural Networks
- **Model Optimization**: Hyperparameter tuning to maximize performance metrics
- **Comparative Analysis**: Rigorous evaluation of model performance using accuracy, precision, recall, and F1-score

## 📊 Dataset Description

The dataset contains information about risk factors for cervical cancer, including:

- Demographic information (age)
- Habits (smoking, contraceptive use)
- Sexual history
- Medical history
- STD-related information

The target variables for prediction are:
- Hinselmann
- Schiller
- Citology
- Biopsy

## 🔍 Key Findings

### Part 1: Traditional ML Approach
- SVM outperformed Decision Trees with an accuracy of **74.63%** vs. **64.18%**
- SVM showed better recall values across all classes, indicating fewer false negatives
- SVM demonstrated improved performance in precision, recall, and F1-score metrics

### Part 2: Neural Network Approach
- The Evolutionary Neural Network model achieved a test accuracy of **75.25%**
- Precision: **76.30%**
- Recall: **79.52%**
- F1 Score: **77.88%**
- The hybrid model showed significant improvement over the traditional feedforward neural network

## 💻 Technologies Used

- Python
- pandas
- NumPy
- scikit-learn
- TensorFlow/Keras
- DEAP (Distributed Evolutionary Algorithms in Python)
- Matplotlib
- Seaborn

## 🔧 Implementation

The project is divided into two main parts:

1. **Part 1**: Traditional machine learning approach with feature selection and model comparison
2. **Part 2**: Advanced neural network models with evolutionary optimization

## 📋 Prerequisites

- Python 3.7+
- pip or conda for package management
- Required packages (can be installed via `pip install -r requirements.txt`):
  - numpy
  - pandas
  - scikit-learn
  - tensorflow
  - matplotlib
  - seaborn
  - deap

## 🚀 Getting Started

Follow these steps to set up and run the project on your local machine:

### 1. Clone the repository
```bash
git clone https://github.com/danishayman/Cervical-Cancer-Predictive-Model.git
cd Cervical-Cancer-Predictive-Model
```

### 2. Create and activate a virtual environment (optional but recommended)
```bash
# Using venv
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the notebooks
```bash
jupyter notebook
```
Navigate to the notebooks directory and open the desired notebook to run the analysis.

## 📈 Results

### Model Performance Comparison
| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Decision Trees | 64.18% | 65.32% | 63.95% | 64.63% |
| Support Vector Machines (SVM) | 74.63% | 73.87% | 75.21% | 74.53% |
| Traditional Neural Network | 71.34% | 70.85% | 72.46% | 71.64% |
| Evolutionary Neural Network | 75.25% | 76.30% | 79.52% | 77.88% |

Our best-performing model (Evolutionary Neural Network) achieved:
- **75.25%** accuracy
- Improved false positive rate compared to traditional models
- Better balance between precision and recall

## 🔮 Future Work

- Explore additional feature engineering techniques
- Implement ensemble methods
- Test on larger, more diverse datasets
- Deploy as a web application for clinical use

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
