# Diabetes Prediction System

## 📋 Project Overview

The **Diabetes Prediction System** is a machine learning application that predicts whether a patient has diabetes based on health metrics. The system uses **Logistic Regression** algorithm and provides a user-friendly GUI built with Tkinter for easy interaction.

## ✨ Features

- **Dataset Loading**: Load custom CSV files containing patient health data
- **Variable Configuration**: Select dependent and independent variables
- **Algorithm Training**: Train Logistic Regression model on your data
- **Model Evaluation**: Get accuracy, confusion matrix, and classification reports
- **Interactive Predictions**: Make predictions on new patient data through GUI
- **Data Scaling**: Automatic feature scaling using StandardScaler
- **User-Friendly Interface**: Easy-to-use Tkinter GUI with clear navigation

## 📊 Project Structure

```
Model/
├── model.py                    # Main application with GUI
├── diabetesprediction.csv      # Sample diabetes dataset
└── README.md                   # Documentation
```

## 🔧 Installation & Setup

### Requirements
- Python 3.7 or higher
- pandas
- numpy
- scikit-learn
- tkinter (usually comes with Python)

### Installation Steps

1. **Clone/Navigate to the project directory**:
```bash
cd "d:\Eshu\DS Intern2026\Model"
```

2. **Create a Virtual Environment** (Optional but recommended):
```bash
python -m venv venv
venv\Scripts\activate
```

3. **Install Required Packages**:
```bash
pip install pandas numpy scikit-learn
```

## 🚀 How to Use

### Running the Application

```bash
python model.py
```

### Workflow Steps

1. **Load Dataset**
   - Click "Load Dataset" button
   - Select a CSV file containing patient health data
   - The application will display loaded columns

2. **Configure Variables**
   - Click "Next" to go to Variable Configuration page
   - Enter the dependent variable name (e.g., "diabetes_status")
   - Independent variables are automatically selected from remaining columns
   - Click "Set Variables" to proceed

3. **Select Algorithm**
   - Choose "Logistic Regression" algorithm
   - This is the default and recommended algorithm for binary classification

4. **Train Model**
   - Click "Train Model" to train the classifier
   - Review the accuracy, confusion matrix, and classification report
   - The model will be ready for predictions

5. **Make Predictions**
   - Click "Go to Prediction Page"
   - Enter input values for all health metrics
   - Click "Predict" to get the prediction result
   - Result: "Diabetes" or "No Diabetes"

## 📝 Dataset Format

The CSV file should contain health metrics. Example columns:
- `age`: Patient's age
- `hypertension`: Whether patient has hypertension (0/1)
- `heart_disease`: Whether patient has heart disease (0/1)
- `bmi`: Body Mass Index
- `HbA1c_level`: Hemoglobin A1c level
- `blood_glucose_level`: Blood glucose level
- `diabetes_status`: Target variable (0/1) - Diabetes/No Diabetes

## 🤖 Model Details

### Algorithm: Logistic Regression
- **Type**: Binary Classification
- **Train-Test Split**: 80-20
- **Hyperparameters**:
  - max_iter: 3000
  - random_state: 42

### Data Processing
- **Feature Scaling**: StandardScaler normalization
- **Categorical Encoding**: LabelEncoder for categorical columns
- **Missing Values**: Handled automatically

## 📈 Evaluation Metrics

The system provides:
- **Accuracy Score**: Overall accuracy percentage
- **Confusion Matrix**: True/False Positives and Negatives
- **Classification Report**: Precision, Recall, F1-score

## 🔐 Key Functions

| Function | Purpose |
|----------|---------|
| `load_dataset()` | Load CSV file with patient data |
| `go_to_enter_variables_page()` | Configure dependent/independent variables |
| `go_to_algorithm_page()` | Select ML algorithm |
| `go_to_train_model_page()` | Train the classifier |
| `go_to_prediction_page()` | Make predictions on new data |
| `predict()` | Generate prediction for input values |

## 💡 Tips

- Ensure your CSV file has proper column names without spaces or special characters
- The dependent variable should be binary (0/1 or No/Yes)
- Larger datasets provide better model training
- Use realistic values when making predictions for accurate results

## 🛠️ Troubleshooting

| Issue | Solution |
|-------|----------|
| "Failed to load dataset" | Check CSV file format and path |
| "Invalid dependent variable" | Ensure variable name matches exactly |
| "Error during prediction" | Verify all input values are numeric |
| Import errors | Install required packages: `pip install pandas numpy scikit-learn` |
