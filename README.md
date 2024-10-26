# AWARE: Autism Wellness Assessment and Recognition Engine

## Overview
The AWARE system employs a dual assessment strategy for early diagnosis of autism spectrum disorder (ASD) using:
1. **Questionnaire Analysis**: Analyzes responses to a structured questionnaire filled out by parents using a logistic regression model.
2. **Computer Vision Analysis**: Captures and analyzes facial expressions and eye movements of the child through a standard webcam while reading a passage.
3. **Integration**: Combines results from both assessments to generate a comprehensive and accurate prediction.

## Proposed Approach

### 1. Questionnaire Analysis

#### 1.1 Data Collection and Preparation
- **Dataset Selection**: Choose a dataset containing responses to screening questionnaires designed to assess ASD risk.
- **Data Cleaning**: Handle missing values, outliers, and ensure consistency in variable formatting and coding.

#### 1.2 Data Preprocessing
- **Handle Missing Data**: Address missing values through imputation or removal.
- **Categorical Variables**: Encode categorical variables using one-hot or label encoding.
- **Target Variable**: Define the target variable as binary: `0` for no ASD risk and `1` for ASD risk.

#### 1.3 Feature Selection
- Identify relevant features based on domain knowledge or exploratory data analysis.

#### 1.4 Dataset Splitting
- Split the dataset into training and testing sets (e.g., 70-30 or 80-20).

#### 1.5 Model Training
- Train a logistic regression model on the training dataset.

#### 1.6 Model Evaluation
- Use various metrics (accuracy, precision, recall, F1-score, ROC AUC) to evaluate the model.

#### 1.7 Interpretation and Validation
- Interpret coefficients to identify significant predictors of ASD risk.
- Validate the model's performance using cross-validation techniques.

#### 1.8 Deployment and Application
- Deploy the logistic regression model for screening new individuals.

### 2. Computer Vision Analysis

#### 2.1 Dataset Description
- The Autism Facial Image Dataset sourced from Kaggle, containing images categorized into two classes: "autistic" and "non_autistic".

#### 2.2 Data Preprocessing
- Use the ImageDataGenerator class from TensorFlow for resizing, normalization, and data augmentation.

#### 2.3 Model Architecture
- Utilize a Convolutional Neural Network (CNN) based on the ResNet-50 model for binary classification.

#### 2.4 Model Training
- Compile the ResNet-50 model using the Adam optimizer and binary cross-entropy loss function.

#### 2.5 Evaluation Metrics
- Evaluate model performance using accuracy, precision, recall, and F1-score.

#### 2.6 Results Analysis
- Analyze results to demonstrate the effectiveness of the model in classifying facial images.

### 3. Integration Method
- **Majority Voting Method**: Combines predictions through voting.
- **Weighted Probabilities Method**: Combines predictions using specified weights.
- **Bayesian Approach**: Integrates predictions using log-odds for better robustness.

## Installation

To set up the AWARE system, clone this repository and install the required packages:
```bash
    git clone <https://github.com/jonahprashanth/AWARE->
    cd <cd AWARE>
    pip install -r requirements.txt
```
## Usage

To Run the AWARE System
Prepare your dataset by refering https://www.kaggle.com/datasets/cihan063/autism-image-data and download it.
Execute the app script to start the analysis:
```bash
python app.py
```
## Contributing

Contributions are welcome! Please follow these steps:
#### 1. Fork the repository.
#### 2. Create a new branch for your feature or fix.
#### 3. Submit a pull request detailing your changes.
