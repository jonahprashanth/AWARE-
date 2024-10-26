# AWARE: Autism Wellness Assessment and Recognition Engine

## Overview
The AWARE system employs a dual assessment strategy for early diagnosis of autism spectrum disorder (ASD) using:
1. **Questionnaire Analysis**: Analyzes responses to a structured questionnaire filled out by parents using a logistic regression model.
2. **Computer Vision Analysis**: Captures and analyzes facial expressions and eye movements of the child through a standard webcam while reading a passage.
3. **Integration**: Combines results from both assessments to generate a comprehensive and accurate prediction.

## Installation
To set up the AWARE system, clone this repository and install the required packages:
```bash
    git clone <https://github.com/jonahprashanth/AWARE->
    cd <cd AWARE>
    pip install -r requirements.txt
```

## Usage
To Run the AWARE System
Prepare your dataset by referring to this [link](https://www.kaggle.com/datasets/cihan063/autism-image-data) and download it.
Execute the app script to start the analysis:
```bash
python app.py
```

## Project Structure
The project is organized as follows:
```bash
AWARE/
│
├── Computer_Vision/
│   ├── autism_dataset/          # Directory containing images for the autism dataset
│   ├── model_creation/          # Scripts for creating and training the computer vision model
│   └── saved_models/            # Directory for storing trained model files
│
├── Questionnaire_Analysis/
│   ├── autism_model.sav         # Saved logistic regression model
│   ├── data.csv                 # CSV file containing questionnaire data
│   └── linear_regression.ipynb   # Jupyter notebook for training and evaluating the regression model
│
├── readme.txt                   # Additional information about the project
├── app.py                       # Main application script to run the analysis
└── requirements.txt             # File listing required packages for the project
```

## Contributing
Contributions are welcome! Please follow these steps:
#### 1. Fork the repository.
#### 2. Create a new branch for your feature or fix.
#### 3. Submit a pull request detailing your changes.

### Instructions for Use
- Make sure to have all datasets and models properly placed in their respective directories before running the application.
- Refer to `readme.txt` for any additional instructions or notes about the project.
