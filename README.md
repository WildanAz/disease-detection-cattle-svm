🐄 Cattle Disease Detection Using Image-Based Data Mining & SVM

A Machine Learning System for Classifying Foot-and-Mouth Disease, Foot Rot, Necrotic Stomatitis, and Healthy Cattle

📌 Overview

This project develops an automated cattle disease diagnosis system using image data, the KDD (Knowledge Discovery in Database) framework, and a Support Vector Machine (SVM) multiclass classifier (One-vs-Rest).
The system classifies four health conditions:

Healthy

Foot-and-Mouth Disease (FMD / PMK)

Foot Rot

Necrotic Stomatitis

The final model achieves:

Accuracy: 95.4%

Precision: 95.6%

Recall: 95.4%

F1-Score: 95.2%

The system is implemented as a web application that accepts cattle images, extracts visual features, runs classification, and provides disease handling recommendations.

🎯 Objectives

Build an early detection system for cattle mouth & hoof diseases.

Convert raw image data into structured features using Azure Custom Vision.

Train an interpretable machine learning model (SVM) to classify diseases.

Provide actionable recommendations for farmers and veterinarians.

📂 Dataset

Total images: 374 original → 555 after augmentation

Class	Open Data	Field Data	After Augmentation
Healthy	98	7	133
PMK	138	4	156
Foot Rot	20	0	153
Necrotic Stomatitis	20	0	113
Sources:

Zenodo Open Dataset (official cattle disease dataset)

Local farm field images validated by PMK officers from Dinas Peternakan Jawa Barat

🔍 Feature Extraction

Feature extraction is performed using 4 Azure Custom Vision models:

Feature	Classes	Purpose
Location	Gum, Tongue, Hoof	Identify area of symptoms
Wound	Wounded, Not Wounded	Detect lesion presence
Color	Red, Yellow, Black	Identify inflammation/necrosis
Texture	Smooth, Rough	Detect tissue damage

Each model outputs labels via API, combined into a structured CSV dataset.

🧠 Model Architecture

Algorithm: Support Vector Machine (SVM)
Mode: One-vs-Rest multiclass

Steps:

Encode categorical features → numerical vectors

Normalize dataset

Train SVM for 4 classes

Evaluate using confusion matrix & standard ML metrics

Use decision functions to determine highest scoring class

🧪 Model Performance
Metric	Score
Accuracy	95.4%
Precision	95.6%
Recall	95.4%
F1-Score	95.2%
🖥️ System Implementation

The system is deployed as a web application capable of:

Uploading cattle mouth/hoof images

Extracting features via Azure API

Running SVM classification

Displaying predicted disease & confidence score

Showing treatment recommendations

Example Output:

Predicted Class: Foot Rot

Detected Features: Hoof, Wounded, Yellow, Rough

Recommendation: Clean wound, apply antibiotics, improve sanitation

📌 Key Contributions (Your Role)

You can copy this section directly as part of your GitHub project:

Built and trained all Azure Custom Vision feature extraction models

Conducted data exploration, cleaning, encoding, and augmentation

Implemented custom feature extraction pipeline using Azure API

Designed and trained SVM multiclass classifier from scratch

Evaluated model performance and optimized hyperparameters

Designed the web-based diagnosis interface and API integration

Created the recommendation system based on veterinary guidelines

📁 Repository Structure
/data
  /raw_images
  /augmented_images
  dataset.csv

/model
  svm_classifier.pkl
  feature_encoder.pkl

/src
  feature_extraction.py
  preprocessing.py
  train_svm.py
  predict.py
  web_app/

README.md

🚀 How to Run
1. Install dependencies
pip install -r requirements.txt

2. Run preprocessing
python src/preprocessing.py

3. Train the model
python src/train_svm.py

4. Start the web app
python src/web_app/app.py
