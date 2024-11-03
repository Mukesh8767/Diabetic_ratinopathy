# A machine learning-based project for detecting the severity of diabetic retinopathy from retinal images using transfer learning on DenseNet121. This project aims to provide accurate classification of retinopathy severity, assisting in early diagnosis and treatment planning.
 Datasetet link : https://www.kaggle.com/datasets/sovitrath/diabetic-retinopathy-224x224-2019-data
Table of Contents
Project Overview
Features
Dataset
Requirements
Installation
Usage
Model Training
Results
Contributions
Acknowledgments
Project Overview
Diabetic retinopathy is a complication of diabetes that affects the eyes and can lead to vision loss if left untreated. This project uses deep learning techniques to analyze retinal images and classify them into five categories: No_DR, Mild, Moderate, Severe, and Proliferative_DR. The model utilizes the DenseNet121 architecture with transfer learning to improve accuracy and reduce training time.

Features
Automated Image Preprocessing: Gaussian filtering, resizing, and label encoding.
Balanced Data: Class balancing via resampling.
Efficient Model Training: DenseNet121 with dropout, batch normalization, and early stopping.
Interactive Visualization: Visualization of model predictions and analysis of diabetic retinopathy severity levels.
Dataset
The dataset consists of retinal images, categorized by severity level. Gaussian-filtered images are organized in labeled folders. You may need to preprocess the dataset (resize and filter images) as per the project requirements.

Note: Update the dataset path in the code to your local path.

Requirements
The project requires the following libraries:

pandas
numpy
opencv-python
tensorflow
keras
plotly
scikit-learn
imutils
tqdm
efficientnet
You can install the required packages with:

bash
Copy code
pip install -r requirements.txt
Installation
Clone the Repository:
bash
Copy code
git clone https://github.com/yourusername/diabetic-retinopathy-detection.git
Navigate to the Project Directory:
bash
Copy code
cd diabetic-retinopathy-detection
Install Dependencies:
bash
Copy code
pip install -r requirements.txt
Usage
Prepare Dataset: Place the dataset images in the directory specified in the code (img_dir).
Run Preprocessing: Preprocess images using Gaussian filters, resizing, and resampling as shown in the code.
Train the Model:
python
Copy code
python train_model.py
Evaluate Model: Evaluate the trained model using confusion matrices, accuracy, and validation metrics.
Model Training
The DenseNet121 model, with frozen base layers, is fine-tuned for diabetic retinopathy detection. Early stopping is applied to prevent overfitting, and the model is optimized using Adam with a learning rate of 0.0001. The dataset is balanced to ensure fair training across all severity levels.

Results
Visualize model performance on test data using accuracy scores and confusion matrices. The results can be further visualized through bar charts for a better understanding of class-specific performance.

Contributions
Mukesh Paliwal

Data Collection and Preprocessing
Model Development and Tuning
Model Evaluation and Documentation
Manan Daxini

Data Analysis and Visualization
Model Evaluation and Report Review
Acknowledgments
DenseNet121 Model
Special thanks to contributors and resources that supported this project.

