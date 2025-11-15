PlantVillage Potato Disease Classification
Deep Learning | Computer Vision | EfficientNet-B0 | PyTorch

This project detects potato leaf diseases using the PlantVillage Dataset.
The model classifies images into three categories:

🟢 Healthy

🟡 Early Blight

🔴 Late Blight

This is part of my Computer Vision assignment at Shoolini University, developed using PyTorch, EfficientNet, and trained locally in VS Code.

📌 Table of Contents

Overview

Dataset

Tech Stack

Project Structure

Model Architecture

Training Instructions

Prediction

Results

How to Run

Future Work

Author

📝 Overview

This project focuses on building a high-accuracy image classification model to detect potato diseases.
It uses EfficientNet-B0 (Transfer Learning) for fast and robust training.

The project includes:
✔ Model training
✔ Evaluation
✔ Prediction script
✔ Clean folder structure
✔ Portable for GitHub/LinkedIn

📂 Dataset

Dataset used: PlantVillage — Potato
Contains three classes:

Potato___Healthy

Potato___Early_Blight

Potato___Late_Blight

You can download from Kaggle:
🔗 PlantVillage Dataset (Potato)
https://www.kaggle.com/datasets

Dataset was split into:

70% Train
15% Validation
15% Test

🛠 Tech Stack
Component	Technology
Language	Python
DL Framework	PyTorch
Model	EfficientNet-B0
Tools	VS Code, CMD
Visualization	Matplotlib
Dataset	PlantVillage
📁 Project Structure
PlantVillage-Potato/
│── src/
│     ├── train.py
│     ├── model.py
│     ├── predict.py
│     ├── utils.py
│
│── saved_model/
│── requirements.txt
│── potato_classification.ipynb
│── README.md
│── .gitignore

🧠 Model Architecture

Model: EfficientNet-B0

Pretrained on ImageNet

Custom classifier head:

1280 -> 3 (output classes)


Optimizer: Adam

Loss Function: CrossEntropy

Image Size: 224 × 224

Batch Size: 32

🏋️ Training Instructions
1️⃣ Create virtual environment
python -m venv env
env\Scripts\activate

2️⃣ Install dependencies
pip install -r requirements.txt

3️⃣ Run training
python src/train.py


Model will be saved to:

saved_model/potato_model.pth

🔍 Prediction

To run prediction on any potato leaf image:

python src/predict.py


Output example:

Predicted Class: Late Blight

📊 Results

Add your actual values after training:

Training Accuracy: XX%

Validation Accuracy: XX%

Test Accuracy: XX%

Accuracy and loss graphs can be added here for visibility.

🧩 How to Run This Project Locally
git clone https://github.com/Rishi-rsk/PlantVillage-Potato-Classification.git
cd PlantVillage-Potato-Classification
pip install -r requirements.txt
python src/train.py

🚀 Future Work

Integrate Grad-CAM visualizations

Build a Streamlit web UI

Add real-time disease detection using webcam

Deploy model on cloud (AWS / HuggingFace Spaces)

👨‍💻 Author

Rishi Kulshresth
Campus Ambassador — Shoolini University
B.Tech CSE (Artificial Intelligence)
GitHub: https://github.com/Rishi-rsk

LinkedIn: (add your link)

If you found this helpful, please ⭐ the repo!
