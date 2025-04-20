# Skin_Disease_Classifier

🩺 Skin Disease Classification using CNN & ResNet18
This project focuses on the binary classification of skin images as Infected or Not Infected using deep learning models, specifically a custom CNN and a fine-tuned ResNet18 (via ResNet50). The goal is to assist in early skin disease detection using image-based AI diagnosis tools.




# Features
✅ Custom-built Convolutional Neural Network (CNN)

✅ Transfer Learning with ResNet18

✅ Evaluation using Accuracy, Precision, Recall, F1-Score

✅ Grad-CAM visualization for explainable AI

✅ Preprocessing pipeline for image normalization and resizing

✅ Designed for lightweight deployment in resource-constrained environments







# Dataset
- Total Images: 1,287
- Classes: Infected, Not Infected
- Format: .JPG images
- Preprocessing: Resized to 200x200, normalized to [0, 1]









# Explainability
This project integrates Grad-CAM to highlight which regions of the image the model focused on during classification, enhancing trust and transparency in AI-driven diagnostics.


# Clone the repository
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name

# Install dependencies
pip install -r requirements.txt

# Run training
python train_cnn.py

# Run Grad-CAM
python gradcam.py





🛠# Future Enhancements
- Multi-class skin condition classification
- Mobile deployment with TensorFlow Lite
- Integration with real-time webcam inference
- Incorporate patient metadata for hybrid decision-making
  









# License
- This project is open-source and licensed under the MIT License.
  



# Dataset Link
- https://data.mendeley.com/datasets/x4hgnjj5gv/2
