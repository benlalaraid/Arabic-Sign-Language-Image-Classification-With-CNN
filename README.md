# Arabic Hand Sign Classification with CNN 

This project implements a Convolutional Neural Network (CNN) for classifying Arabic hand signs. The implementation includes data loading, preprocessing, model training, evaluation, and a web application for real-time detection.

## Project Structure

- `app.py`: FastAPI web application for real-time arabic signs detection
- `templates/index.html`: HTML template for the web interface
- `static/script.js`: JavaScript for frontend functionality
- `static/styles.css`: CSS styling for the web interface
- `notebook/main.ipynb`: Jupyter notebook with data loading and preprocessing steps

## Web Application

The project includes a web application that provides a user-friendly interface for real-time detection of the Arabic hand signs.

To run the web application:

```bash
python -m uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

For more details about the web application, see [README_webapp.md](README_webapp.md).

## Model Architecture

The implementation uses a Convolutional Neural Network (CNN) with the following components:

1. Convolutional layers for feature extraction
2. Max pooling layers for spatial dimension reduction
3. Dropout layers for regularization
4. Dense layers for classification

## Fine-tuning

After initial training, the model is fine-tuned by unfreezing the last few layers of the CNN model and training with a lower learning rate. This helps the model adapt better to the specific characteristics of Arabic hand signs.

## Outputs

The training process generates the following outputs:

- `arabic_hand_signs_model.h5`: Trained model file
- `arabic_hand_signs_model_finetuned.h5`: Fine-tuned model file
- `training_history.png`: Plot of training and validation accuracy/loss
- `confusion_matrix.png`: Confusion matrix visualization
- `prediction_samples.png`: Visualization of model predictions on sample images
- `classification_report.txt`: Detailed classification metrics

## Requirements

- TensorFlow 2.x
- FastAPI
- Uvicorn
- NumPy
- OpenCV
- Pillow
- Matplotlib
- Scikit-learn
- Seaborn
