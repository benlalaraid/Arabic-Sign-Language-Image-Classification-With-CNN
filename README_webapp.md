# Arabic Hand Sign Classification Web Application

This web application provides a modern, responsive interface for real-time detection of Arabic hand signs using a trained machine learning model.

## Features

- Real-time camera feed for capturing images
- Upload images for hand sign classification
- Real-time detection mode
- Modern responsive UI
- Display of predictions with confidence levels

## Requirements

All required packages are listed in `requirements.txt`. You can install them using:

```
pip install -r requirements.txt
```

## Running the Application

1. Make sure your model is saved at `models/arabic_letters_model.h5`
2. Run the FastAPI server:

```
python -m uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

3. Open your browser and navigate to `http://localhost:8000`

## Application Structure

- `app.py`: FastAPI backend server
- `templates/index.html`: Main HTML template
- `static/styles.css`: CSS styling
- `static/script.js`: JavaScript for frontend functionality

## Using the Application

1. Start the camera using the "Start Camera" button
2. Either capture an image or enable real-time mode
3. View the prediction results on the right panel
4. Alternatively, upload an image using the file upload form

## Model Information

The application uses a Convolutional Neural Network (CNN) model for accurate classification of Arabic hand signs. The model is trained to recognize various hand signs and displays the top predictions along with their confidence scores.
