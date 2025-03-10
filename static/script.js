// JavaScript for Arabic Letter Classification App

// DOM Elements
const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const capturedImage = document.getElementById('capturedImage');
const startCameraBtn = document.getElementById('startCamera');
const captureImageBtn = document.getElementById('captureImage');
const fileUpload = document.getElementById('fileUpload');
const fileName = document.getElementById('fileName');
const uploadForm = document.getElementById('uploadForm');
const mainPrediction = document.getElementById('mainPrediction');
const mainConfidenceBar = document.getElementById('mainConfidenceBar');
const mainConfidence = document.getElementById('mainConfidence');
const topPredictions = document.getElementById('topPredictions');
const realtimeToggle = document.getElementById('realtimeToggle');
const realtimeStatus = document.getElementById('realtimeStatus');

// Global variables
let stream = null;
let websocket = null;
let realtimeMode = false;
let captureInterval = null;

// Initialize the application
document.addEventListener('DOMContentLoaded', () => {
    // Setup event listeners
    setupEventListeners();
    
    // Create a placeholder image
    createPlaceholderImage();
});

// Setup all event listeners
function setupEventListeners() {
    // Start camera button
    startCameraBtn.addEventListener('click', toggleCamera);
    
    // Capture image button
    captureImageBtn.addEventListener('click', captureImage);
    
    // File upload change event
    fileUpload.addEventListener('change', handleFileSelect);
    
    // Form submit event
    uploadForm.addEventListener('submit', handleFormSubmit);
    
    // Realtime toggle
    realtimeToggle.addEventListener('change', toggleRealtimeMode);
}

// Create a placeholder image for the captured image container
function createPlaceholderImage() {
    const ctx = canvas.getContext('2d');
    canvas.width = 300;
    canvas.height = 225;
    
    // Fill with light gray
    ctx.fillStyle = '#f5f5f5';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    // Add text
    ctx.fillStyle = '#999';
    ctx.font = '16px Roboto';
    ctx.textAlign = 'center';
    ctx.fillText('No image captured', canvas.width / 2, canvas.height / 2);
    
    // Set as placeholder
    capturedImage.src = '/static/placeholder.png';
}

// Toggle camera on/off
async function toggleCamera() {
    if (stream) {
        // Stop the camera
        stopCamera();
        startCameraBtn.innerHTML = '<i class="fas fa-play"></i> Start Camera';
        captureImageBtn.disabled = true;
    } else {
        // Start the camera
        try {
            stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    width: { ideal: 640 },
                    height: { ideal: 480 },
                    facingMode: 'environment' // Use back camera on mobile devices
                },
                audio: false
            });
            
            video.srcObject = stream;
            startCameraBtn.innerHTML = '<i class="fas fa-stop"></i> Stop Camera';
            captureImageBtn.disabled = false;
            
            // If realtime mode is on, start capturing
            if (realtimeMode) {
                startRealtimeCapture();
            }
        } catch (err) {
            console.error('Error accessing camera:', err);
            alert('Error accessing camera. Please make sure you have granted camera permissions.');
        }
    }
}

// Stop the camera
function stopCamera() {
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        video.srcObject = null;
        stream = null;
        
        // Stop realtime capture if active
        stopRealtimeCapture();
    }
}

// Capture image from video
function captureImage() {
    if (!stream) return;
    
    const context = canvas.getContext('2d');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    
    // Draw the video frame to the canvas
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    
    // Convert to image and display
    capturedImage.src = canvas.toDataURL('image/png');
    
    // Send to backend for prediction
    sendImageForPrediction(canvas.toDataURL('image/png'));
}

// Handle file selection
function handleFileSelect(event) {
    const file = event.target.files[0];
    if (file) {
        fileName.textContent = file.name;
        
        // Display the selected image
        const reader = new FileReader();
        reader.onload = function(e) {
            capturedImage.src = e.target.result;
        };
        reader.readAsDataURL(file);
    } else {
        fileName.textContent = 'No file chosen';
    }
}

// Handle form submission
async function handleFormSubmit(event) {
    event.preventDefault();
    
    const file = fileUpload.files[0];
    if (!file) {
        alert('Please select an image file first.');
        return;
    }
    
    // Create form data
    const formData = new FormData();
    formData.append('file', file);
    
    try {
        // Show loading state
        mainPrediction.textContent = 'Processing...';
        mainConfidenceBar.style.width = '0%';
        mainConfidence.textContent = '0%';
        topPredictions.innerHTML = '';
        
        // Send request to backend
        const response = await fetch('/predict/', {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        
        // Handle error
        if (result.error) {
            alert(`Error: ${result.error}`);
            return;
        }
        
        // Display prediction results
        displayPredictionResults(result);
    } catch (error) {
        console.error('Error submitting image:', error);
        alert('Error submitting image. Please try again.');
    }
}

// Send image for prediction via WebSocket
function sendImageForPrediction(imageDataUrl) {
    // If WebSocket is not connected, connect it
    if (!websocket || websocket.readyState !== WebSocket.OPEN) {
        connectWebSocket();
        
        // Wait for connection and then send
        setTimeout(() => {
            if (websocket && websocket.readyState === WebSocket.OPEN) {
                websocket.send(imageDataUrl);
            }
        }, 1000);
        return;
    }
    
    // Send the image data
    websocket.send(imageDataUrl);
    
    // Show loading state
    mainPrediction.textContent = 'Processing...';
    mainConfidenceBar.style.width = '0%';
    mainConfidence.textContent = '0%';
    topPredictions.innerHTML = '';
}

// Connect to WebSocket
function connectWebSocket() {
    // Close existing connection if any
    if (websocket) {
        websocket.close();
    }
    
    // Create new WebSocket connection
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws`;
    
    websocket = new WebSocket(wsUrl);
    
    websocket.onopen = () => {
        console.log('WebSocket connected');
    };
    
    websocket.onmessage = (event) => {
        const data = JSON.parse(event.data);
        
        // Handle error
        if (data.error) {
            alert(`Error: ${data.error}`);
            return;
        }
        
        // Display prediction results
        displayPredictionResults(data);
    };
    
    websocket.onerror = (error) => {
        console.error('WebSocket error:', error);
    };
    
    websocket.onclose = () => {
        console.log('WebSocket disconnected');
    };
}

// Display prediction results
function displayPredictionResults(data) {
    // Check if this is a dummy prediction
    const isDummyPrediction = data.note !== undefined;
    
    // Show notification if using dummy predictions
    if (isDummyPrediction) {
        // Create notification if it doesn't exist
        if (!document.getElementById('model-notification')) {
            const notification = document.createElement('div');
            notification.id = 'model-notification';
            notification.className = 'notification';
            notification.innerHTML = `
                <div class="notification-content warning">
                    <i class="fas fa-exclamation-triangle"></i>
                    <span>Model not loaded: ${data.note}</span>
                </div>
            `;
            document.querySelector('.prediction-container').prepend(notification);
        }
    } else {
        // Remove notification if it exists
        const notification = document.getElementById('model-notification');
        if (notification) {
            notification.remove();
        }
    }
    
    // Main prediction
    mainPrediction.textContent = data.prediction || '-';
    
    // Confidence percentage
    const confidencePercent = Math.round((data.confidence || 0) * 100);
    mainConfidenceBar.style.width = `${confidencePercent}%`;
    mainConfidence.textContent = `${confidencePercent}%`;
    
    // Change color based on confidence
    if (confidencePercent > 80) {
        mainConfidenceBar.style.backgroundColor = '#2ecc71'; // Green
    } else if (confidencePercent > 50) {
        mainConfidenceBar.style.backgroundColor = '#f39c12'; // Orange
    } else {
        mainConfidenceBar.style.backgroundColor = '#e74c3c'; // Red
    }
    
    // Top predictions
    topPredictions.innerHTML = '';
    
    if (data.top_predictions && data.top_predictions.length > 0) {
        // Skip the first one if it's the same as the main prediction
        const otherPredictions = data.top_predictions.filter((pred, index) => 
            index > 0 || pred.label !== data.prediction
        );
        
        otherPredictions.forEach(pred => {
            const predPercent = Math.round(pred.confidence * 100);
            
            const predItem = document.createElement('div');
            predItem.className = 'prediction-item';
            predItem.innerHTML = `
                <span class="prediction-label">${pred.label}</span>
                <div class="prediction-confidence-bar-container">
                    <div class="prediction-confidence-bar" style="width: ${predPercent}%"></div>
                </div>
                <span class="prediction-confidence">${predPercent}%</span>
            `;
            
            topPredictions.appendChild(predItem);
        });
    } else {
        topPredictions.innerHTML = '<p>No other predictions available</p>';
    }
}

// Toggle realtime mode
function toggleRealtimeMode() {
    realtimeMode = realtimeToggle.checked;
    realtimeStatus.textContent = realtimeMode ? 'On' : 'Off';
    
    if (realtimeMode && stream) {
        startRealtimeCapture();
    } else {
        stopRealtimeCapture();
    }
}

// Start realtime capture
function startRealtimeCapture() {
    if (captureInterval) return;
    
    // Capture every 1 second
    captureInterval = setInterval(() => {
        if (stream && realtimeMode) {
            captureImage();
        }
    }, 1000);
}

// Stop realtime capture
function stopRealtimeCapture() {
    if (captureInterval) {
        clearInterval(captureInterval);
        captureInterval = null;
    }
}

// Clean up resources when page is unloaded
window.addEventListener('beforeunload', () => {
    stopCamera();
    if (websocket) {
        websocket.close();
    }
});
