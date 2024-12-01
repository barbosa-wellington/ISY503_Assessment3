
import socketio
import eventlet
import numpy as np
from flask import Flask
from tensorflow.keras.models import load_model
import tensorflow.keras.metrics as metrics
import base64
from io import BytesIO
from PIL import Image
import cv2
import logging
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import MeanSquaredError as MSE

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize SocketIO server
sio = socketio.Server(cors_allowed_origins="*")
app = Flask(__name__)
speed_limit = 10

def img_preprocess(img):
    try:
        img = img[60:135, :, :]
        img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        img = cv2.GaussianBlur(img, (3, 3), 0)
        img = cv2.resize(img, (200, 66))
        img = img / 255.0
        return img
    except Exception as e:
        logger.error(f"Error in image preprocessing: {e}")
        return None






@sio.on('telemetry')
def telemetry(sid, data):
    try:
        print("=== Received Telemetry Data ===")
        print(f"Data keys: {data.keys()}")
        
        if 'image' not in data or 'speed' not in data:
            print("Missing required telemetry data")
            return
            
        speed = float(data['speed'])
        print(f"Current speed: {speed}")
        
        # Process image
        image = Image.open(BytesIO(base64.b64decode(data['image'])))
        image = np.asarray(image)
        processed_image = img_preprocess(image)
        
        if processed_image is None:
            print("Image processing failed")
            return
            
        # Make prediction
        steering_angle = float(model.predict(np.array([processed_image]), verbose=0))
        
        # Set minimum throttle to ensure movement
        throttle = max(0.3, 1.0 - speed / speed_limit)  # Increased minimum throttle
        
        print(f"Sending controls - Steering: {steering_angle:.3f}, Throttle: {throttle:.3f}")
        send_control(steering_angle, throttle)
        
    except Exception as e:
        print(f"Error in telemetry handler: {str(e)}")
        import traceback
        print(traceback.format_exc())

@sio.on('connect')
def connect(sid, environ):
    print('=== Client Connected ===')
    print(f'Session ID: {sid}')
    print('Sending initial control command...')
    # Send a stronger initial throttle
    send_control(0, 0.5)


@sio.on('disconnect')
def disconnect(sid):
    logger.info('Client disconnected')

def send_control(steering_angle, throttle):
    try:
        sio.emit('steer', data={
            'steering_angle': str(steering_angle),
            'throttle': str(throttle)
        })
    except Exception as e:
        logger.error(f"Error sending control: {e}")

# Load the model
model_path = 'model/model.h5'
try:
    model = load_model(
        model_path,
        custom_objects={'mse': MSE()}
    )
    logger.info(f"Model loaded successfully from {model_path}")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    model = None

if __name__ == '__main__':
    if model is None:
        logger.error("Model loading failed. Server not started.")
    else:
        try:
            app = socketio.Middleware(sio, app)
            port = 4567
            logger.info(f"Starting server on port {port}...")
            eventlet.wsgi.server(eventlet.listen(('', port)), app)
        except Exception as e:
            logger.error(f"Server error: {e}")