# import socketio
# import eventlet
# import numpy as np
# from flask import Flask
# from keras.models import load_model
# import base64
# from io import BytesIO
# from PIL import Image
# import cv2


# sio = socketio.Server()

# app = Flask(__name__) #'__main__'
# speed_limit = 10
# def img_preprocess(img):
#     img = img[60:135,:,:]
#     img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
#     img = cv2.GaussianBlur(img,  (3, 3), 0)
#     img = cv2.resize(img, (200, 66))
#     img = img/255
#     return img


# @sio.on('telemetry')
# def telemetry(sid, data):
#     speed = float(data['speed'])
#     image = Image.open(BytesIO(base64.b64decode(data['image'])))
#     image = np.asarray(image)
#     image = img_preprocess(image)
#     image = np.array([image])
#     steering_angle = float(model.predict(image))
#     throttle = 1.0 - speed/speed_limit
#     print('{} {} {}'.format(steering_angle, throttle, speed))
#     send_control(steering_angle, throttle)



# @sio.on('connect')
# def connect(sid, environ):
#     print('Connected')
#     send_control(0, 0)

# def send_control(steering_angle, throttle):
#     sio.emit('steer', data = {
#         'steering_angle': steering_angle.__str__(),
#         'throttle': throttle.__str__()
#     })




# from tensorflow.keras.models import load_model, save_model
# from tensorflow.python.keras.saving import h5_format

# # Attempt to load model in a backward-compatible manner
# try:
#     with h5_format.HDF5Format():
#         model = load_model('model/model.h5')
# except ValueError as e:
#     print(f"Model loading failed: {e}")

# if __name__ == '__main__':
#     model = load_model('model/model.h5')
#     app = socketio.Middleware(sio, app)
#     eventlet.wsgi.server(eventlet.listen(('', 4567)), app)

import socketio
import eventlet
import numpy as np
from flask import Flask
from tensorflow.keras.models import load_model
import tensorflow.keras.metrics as metrics  # Import built-in Keras metrics
import base64
from io import BytesIO
from PIL import Image
import cv2

# Ensure 'mse' is registered with Keras
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import MeanSquaredError as MSE

# Register mse explicitly (needed for older saved models)
mse = MSE()

# Initialize SocketIO server
sio = socketio.Server(cors_allowed_origins="*")

# Flask application
app = Flask(__name__)
speed_limit = 10

# Image preprocessing function
def img_preprocess(img):
    img = img[60:135, :, :]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.resize(img, (200, 66))
    img = img / 255.0
    return img

# Handle telemetry data
@sio.on('telemetry')
def telemetry(sid, data):
    speed = float(data['speed'])
    image = Image.open(BytesIO(base64.b64decode(data['image'])))
    image = np.asarray(image)
    image = img_preprocess(image)
    image = np.array([image])
    
    steering_angle = float(model.predict(image))
    throttle = 1.0 - speed / speed_limit
    
    print(f"Steering Angle: {steering_angle}, Throttle: {throttle}, Speed: {speed}")
    send_control(steering_angle, throttle)

# Handle client connection
@sio.on('connect')
def connect(sid, environ):
    print('Client connected')
    send_control(0, 0)

# Send control commands to the client
def send_control(steering_angle, throttle):
    sio.emit('steer', data={
        'steering_angle': str(steering_angle),
        'throttle': str(throttle)
    })

# Load the model
model_path = 'model/model.h5'
try:
    model = load_model(
        model_path,
        custom_objects={'mse': mse}  # Register the mse function explicitly
    )
    print(f"Model loaded successfully from {model_path}")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Run the server
if __name__ == '__main__':
    if model:
        app = socketio.Middleware(sio, app)
        port = 4567
        print(f"Starting server on port {port}...")
        eventlet.wsgi.server(eventlet.listen(('', port)), app)
    else:
        print("Model loading failed. Server not started.")
