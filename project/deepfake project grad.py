import os
import cv2 as cv
import logging
import numpy as np
from flask import Flask, request, redirect, render_template, send_from_directory
from werkzeug.utils import secure_filename
from skimage.feature import hog                                           
import skimage
#print(skimage.__version__)
import torch 
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image

# Paths
face_cascade_path = "C:/Users/User/Desktop/project/haarcascade_frontalface_default.xml"
output_image_dir = "C:/Users/User/Desktop/grad proj/image_file"
resized_image_dir = "C:/Users/User/Desktop/grad proj/resized"
face_dir = "C:/Users/User/Desktop/grad proj/face_file"
hog_dir = "C:/Users/User/Desktop/grad proj/hog" 

#lbp_dir = "C:/Users/User/Desktop/grad proj/lbp"

# Flask configuration
UPLOAD_FOLDER = 'uploaded_videos'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load pre-trained Haar cascade classifier for face detection
face_cascade = cv.CascadeClassifier(face_cascade_path)

# Extract Frames from Videos
def extract_frames(video_file, output_dir, frame_rate=1):
    try:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        capture = cv.VideoCapture(video_file) # Open the video file
        if not capture.isOpened():
            raise IOError("Error opening video file:", video_file)
        frame_rate_video = capture.get(cv.CAP_PROP_FPS) # Get frame rate of the video
        sampling_interval = int(frame_rate_video / frame_rate)  # Calculate frame sampling interval
        frame_count = 0
        success = True
        while success:   # Loop through the video frames
            success, frame = capture.read()
            if not success:
                break
            if frame_count % sampling_interval == 0:    # Only save frames at the specified rate
                frame_count += 1
                cv.imwrite(os.path.join(output_dir, f"frame_{frame_count}.jpg"), frame)
            else:
                frame_count += 1
        # Release the video capture object
        capture.release()
    except Exception as e:
         # Log the error
        logging.error(f"An error occurred: {e}")
        raise
        # Optionally, re-raise the exception to propagate it further


# Extract faces from images
def extract_faces(image_dir, output_dir): #detect faces and then crop them, and save them in a file. 
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for img_name in os.listdir(image_dir):
        if img_name.endswith(".jpg"):
            image_path = os.path.join(image_dir, img_name) # get the frame[i] from image dir
            img = cv.imread(image_path)
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
            for i, (x, y, w, h) in enumerate(faces):
                face_img = img[y:y+h, x:x+w]
                cv.imwrite(os.path.join(output_dir, f"face_{img_name}_{i}.jpg"), face_img)



# Resize images
def resize_images(input_dir, output_dir, scale=2):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for img_name in os.listdir(input_dir):
        if img_name.endswith(".jpg"):
            img_path = os.path.join(input_dir, img_name) # get the face[i] from face dir
            img = cv.imread(img_path)
            img_width = int(img.shape[1] * scale)   #shape(x, y) 
            img_height = int(img.shape[0] * scale)
            resized_img = cv.resize(img, (img_width, img_height), interpolation=cv.INTER_LINEAR)
            cv.imwrite(os.path.join(output_dir, f"resized_{img_name}"), resized_img)


# Extract HOG features from images
def extract_hog_features(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    features, hog_image = hog(gray, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
    return features, hog_image

def extract_and_save_hog(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir) 
    for img_name in os.listdir(input_dir):
        if img_name.endswith(".jpg"):
            img_path = os.path.join(input_dir, img_name)
            img = cv.imread(img_path)
            features, hog_image = extract_hog_features(img)
            hog_image = cv.normalize(hog_image, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)
            cv.imwrite(os.path.join(output_dir, f"hog_{img_name}"), hog_image)



# Set the device to GPU if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Load pre-trained PyTorch model
model = models.resnet18(pretrained=True)
model.to(device)
model.eval()


# Preprocess image for PyTorch model
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(image).unsqueeze(0).to(device) # unsqueeze: adds an extra dimension to the tensor,


# Perform inference using PyTorch model
def predict_with_pytorch(image_path):
    image = Image.open(image_path).convert("RGB")
    image_tensor = preprocess_image(image)
    with torch.no_grad():
        output = model(image_tensor)
    _, predicted = torch.max(output, 1)
    return predicted.item()

# Analyze video
def analyze_video(video_path):
    try:
        video_name = os.path.basename(video_path) #store the name of the video 
        print("Analyzing video:", video_name)
        extract_frames(video_path, output_image_dir)
        extract_faces(output_image_dir, face_dir)
        resize_images(face_dir, resized_image_dir)
        extract_and_save_hog(resized_image_dir, hog_dir) 

        predictions = [] #[fake, fake, real, fake]
        for img_name in os.listdir(resized_image_dir):
            if img_name.endswith(".jpg"):
                image_path = os.path.join(resized_image_dir, img_name)
                prediction = predict_with_pytorch(image_path)
                predictions.append(prediction)

        fake_score = np.mean(predictions)
        result = "fake" if fake_score > 0.5 else "real"
        print("Result:", result)
        return result

    except Exception as e:
        logging.error(f"An error occurred while analyzing the video: {e}")
        raise

# Flask routes
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS    #mp4

@app.route('/')
def upload_form():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':  #if file name is empty 
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        result = analyze_video(filepath)  # Call the analyze_video function
        return render_template('result.html', result=result, video_filename=filename)
    return redirect(request.url) 

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == "__main__":
    app.run(debug=True)


