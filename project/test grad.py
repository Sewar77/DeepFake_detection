import os
import cv2 as cv
import logging
import numpy as np
import matplotlib.pyplot as plt 
from flask import Flask, request, redirect, url_for, render_template, send_from_directory
from werkzeug.utils import secure_filename
from skimage.feature import hog
import skimage
print(skimage.__version__)
import torch
import torchvision.transforms as transforms
from torchvision import models
from sklearn.metrics import accuracy_score
from PIL import Image 

# Set the device to GPU if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
face_cascade_path = "C:/Users/User/Desktop/project/haarcascade_frontalface_default.xml"
image_file = "C:/Users/User/Desktop/project/image_file"
resized_image_dir = "C:/Users/User/Desktop/project/resized"
face_dir = "C:/Users/User/Desktop/project/face_file"
hog_dir = "C:/Users/User/Desktop/project/hog"
lbp_dir = "C:/Users/User/Desktop/project/lbp"
shape_predictor_path = "C:/Users/User/Desktop/grad proj/tests python/graduation project/shape_predictor_68_face_landmarks.dat"
model_path = "C:/Users/User/Desktop/grad proj/tests python/graduation project/deepfake/archive/xception-b5690688.pth"

# Flask configuration
UPLOAD_FOLDER = 'uploaded_videos' #specify the directory where uploaded video files will be stored.
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}
app = Flask(__name__) #This line initializes a new Flask web application instance by creating an app object. The __name__ argument tells Flask to use the current module or package's name, which is necessary for locating resources like templates and static files.
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
        total_frames = int(capture.get(cv.CAP_PROP_FRAME_COUNT)) # Get total number of frames
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
            image_path = os.path.join(image_dir, img_name)
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
            img_path = os.path.join(input_dir, img_name)
            img = cv.imread(img_path)
            img_width = int(img.shape[1] * scale) #shape(x,y)
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
            hog_image_uint8 = hog_image.astype(np.uint8) #converts it to an 8-bit unsigned integer format for saving as an image file.
            cv.imwrite(os.path.join(output_dir, f"hog_{img_name}"), hog_image_uint8)
            plt.figure(figsize=(8, 8))
            plt.axis('off')
            plt.title(f'HOG for {img_name}')
            plt.imshow(hog_image_uint8, cmap='gray')
            # Save the plot
            plot_path = os.path.join(output_dir, f"hog_plot_{img_name}.png")
            plt.savefig(lbp_dir)
            plt.close()

# Load pre-trained PyTorch model
#The pretrained=True argument ensures that the model weights are loaded from the pre-trained version rather than being randomly initialized.
# Load pre-trained ResNet model and modify the final layer
model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 2)  # Assuming binary classification (fake/real)
model = model.to(device)

# Define loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Load the model state if available
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path))

model.to(device) #Moves the model to the specified device, which can be either a GPU or CPU. The device is determined by the earlier line device = torch.device("cuda" if torch.cuda.is_available() else "cpu").
model.eval() #Sets the model to evaluation mode. 

# Preprocess image for PyTorch model
def preprocess_image(image):
    transform = transforms.Compose([   #Creates a composition of several image transformations
        transforms.Resize(256),    #Resizes the image to 256 pixels on its shorter side.
        transforms.CenterCrop(224), #Crops the center 224x224 region from the resized image.
        transforms.ToTensor(),  #Converts the image to a PyTorch tensor.
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),   #common for models trained on ImageNet
    ])
    return transform(image).unsqueeze(0).to(device)


# Perform inference using PyTorch model
def predict_with_pytorch(image_path):    
    image = Image.open(image_path).convert("RGB")
    image_tensor = preprocess_image(image)
    with torch.no_grad():  #Disables gradient computation, which reduces memory usage and speeds up inference.
        output = model(image_tensor) #Passes the preprocessed image tensor through the model to get the output.
    _, predicted = torch.max(output, 1) #Finds the class with the highest score in the output tensor.
    return predicted.item() #Returns the predicted class index as an integer.

# Analyze video
def analyze_video(video_path):
    try:
        video_name = os.path.basename(video_path)
        print("Analyzing video:", video_name)
        extract_frames(video_path, image_file)
        extract_faces(image_file, face_dir)
        resize_images(face_dir, resized_image_dir)
        extract_and_save_hog(resized_image_dir, hog_dir)

        predictions = [] #[fake, real, real, fake, real, real]
        for img_name in os.listdir(resized_image_dir):
            if img_name.endswith(".jpg"):
                image_path = os.path.join(resized_image_dir, img_name)
                prediction = predict_with_pytorch(image_path) #Predicts the class of the image using the PyTorch model.
                predictions.append(prediction)

        fake_score = np.mean(predictions) # Computes the mean of the predictions.
        result = "fake" if fake_score > 0.4 else "real"
        print("Result:", result)
        return result

    except Exception as e:
        logging.error(f"An error occurred while analyzing the video: {e}")
        raise

# Flask routes
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS   #MP4

@app.route('/')
def upload_form(): #Maps the root URL to the upload_form function.
    return render_template('upload.html')

@app.route('/upload', methods=['POST']) #allowing only POST requests.

def upload_video():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename) # Secures the filename.
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        result1 = analyze_video(filepath)  # Call the analyze_video function
        return render_template('result.html', result=result1, video_filename=filename)
    return redirect(request.url)    #Renders the result.html template with the analysis result and video filename.

@app.route('/uploads/<filename>')
def uploaded_file(filename): 
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename) #Redirects back to the upload form if

if __name__ == "__main__":
    app.run(debug=True)



















