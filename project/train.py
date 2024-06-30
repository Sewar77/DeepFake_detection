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
from torchvision import models, datasets
from torch.utils.data import DataLoader
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
data_dir = "C:/Users/User/Desktop/tests python/Data Set/FF_Face_only_data"
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
        capture = cv.VideoCapture(video_file)
        if not capture.isOpened():
            raise IOError("Error opening video file:", video_file)
        total_frames = int(capture.get(cv.CAP_PROP_FRAME_COUNT))
        frame_rate_video = capture.get(cv.CAP_PROP_FPS)
        sampling_interval = int(frame_rate_video / frame_rate)
        frame_count = 0
        success = True
        while success:
            success, frame = capture.read()
            if not success:
                break
            if frame_count % sampling_interval == 0:
                frame_count += 1
                cv.imwrite(os.path.join(output_dir, f"frame_{frame_count}.jpg"), frame)
            else:
                frame_count += 1
        capture.release()
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        raise

# Extract faces from images
def extract_faces(image_dir, output_dir):
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
            img_width = int(img.shape[1] * scale)
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
            hog_image_uint8 = hog_image.astype(np.uint8)
            cv.imwrite(os.path.join(output_dir, f"hog_{img_name}"), hog_image_uint8)
            plt.figure(figsize=(8, 8))
            plt.axis('off')
            plt.title(f'HOG for {img_name}')
            plt.imshow(hog_image_uint8, cmap='gray')
            plot_path = os.path.join(output_dir, f"hog_plot_{img_name}.png")
            plt.savefig(plot_path)
            plt.close()

# Load and modify pre-trained ResNet model
model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 2)
model = model.to(device)

# Define loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Define data transformations
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
}

# Ensure dataset path is correct and accessible
if not os.path.exists(data_dir):
    raise FileNotFoundError(f"Dataset directory {data_dir} does not exist")

# Load datasets
try:
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                      for x in ['train', 'val']}
except Exception as e:
    logging.error(f"An error occurred while loading the datasets: {e}")
    raise

# Create data loaders
dataloaders = {x: DataLoader(image_datasets[x], batch_size=32, shuffle=True, num_workers=4)
               for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

# Train the model
def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, num_epochs=25):
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        if epoch % 5 == 0:
            torch.save(model.state_dict(), model_path)

    return model

# Perform image preprocessing
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(image).unsqueeze(0).to(device)

# Perform inference using the trained model
def predict_with_pytorch(image_path):
    image = Image.open(image_path).convert("RGB")
    image_tensor = preprocess_image(image)
    with torch.no_grad():
        output = model(image_tensor)
    _, predicted = torch.max(output, 1)
    return predicted.item()

# Analyze video function
def analyze_video(video_path):
    try:
        video_name = os.path.basename(video_path)
        print("Analyzing video:", video_name)
        extract_frames(video_path, image_file)
        extract_faces(image_file, face_dir)
        resize_images(face_dir, resized_image_dir)
        extract_and_save_hog(resized_image_dir, hog_dir)

        predictions = []
        for img_name in os.listdir(resized_image_dir):
            if img_name.endswith(".jpg"):
                image_path = os.path.join(resized_image_dir, img_name)
                prediction = predict_with_pytorch(image_path)
                predictions.append(prediction)

        fake_score = np.mean(predictions)
        result = "fake" if fake_score > 0.4 else "real"
        print("Result:", result)
        return result

    except Exception as e:
        logging.error(f"An error occurred while analyzing the video: {e}")
        raise

# Flask routes
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def upload_form():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        result1 = analyze_video(filepath)
        return render_template('result.html', result=result1, video_filename=filename)
    return redirect(request.url)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == "__main__":
    if not os.path.exists(model_path):
        model = train_model(model, dataloaders, dataset_sizes, criterion, optimizer, num_epochs=25)
        torch.save(model.state_dict(), model_path)
    else:
        model.load_state_dict(torch.load(model_path))
        model.eval()
    app.run(debug=True)
