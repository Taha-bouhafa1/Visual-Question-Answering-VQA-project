import os
from flask import Flask, render_template, request,Response
from werkzeug.utils import secure_filename
from ultralytics import YOLO
import shutil
import numpy as np
import base64
import cv2
from moviepy.editor import *
from base64 import b64decode
import json

app = Flask(__name__, static_url_path='/static')

model = YOLO("best (1).pt")

names = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N",
         "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]

def convert_video(input_path, output_path):
    video = VideoFileClip(input_path)
    video.write_videofile(output_path)
    
def process_detection_result(results, frame):
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            # Draw bounding box with a green color
            cv2.rectangle(frame, (x1, y1), (x2, y2), (50, 205, 50), 2)  # Green color, thickness 2

            # Prepare label
            conf = round(box.conf[0].item() * 100)
            cls = int(box.cls[0])
            label = f'{names[cls]}: {conf}%'  # Format label

            # Define text properties
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1  # Increase font size
            font_thickness = 2  # Increase font thickness
            text_size = cv2.getTextSize(label, font, font_scale, font_thickness)[0]
            text_width, text_height = text_size[0], text_size[1]

            # Draw filled rectangle behind the label
            cv2.rectangle(frame, (x1, y1 - text_height - 10), (x1 + text_width + 10, y1), (50, 205, 50), cv2.FILLED)  # Green color

            # Draw label text
            cv2.putText(frame, label, (x1 + 5, y1 - 5), font, font_scale, (255, 255, 255), font_thickness)

    return frame

def detect_objects(frame_data):
    try:
        frame_bytes = b64decode(frame_data.split(',')[1])
        frame_arr = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(frame_arr, cv2.IMREAD_COLOR)
        results = model.predict(frame,stream=True)
        annotated_frame = process_detection_result(results, frame)  
        _, buffer = cv2.imencode('.jpg', annotated_frame)
        frame_base64 = base64.b64encode(buffer).decode('utf-8')
        return json.dumps({"frame": frame_base64})
    except Exception as e:
        print("Error processing frame:", e)
        return None
    
UPLOAD_FOLDER = 'uploaded'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/detect', methods=["GET", "POST"])
def predict_img():
    if request.method == "POST":
        if 'file' not in request.files:
            return "No file part"
        
        file = request.files['file']
        
        if file.filename == '':
            return "No selected file"
        
        # Check if the file is an image or video
        if file.mimetype.startswith('image') or file.mimetype.startswith('video'):
            # Save the uploaded file
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            # Call model.predict() with the saved file path
            model.predict(source=file_path, save=True)
        else:
            return "Unsupported file type"
    
    # Initialize variable to store the path of the latest image or video
    latest_file_path = None
    
    # Check if the directory 'runs/detect' exists
    detect_dir = 'runs/detect'
    if os.path.exists(detect_dir):
        # Get a list of all subdirectories in the detect_dir
        subdirectories = [os.path.join(detect_dir, d) for d in os.listdir(detect_dir) if os.path.isdir(os.path.join(detect_dir, d))]
        
        # Sort the subdirectories by modification time (latest first)
        subdirectories.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        
        # Get the path of the latest subdirectory
        latest_subdirectory = subdirectories[0] if subdirectories else None
        
        if latest_subdirectory:
            # Get a list of all files in the latest subdirectory
            files = [f for f in os.listdir(latest_subdirectory) if os.path.isfile(os.path.join(latest_subdirectory, f))]
            
            # Filter files to find the latest image or video
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.mp4', '.avi', '.mov')):
                    latest_file_path = os.path.join(latest_subdirectory, file)
                    break  # Stop after finding the first image or video

    # Move the latest file to the static folder
    if latest_file_path:
        # Check if the latest file is an image or video
        if latest_file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            # If it's an image, move it to /static folder as Detected_img.jpg
            destination_path = os.path.join('static', 'Detected_img.jpg')
            shutil.move(latest_file_path, destination_path)
            latest_file_path = destination_path  # Update latest_file_path

        elif latest_file_path.lower().endswith(('.mp4', '.avi', '.mov')):
            # If it's a video, move it to /static/video folder as Detected_video.mp4
            destination_folder = 'static/video'
            destination_path = os.path.join(destination_folder, 'Detected_video.mp4')

            # Delete the existing Detected_video.mp4 if it exists
            if os.path.exists(destination_path):
                os.remove(destination_path)

            # If it's not already in MP4 format, convert it to MP4
            if not latest_file_path.lower().endswith('.mp4'):
                mp4_path = os.path.splitext(destination_path)[0] + '.mp4'
                convert_video(latest_file_path, mp4_path)
                destination_path = mp4_path  # Update destination path to MP4
            else:
                # Move the video to the destination folder
                shutil.move(latest_file_path, destination_path)

            latest_file_path = destination_path  # Update latest_file_path
    
        if file_path and os.path.exists(file_path):
            os.remove(file_path)
    
    # Remove the 'runs' folder
    if os.path.exists('runs'):
        shutil.rmtree('runs')
    # Pass the path of the latest image or video to the HTML template
    return render_template('detect.html', latest_file_path=latest_file_path)



@app.route('/stream')
def stream():
    return render_template('stream.html')


# Your Flask route for processing the detection
@app.route('/video_feed', methods=['POST'])
def video_feed():
    if 'frame' in request.form:
        frame_data = request.form['frame']
        frame_bytes = detect_objects(frame_data)
        if frame_bytes:
            return Response(frame_bytes, mimetype='image/jpeg')
        else:
            return Response("Error processing frame", status=500)

@app.route('/about')
def home3():
    return render_template('about.html')


if __name__ == '__main__':
    app.run(debug=True)

