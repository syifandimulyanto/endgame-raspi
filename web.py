from flask import Flask, render_template, Response
import cv2
import numpy as np
import os 

app = Flask(__name__)



recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);

font = cv2.FONT_HERSHEY_SIMPLEX

# Initialize and start realtime video capture
camera = cv2.VideoCapture(0)  # use 0 for web camera
camera.set(3, 640) # set video widht
camera.set(4, 480) # set video height

# Define min window size to be recognized as a face
minW = 0.1*camera.get(3)
minH = 0.1*camera.get(4)

#iniciate id counter
id = 0

def gen_frames():  # generate frame by frame from camera
    while True:
        # Capture frame-by-frame
        success, img = camera.read()  # read the camera frame
        if not success:
            break
        else:
        
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            faces = faceCascade.detectMultiScale( 
                gray,
                scaleFactor = 1.2,
                minNeighbors = 5,
                minSize = (int(minW), int(minH)),
            )

            for(x,y,w,h) in faces:
                cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
                id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
                # Check if confidence is less them 100 ==> "0" is perfect match 
                if (round(100 - confidence) < 60):
                    id = "unknown"

                confidence = "  {0}%".format(round(100 - confidence))
                
                # confidence = "  {0}%".format(round(100 - confidence))
                cv2.putText(img, str(id), (x+5,y-5), font, 1, (255,255,255), 2)
                cv2.putText(img, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1)  

            ret, buffer = cv2.imencode('.jpg', img)
            frame = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result


@app.route('/video_feed')
def video_feed():
    #Video streaming route. Put this in the src attribute of an img tag
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)