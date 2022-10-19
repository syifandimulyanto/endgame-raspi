import cv2
import numpy as np
import os 
from os import system
import requests
import json

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);

font = cv2.FONT_HERSHEY_SIMPLEX

#iniciate id counter
id = 0

# Initialize and start realtime video capture
cam = cv2.VideoCapture(0)
cam.set(3, 640) # set video widht
cam.set(4, 480) # set video height

# Define min window size to be recognized as a face
minW = 0.1*cam.get(3)
minH = 0.1*cam.get(4)

nrpFound = None
halloMsg = None
classMsg = None
classInfo = None

while True:
    ret, img =cam.read()
    # img = cv2.flip(img, -1) # Flip vertically
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
            id = "Wajah Tidak dikenali"
            nrpFound = None
            halloMsg = None
            classMsg = None
            classInfo = None

        confidence = "  {0}%".format(round(100 - confidence))
        
        # confidence = "  {0}%".format(round(100 - confidence))
        cv2.putText(img, str(id), (x+5,y-5), font, 1, (255,255,255), 2)
        cv2.putText(img, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1)

        if nrpFound != id and id != "Wajah Tidak dikenali":
            API_URL = str("http://127.0.0.1:8000/api/attend-with-nrp?nrp=" + str(id))
            response = requests.get(API_URL)
            rspData = (response.json())
            rspStatusCode = response.status_code
            
            print(API_URL)
            print(response.status_code)

            
            if (rspStatusCode == 200):
                studentFullName = str(rspData['student']['user']['first_name'] + " " + rspData['student']['user']['last_name'])
                halloMsg = str("Hallo "+ str(studentFullName)+" ("+ str(id) +")")
                rspCourse = rspData['attend']['student_schedule']['schedule']['course']
                classMsg = str("Absensi " + str(rspCourse['name']))
                classInfo = str("SUCCESS")

                writeImgFileName = str(id) + '-' + str(rspData['attend']['id']) + ".jpg"
                cv2.imwrite("attend/" + writeImgFileName, img)

                API_POST_IMAGE_URL = str("http://127.0.0.1:8000/api/attend-upload?nrp=" + str(id) + "&attend=" + str(rspData['attend']['id']) + "&filename=" + writeImgFileName)
                
                pathImg = "attend/" + writeImgFileName
                files = {'image': (writeImgFileName, open(pathImg, "rb"), 'multipart/form-data', {'Expires': '0'})}
                responsePostImage = requests.post(API_POST_IMAGE_URL,files=files)
                print(API_POST_IMAGE_URL)
                print(responsePostImage.status_code)
                print(responsePostImage.json())


            if rspStatusCode == 401:
                studentFullName = str(rspData['student']['user']['first_name'] + " " + rspData['student']['user']['last_name'])
                halloMsg = str("Hallo "+ str(studentFullName)+" ("+ str(id) +")")
                classMsg = str("Jadwal tidak ditemukan")
            if rspStatusCode == 402:
                studentFullName = str(rspData['student']['user']['first_name'] + " " + rspData['student']['user']['last_name'])
                halloMsg = str("Hallo "+ str(studentFullName)+" ("+ str(id) +")")
                rspCourse = rspData['attend']['student_schedule']['schedule']['course']
                classMsg = str("Anda Sedang berapa di kelas")    
                classInfo = str(rspCourse['name'])

            nrpFound = id

        if halloMsg != None:
            cv2.putText(img, halloMsg, (x+5,y+h+35), font, 1, (255,255,0), 1)
        if classMsg != None:    
            cv2.putText(img, classMsg, (x+5,y+h+75), font, 1, (255,255,0), 1)
        if classInfo != None:
            cv2.putText(img, classInfo, (x+5,y+h+105), font, 1, (255,255,0), 1)
            # cv2.putText(img,  "HALLO FANDI", (x+10,y-5), font, 1, (255,255,255), 3)  

            
    cv2.imshow('camera',img) 

    k = cv2.waitKey(10) & 0xff # Press 'ESC' for exiting video
    if k == 27:
        break



# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()