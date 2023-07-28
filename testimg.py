import cv2
import copy
import numpy as np
import face_recognition
import os
import pickle
from datetime import datetime , date

encodeListKnown = pickle.load(open("encodeListKnown.dat", "rb") )
classNames = pickle.load(open("classNames.dat", "rb") )

def markAttendance(id):
    today = date.today()
    tdate = today.strftime("%b-%d-%Y")
    with open(f'Attendance_DB/{tdate}.csv','r+') as f:    # read and write at the same time (r+)
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        name, reg = id.split('_')
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{reg},{dtString}')



cap = cv2.VideoCapture(0)   #initializes the camera
 
while True:
    success, img = cap.read()
    #img = captureScreen()
    imgS = cv2.resize(img,(0,0),None,0.25,0.25)         #Resize the image for better suit the process
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
 
    facesCurFrame = face_recognition.face_locations(imgS)     #Finds all locations of the faces in image
    encodesCurFrame = face_recognition.face_encodings(imgS,facesCurFrame)      #Gets the encoded parameters from all the images


    for encodeFace,faceLoc in zip(encodesCurFrame,facesCurFrame):

        matches = face_recognition.compare_faces(encodeListKnown,encodeFace)   
        faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)
        print(faceDis)
        matchIndex = np.argmin(faceDis)   #Gives the best match comparing the distance

        
        
        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            print(name)
            
            y1,x2,y2,x1 = faceLoc
            y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            markAttendance(name)
            
 
    cv2.imshow('Webcam',img)
    cv2.waitKey(1)