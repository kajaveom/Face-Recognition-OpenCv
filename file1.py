import cv2
import numpy as np
import face_recognition
 
mytrain = face_recognition.load_image_file('ImagesBasic/mytrain.jpg')  #Function to load images

mytrain = cv2.cvtColor(mytrain,cv2.COLOR_BGR2RGB)  #We need to convert it BGR to RGB

mytest = face_recognition.load_image_file('ImagesBasic/mytest.jpg')
mytest = cv2.cvtColor(mytest,cv2.COLOR_BGR2RGB)
 

print(face_recognition.face_locations(mytrain))



faceLoc = face_recognition.face_locations(mytrain)[0]  #Since only one image is there the first index will hold the face loaction
encodetrain = face_recognition.face_encodings(mytrain)[0]     #Encodes the image parameters
print(encodetrain)
cv2.rectangle(mytrain,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2) 
 
print("\n\n")
faceLocTest = face_recognition.face_locations(mytest)[0]
encodetest = face_recognition.face_encodings(mytest)[0]
print(encodetest)
cv2.rectangle(mytest,(faceLocTest[3],faceLocTest[0]),(faceLocTest[1],faceLocTest[2]),(255,0,255),2)
 
results = face_recognition.compare_faces([encodetrain],encodetest)        
faceDis = face_recognition.face_distance([encodetrain],encodetest)   #It will give the distance between the train and test images.
print(results,faceDis)

cv2.putText(mytest,f'{results} {round(faceDis[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
 
cv2.imshow('Train Image',mytrain)
cv2.imshow('Test Image',mytest)
cv2.waitKey(0)


