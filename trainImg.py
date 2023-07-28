import cv2
import copy
import numpy as np
import face_recognition
import os
import pickle
from datetime import datetime , date

os.system("cls")  

path = 'trainImages'     #Path for training images

images = []
myList = os.listdir(path)    #Containns all the folders of images

classNames = []
encodeListKnown = []

#pickle.dump(encodeListKnown, open("encodeListKnown.dat","wb") )

def find_centroids(myList):
    for fold in myList:
        ids = os.listdir(f'{path}/{fold}')
        classNames.append(fold)
        final_encode = []
        
        for cl in ids:
            curImg = cv2.imread(f'{path}/{fold}/{cl}')
            #images.append(curImg)
            img = cv2.cvtColor(curImg,cv2.COLOR_BGR2RGB)          #Pixel Orderings (different)
            try:
                encode = face_recognition.face_encodings(img)[0]             #If face is not detected it will go to next image.
            except:
                continue
            if(len(final_encode) == 0 ):
                final_encode = copy.deepcopy(encode)
        
            for i in range(0, len(encode)):
                final_encode[i] = ( final_encode[i] + encode[i] )/2

        encodeListKnown.append(final_encode)
        
#Time complexity = (total no. of students) * (Total no. of images for each student)
    
find_centroids(myList)
print("Images Trained \n\n")


for x in encodeListKnown:
    print("\n\n")
    print(x)



pickle.dump(classNames, open("classNames.dat","wb") )
pickle.dump(encodeListKnown, open("encodeListKnown.dat","wb") )

