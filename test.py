from sklearn.neighbors import KNeighborsClassifier
import cv2
import pickle
import numpy as np
import os
import pandas as pd
import csv
import time
from datetime import datetime

video=cv2.VideoCapture(1)
facedetect=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

import pickle as pkl
import pandas as pd

with open('data/names.pkl','rb') as f:
        LABELS= pickle.load(f)
        print(type(LABELS), LABELS)
with open('data/faces_data.pkl','rb') as f:
        FACES= pickle.load(f)

knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(FACES, LABELS)
column_name=['NAME','TIME']

while True:
    ret,frame=video.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=facedetect.detectMultiScale(gray,1.3,5,)
    for (x,y,w,h) in faces:
        crp_img=frame[y:y+h,x:x+w,:]
        resized_img=cv2.resize(crp_img,(50,50)).flatten().reshape((1,-1))
        output=knn.predict(resized_img)
        ts=time.time()
        date=datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
        timestamp=datetime.fromtimestamp(ts).strftime("%H:%M:%S")
        exist=os.path.isfile("Attendance/Attendance_" + date + ".csv")
        cv2.putText(frame, str(output[0]),(x,y-15),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(50,50,250),2)
        attendance=[str(output[0]),str(timestamp)]
    cv2.imshow("frame",frame)
    k=cv2.waitKey(1)
    if k==ord('o'):
             if exist:
                    with open("Attendance/Attendance_" + date + ".csv","+a") as csvfile:
                         writer=csv.writer(csvfile)
                         writer.writerow(attendance)  
                    csvfile.close()
             else:
                     with open("Attendance/Attendance_" + date + ".csv","+a") as csvfile:
                         writer=csv.writer(csvfile)
                         writer.writerow(column_name)
                         writer.writerow(attendance)  
                     csvfile.close()
    if k==ord('q'):
        break
video.release()
cv2.destroyAllWindows()






        