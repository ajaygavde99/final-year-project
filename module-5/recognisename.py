# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 19:58:28 2019

@author: Sumit
"""
import os
import numpy as np
import cv2
import imutils
from collections import deque
import pickle
import winsound
import urllib
import time
import datetime
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
#eye_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_eye.xml')
#smile_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_smile.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("./recognizers/face-trainner1.yml")

import smtplib 
from email.mime.multipart import MIMEMultipart 
from email.mime.text import MIMEText 
from email.mime.base import MIMEBase 
from email import encoders 
import winsound
duration = 1000  # milliseconds
freq = 1040  # Hz
winsound.Beep(freq, duration)
def sendemailtouser(usertoaddress,filetosend):   
    fromaddr = "beprojectcomputer@gmail.com"
    toaddr = usertoaddress#"ningesh1406@gmail.com"
   
    #instance of MIMEMultipart 
    msg = MIMEMultipart() 
  
    # storing the senders email address   
    msg['From'] = fromaddr 
  
    # storing the receivers email address  
    msg['To'] = toaddr 
  
    # storing the subject  
    msg['Subject'] = "alarm for unknown person"
  
    # string to store the body of the mail 
    body = "Unknown person"
  
    # attach the body with the msg instance 
    msg.attach(MIMEText(body, 'plain')) 
  
    # open the file to be sent  
    filename = filetosend
    attachment = open(filetosend, "rb") 
  
    # instance of MIMEBase and named as p 
    p = MIMEBase('application', 'octet-stream') 
  
    # To change the payload into encoded form 
    p.set_payload((attachment).read()) 
  
    # encode into base64 
    encoders.encode_base64(p) 
   
    p.add_header('Content-Disposition', "attachment; filename= %s" % filename) 
  
    # attach the instance 'p' to instance 'msg' 
    msg.attach(p) 
  
    # creates SMTP session 
    s = smtplib.SMTP('smtp.gmail.com', 587) 
  
    # start TLS for security 
    s.starttls() 
  
    # Authentication 
    s.login(fromaddr, "beproject1") 
  
    # Converts the Multipart msg into a string 
    text = msg.as_string() 
  
    # sending the mail 
    s.sendmail(fromaddr, toaddr, text) 
  
    # terminating the session 
    s.quit() 
def otpsendingfunction(mobile,msgtosend):
    authkey = "175606AVhvZO37X59c2613b"  # Your authentication key.
    mobiles = mobile  # Multiple mobiles numbers separated by comma.
    message = msgtosend#"unknown face detected in you camera"  # Your message to send.
    sender = "ALARMF"  # Sender ID,While using route4 sender id should be 6 characters long.
    route = "route4"  # Define route
    # Prepare you post parameters
    values = {
        'authkey': authkey,
        'mobiles': mobiles,
        'message': message,
        'sender': sender,
        'route': route
    }
    url = "http://api.msg91.com/api/sendhttp.php"  # API URL
    postdata = urllib.parse.urlencode(values).encode("utf-8")  # URL encoding the data here.
    req = urllib.request.Request(url, postdata)
    response = urllib.request.urlopen(req)
    output = response.read()  # Get Response
    print(output)

labels = {"person_name": 1}
with open("pickles1/face-labels.pickle", 'rb') as f:
    og_labels = pickle.load(f)
    labels = {v:k for k,v in og_labels.items()}
cnt=0
valof=0
# initialize the first frame in the video stream
firstFrame = None
while(cap.isOpened()):
    ret, frame = cap.read()
    cnt=cnt+1
    print(cnt)
    if ret == True:
        width = cap.get(3)   # float
        height = cap.get(4)
        surface = width * height
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow('frame with line drawn',gray)
        faces = face_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(30, 30),flags = cv2.CASCADE_SCALE_IMAGE)
        print(width)
        text = "Unoccupied"
        frame = imutils.resize(frame, width=500)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        blur = cv2.GaussianBlur(frame, (21, 21), 0)
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
 
        lower = [18, 50, 50]
        upper = [35, 255, 255]
        lower = np.array(lower, dtype="uint8")
        upper = np.array(upper, dtype="uint8")
        mask = cv2.inRange(hsv, lower, upper)
    
 
 
        output = cv2.bitwise_and(frame, hsv, mask=mask)
        no_red = cv2.countNonZero(mask)
#         cv2.imshow("output", output)
    #print("output:", frame)
#         if int(no_red) > 20000:
#             print(no_red)
#             if valof==0:
#                     #sendemailtouser('ningesh1406@gmail.com','unknown//1.jpg')
#                     otpsendingfunction('8108385455',"fire detected at your place")
#                     time.sleep(3)
#             print ('Fire detected')

    # if the first frame is None, initialize it
        if firstFrame is None:
            firstFrame = gray
            continue

    # compute the absolute difference between the current frame and
    # first frame
        frameDelta = cv2.absdiff(firstFrame, gray)
        thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]

    # dilate the thresholded image to fill in holes, then find contours
    # on thresholded image
        thresh = cv2.dilate(thresh, None, iterations=2)
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

    # loop over the contours
        for c in cnts:
        # if the contour is too small, ignore it
            if cv2.contourArea(c) < 1000:
                continue
        
        
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            text = "Occupied"

    # draw the text and timestamp on the frame
        cv2.putText(frame, "Room Status: {}".format(text), (10, 20),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
        (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

    # show the frame and record if the user presses a key
        cv2.imshow("Security Feed", frame)
        cv2.imshow("Thresh", thresh)
        cv2.imshow("Frame Delta", frameDelta)
        key = cv2.waitKey(1) & 0xFF
        
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y + h, x:x + w] 
            #cv2.imshow('frame with line drawn',frame)
            id_, conf = recognizer.predict(roi_gray)
            print("confidence is "+str(conf))
            #cv2.imwrite('image'+str(cnt)+'.png',roi_color)
            if conf>=4 and conf <= 85:
 #print(5: #id_)
 #print(labels[id_])
                font = cv2.FONT_HERSHEY_SIMPLEX
                name = labels[id_]
                color = (255, 255, 255)
                stroke = 2
                cv2.putText(frame, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)

                #img_item = "7.png"
                #cv2.imwrite(img_item, roi_color)

                color = (255, 0, 0) #BGR 0-255 
                stroke = 2
                end_cord_x = x + w
                end_cord_y = y + h
                cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, stroke)
                cv2.imshow('frame with line drawn',frame)
            else:
                rectangleframe=frame[y:y+h, x:x+w]
                #cv2.rectangle(frame, (x, y), (x + w, y + h), color, stroke)
                #cv2.imshow('frame with line drawn',frame)
                cv2.imwrite(os.path.join("unknown" , str(1)+'.jpg'), rectangleframe)
                duration = 1000  # milliseconds
                freq = 440  # Hz
                winsound.Beep(freq, duration)
                if valof==0:
                    sendemailtouser('molawadevarshu98@gmail.com','unknown//1.jpg')
                    otpsendingfunction('8108385455',"unknown face detected in you camera")
                    otpsendingfunction('9819372525',"unknown face detected in you camera")
                    valof+=1
                    time.sleep(3)
                
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break 
cap.release()
cv2.destroyAllWindows()
