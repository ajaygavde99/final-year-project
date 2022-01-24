# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 19:52:43 2020

@author: ningesh
"""



from flask import Flask, render_template, request, session, url_for, redirect, jsonify
import pymysql
import pandas as pd
import os
app = Flask(__name__)
app.secret_key = 'random string'

#Database Connection
def dbConnection():
    connection = pymysql.connect(host="localhost", user="root", password="", database="resumeverification")
    return connection
def dbClose():
    dbConnection().close()
    return
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
from register import register_yourself


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


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    # Restrict TensorFlow to only allocate 1GB * 2 of memory on the first GPU
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024 * 2)])
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)

import cv2
import numpy as np
from PIL import Image
import os
import numpy as np
import cv2
import os
import h5py
#import dlib
from imutils import face_utils
from keras.models import load_model
import sys
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D,Dropout
from tensorflow.keras.layers import Dense, Activation, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import backend as K 
from sklearn.model_selection import train_test_split
from Model import model
from tensorflow.keras import callbacks

path = 'dataset'

recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml");


def downsample_image(img):
    img = Image.fromarray(img.astype('uint8'), 'L')
    img = img.resize((32,32), Image.ANTIALIAS)
    return np.array(img)



# function to get the images and label data
def getImagesAndLabels():
    
    path = 'facedata/'
    faceSamples=[]
    ids = []
    print(os.listdir(path))
    foldernames=os.listdir(path)#[name for name in os.listdir(path) if os.path.isdir(name)]
    print('foldernames',foldernames)
    labelslist=[]
    listoffolders={}
    cntof=0
    for ik in foldernames:
        listoffolders[ik]=cntof
        labelslist.append(ik)
        cntof+=1
    
    print('listoffolders',listoffolders)
    
    for i in foldernames:
        filelist=os.listdir(path+i)
        idofval=listoffolders.get(i)
        for filepath in filelist:
            wholepath=os.path.join(path+i,filepath)
            try:
                PIL_img = Image.open(wholepath).convert('L') # convert it to grayscale
            except:
                continue    
            img_numpy = np.array(PIL_img,'uint8')
        
            faceSamples.append(img_numpy)
            id =int(idofval)
            ids.append(id)
            
    #print('val is',faceSamples,ids,labelslist)
    return faceSamples,ids,labelslist  


def trainthemodel():
    print ("\n [INFO] Training faces now.")
    faces,ids,labelslist = getImagesAndLabels()

    K.clear_session()
    n_faces = len(set(ids))
    model1 = model((32,32,1),n_faces)
    faces = np.asarray(faces)
    faces = np.array([downsample_image(ab) for ab in faces])
    ids = np.asarray(ids)
    print('shape of face',[len(a) for a in faces])
    faces = faces[:,:,:,np.newaxis]
    print("Shape of Data: " + str(faces.shape))
    print("Number of unique faces : " + str(n_faces))


    ids = to_categorical(ids)

    faces = faces.astype('float32')
    faces /= 255.

    x_train, x_test, y_train, y_test = train_test_split(faces,ids, test_size = 0.01, random_state = 0)

    checkpoint = callbacks.ModelCheckpoint('trained_model.h5', monitor='val_acc',
                                           save_best_only=True, save_weights_only=True, verbose=1)
      
    print('length',len(y_train),len(y_test))                              
    model1.fit(x_train, y_train,
              batch_size=32,
              epochs=100,
              validation_data=(x_test, y_test),
              shuffle=True,callbacks=[checkpoint])
    model1.save('trained_model.h5')            

# Print the numer of faces trained and end program
    print("\n [INFO] " + str(n_faces) + " faces trained. Exiting Program")


def startingwork():
    _,ids,labelslist = getImagesAndLabels()
    model1 = model((32,32,1),len(set(ids)))
    model1.load_weights('trained_model.h5')
    model1.summary()
    cap = cv2.VideoCapture(0)
    listofpredictednames=[]
    print('here')
    ret = True

    clip = []
    font = cv2.FONT_HERSHEY_SIMPLEX
    cascPath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascPath)
    while ret:
        #read frame by frame
        ret, frame = cap.read()
        nframe = frame
        faces = faceCascade.detectMultiScale(
        frame,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30))

        try:
            (x,y,w,h) = faces[0]
        except:
            continue
        frame = frame[y:y+h,x:x+w]
        frame = cv2.resize(frame, (32,32))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow('result small' , frame)
        c= cv2.waitKey(1)
        if c & 0xFF == ord('q'):
            break
        
        #gray = gray[np.newaxis,:,:,np.newaxis]
        gray = gray.reshape(-1, 32, 32, 1).astype('float32') / 255.
        #print(gray.shape)
        prediction = model1.predict(gray)
        #print("prediction:" + str(prediction))

        
        print("\n\n\n\n")
        print("----------------------------------------------")
        #labels = ['ningesh' ,'Rishabh']
        prediction = prediction.tolist()
        
        listv = prediction[0]
        n = listv.index(max(listv))
        print("\n")
        print("----------------------------------------------")
        #print( "Highest Probability: " + labels[n] + "==>" + str(prediction[0][n]) )
        print( "Highest Probability: " + "User " + str(n) + "==>" + str(prediction[0][n]) )
        
        print("----------------------------------------------")
        print("\n")
        for (x, y, w, h) in faces:
            try:
                cv2.rectangle(nframe, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(nframe, str(labelslist[n]), (x+5,y-5), font, 1, (255,255,255), 2)
                listofpredictednames.append(str(labelslist[n]))
                #cv2.putText(nframe, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1)
            except:
                la = 2 
        prediction = np.argmax(model1.predict(gray), 1)
        print(prediction)
        cv2.imshow('result', nframe)
        c = cv2.waitKey(1)
        if c & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    return listofpredictednames

def identifypersoncnn():
    startingwork()

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

    
def callingrecognise(val):
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
            #cv2.imshow('frame with line drawn',gray)
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
                if cv2.contourArea(c) < val:
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
                if conf>=4 and conf <= 100:
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
                    #winsound.Beep(freq, duration)
                    '''if valof==0:
                        sendemailtouser('molawadevarshu98@gmail.com','unknown//1.jpg')
                        otpsendingfunction('8108385455',"unknown face detected in you camera")
                        otpsendingfunction('9819372525',"unknown face detected in you camera")
                        valof+=1
                        time.sleep(3)'''
            
                
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break 
    cap.release()
    cv2.destroyAllWindows()
#close DB connection



@app.route('/index')
@app.route('/')
def index():
    session['userloc']= request.args.get("location")
    locationis=session['userloc']
    print(locationis)
    return render_template('index.html')

@app.route('/startsurvilliencecam')
def startsurvilliencecam():
    if 'user' in session:
        val= request.args.get("thresh")
        callingrecognise(int(val))
    return redirect(url_for('index'))



@app.route('/trainthemodels', methods=['GET','POST'])
def trainthemodels():
    
    if request.method == "POST":
        trainthemodel()
        return 'success'
    if request.method == "GET":
        trainthemodel()
        return 'success'

@app.route('/home_after_registration', methods=['POST'])
def home_after_registration():
    
    if request.method == "POST":
        #id = request.form['Student_id']
        usernamelist =request.form["username"]
        company_name = request.form["company_name"]
        start_date = request.form["start_date"]
        End_date = request.form["End_date"]
        technlogy_worked = request.form["technlogy_worked"]
    
        con = dbConnection()
        cursor = con.cursor()
        sql = "INSERT INTO user_company_information (username, company_name, start_date, End_date, technology_worked) VALUES (%s, %s, %s, %s, %s)"
        val = (usernamelist, company_name, start_date, End_date, technlogy_worked)
        cursor.execute(sql, val)
        con.commit()
        dbClose()
        register_yourself(usernamelist)
    #flash("Registration Successful")
        return 'success'# render_template("index.html")

@app.route('/captureuserfaceandsavebyname',methods=["GET","POST"])
def captureuserfaceandsavebyname():
    firstFrame = None
    datasetpath='facedata//'
    if request.method == "POST":
        cap = cv2.VideoCapture(0)
        face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
        cnt=0
        usernamelist='ningesh123'
        usernamelist =request.form["username"]
        company_name = request.form["company_name"]
        start_date = request.form["start_date"]
        End_date = request.form["End_date"]
        technlogy_worked = request.form["technlogy_worked"]
    
        con = dbConnection()
        cursor = con.cursor()
        sql = "INSERT INTO user_company_information (username, company_name, start_date, End_date, technology_worked) VALUES (%s, %s, %s, %s, %s)"
        val = (usernamelist, company_name, start_date, End_date, technlogy_worked)
        cursor.execute(sql, val)
        con.commit()
        dbClose()
        
        #register_yourself(usernamelist)
        
        usernamelist =datasetpath+ request.form["username"]
        if not os.path.exists(usernamelist):
            os.makedirs(usernamelist)
    
        while(cap.isOpened()):
            ret, frame = cap.read()
            
            if ret == True:
                width = cap.get(3)   # float
                height = cap.get(4)
                surface = width * height
               
            #cv2.imshow('frame with line drawn',gray)
                print(width)
                text = "Unoccupied"
                frame = imutils.resize(frame, width=500)
                faces = face_cascade.detectMultiScale(frame,scaleFactor=1.1,minNeighbors=5,minSize=(30, 30),flags = cv2.CASCADE_SCALE_IMAGE)
            
    # show the frame and record if the user presses a key
            
         
         
                key = cv2.waitKey(1) & 0xFF
                for (x, y, w, h) in faces:
                    cv2.putText(frame, str(cnt),  (x, y) , cv2.FONT_HERSHEY_SIMPLEX, 2, 255)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    rectangleframe=frame[y:y+h, x:x+w]
                    cv2.imwrite(os.path.join(usernamelist , str(cnt)+'.jpg'), rectangleframe)
                    cnt=cnt+1
                    print(cnt)
                
               
            #cv2.imshow('frame with line drawn',frame)
                
            #cv2.imwrite('image'+str(cnt)+'.png',roi_color)
                cv2.imshow("Security Feed", frame) 
            if cnt>25:
                cnt=0
                break
            if cv2.waitKey(20) & 0xFF == ord('q'):
                break 
        cap.release()
        cv2.destroyAllWindows()
        return render_template('home.html', user=session['user']) 
    #'Success'


@app.route('/attendance', methods=['GET', 'POST'])
def attendance():
    
    nameofusers=startingwork()
    nameofuser=nameofusers[0]
    
    #nameofuser = mark_your_attendance()
    con=dbConnection()
    cursor2 = con.cursor()
        #cursor2.execute('SELECT item, quantity, metrics, calorie, carbohydrates, protein,fat  FROM calorie1 ORDER BY RAND() limit 5')
        #cursor2.execute("SELECT item, quantity, metrics, calorie, carbohydrates, protein,fat  FROM calorie1new where vegornonveg = %s and at = 'd' and type='"+str(opfor)+"' and disease not like '"+diseaseobt+"' ORDER BY RAND()", (vegornonveg,))
    cursor2.execute("select * from user_company_information where username='"+nameofuser+"'")
    print("select * from user_company_information where username='"+nameofuser+"'")
    res11 = cursor2.fetchall()
    li2=[]
    for a in res11:
        li2.append(a)
    dbClose()
    #print('list is',li2)
    return render_template('output.html',data=li2)



@app.route('/recognitionofperson')
def recognitionofperson():
    train_dir = 'facedata/'
    val_dir = 'facedata/'
    con = dbConnection()
    cursor = con.cursor()
    namedir=os.listdir(train_dir)
    print(namedir)
    lengthofclasses=len(namedir)

    train_datagen = ImageDataGenerator(rescale=1./255)
    val_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=(48,48),
            batch_size=64,
            color_mode="grayscale",
            class_mode='categorical')

    validation_generator = val_datagen.flow_from_directory(
            val_dir,
            target_size=(48,48),
            batch_size=64,
            color_mode="grayscale",
            class_mode='categorical')

    emotion_model = Sequential()

    emotion_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
    emotion_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
    emotion_model.add(Dropout(0.25))

    emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
    emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
    emotion_model.add(Dropout(0.25))

    emotion_model.add(Flatten())
    emotion_model.add(Dense(1024, activation='relu'))
    emotion_model.add(Dropout(0.5))
    emotion_model.add(Dense(lengthofclasses, activation='softmax'))
# emotion_model.load_weights('emotion_model.h5')

    cv2.ocl.setUseOpenCL(False)
    cap = cv2.VideoCapture(0)
    nameofuser=''
    a_dict = {}

    while True:
    # Find haar cascade to draw bounding box around face
        ret, frame = cap.read()
        if not ret:
            break
        bounding_box = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        num_faces = bounding_box.detectMultiScale(gray_frame,scaleFactor=1.3, minNeighbors=5)
        
        
        for (x, y, w, h) in num_faces:
            cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
            roi_gray_frame = gray_frame[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)
            emotion_prediction = emotion_model.predict(cropped_img)
            maxindex = int(np.argmax(emotion_prediction))
            namefound=namedir[maxindex]
            nameofuser=namefound
            print(namefound)
            if namefound in a_dict:
                a_dict[namefound] += 1
            else:
                a_dict[namefound] = 1
            result_count = cursor.execute('SELECT * FROM user_company_information WHERE username = %s',(namefound))
            print('SELECT * FROM user_company_information WHERE username = %s',(namefound))
            res = cursor.fetchone()
            #print(res)
            userinfo=''
            if result_count > 0:
                print('inside')
                print(result_count)
            
                userinfo =namefound+'\n'+ str(res[0])+'\n'+ res[1]+'\n'+ res[2]+'\n'+ res[3]+'\n'+ res[4]+'\n'+ res[5]
            #cv2.putText(frame,namedir[maxindex]+str(maxindex), (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            y0, dy = 50, 50
            for i, line in enumerate(userinfo.split('\n')):
                y = y0 + i*dy
                cv2.putText(frame, line, (50, y ), cv2.FONT_HERSHEY_SIMPLEX, 1,(255, 0, 0), 2, cv2.LINE_AA)
            #cv2.putText(frame,userinfo, (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow('Video', cv2.resize(frame,(1200,860),interpolation = cv2.INTER_CUBIC))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    dbClose()
    cap.release()
    cv2.destroyAllWindows()
    Keymax = max(a_dict, key=a_dict.get) 
    print(Keymax) 
    print('dict is ',a_dict)
    nameofuser=Keymax
    cursor2 = con.cursor()
        #cursor2.execute('SELECT item, quantity, metrics, calorie, carbohydrates, protein,fat  FROM calorie1 ORDER BY RAND() limit 5')
        #cursor2.execute("SELECT item, quantity, metrics, calorie, carbohydrates, protein,fat  FROM calorie1new where vegornonveg = %s and at = 'd' and type='"+str(opfor)+"' and disease not like '"+diseaseobt+"' ORDER BY RAND()", (vegornonveg,))
    cursor2.execute("select * from user_company_information where username='"+nameofuser+"'")
    print("select * from user_company_information where username='"+nameofuser+"'")
    res11 = cursor2.fetchall()
    li2=[]
    for a in res11:
        li2.append(a)
    #print('list is',li2)
    return render_template('output.html',data=li2)
    
    
@app.route('/traningdataset')
def traningdataset():
    train_dir = 'facedata/'
    val_dir = 'facedata/'

    namedir=os.listdir(train_dir)
    print(namedir)
    lengthofclasses=len(namedir)

    train_datagen = ImageDataGenerator(rescale=1./255)
    val_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=(48,48),
            batch_size=64,
            color_mode="grayscale",
            class_mode='categorical')

    validation_generator = val_datagen.flow_from_directory(
            val_dir,
            target_size=(48,48),
            batch_size=64,
            color_mode="grayscale",
            class_mode='categorical')

    emotion_model = Sequential()

    emotion_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
    emotion_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
    emotion_model.add(Dropout(0.25))

    emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
    emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
    emotion_model.add(Dropout(0.25))

    emotion_model.add(Flatten())
    emotion_model.add(Dense(1024, activation='relu'))
    emotion_model.add(Dropout(0.5))
    emotion_model.add(Dense(lengthofclasses, activation='softmax'))
# emotion_model.load_weights('emotion_model.h5')

    cv2.ocl.setUseOpenCL(False)

#emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}


    emotion_model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=0.0001, decay=1e-6),metrics=['accuracy'])
    emotion_model_info = emotion_model.fit_generator(
            train_generator,
            steps_per_epoch=28709 // 64,
            epochs=1,
            validation_data=validation_generator,
            validation_steps=7178 // 64)
    emotion_model.save_weights('face_model.h5')
    
    return 'success'



@app.route('/home')
def home():
    if 'user' in session:
        return render_template('home.html', user=session['user'], s=list)
    return redirect(url_for('index'))


@app.route('/login', methods=["GET","POST"])
def login():
    msg = ''
    # ht_cnt = 0
    # toi_cnt = 0
    # ie_cnt = 0
    # d = {'ht': 0, 'toi': 0, 'ie': 0}
    # b= {}
    if request.method == "POST":
        # session.pop('user',None)
        mobno = request.form.get("mobile")
        password = request.form.get("pas")
        con = dbConnection()
        cursor = con.cursor()
        result_count = cursor.execute('SELECT * FROM userdetails WHERE mobileno = %s AND password = %s',(mobno, password))
        res = cursor.fetchone()
        print(res)
        if result_count > 0:
            print(result_count)
            session['user'] = mobno
            session['uid'] = res[0]
            # ht_cnt = res[6]
            # toi_cnt = res[7]
            # ie_cnt = res[8]
            # d['ht'] = ht_cnt
            # d['toi'] = toi_cnt
            # d['ie'] = ie_cnt
            # print(d)
            # a = sorted(d.items(), key=lambda x: x[1], reverse=True)
            # b.update(a)
            # print(b)
            # list = []
            # for key in b.keys():
            #     list.append(key)
            # print(list)

            # session['sorted_dict']= list
            return redirect(url_for('home'))
        else:
            print(result_count)
            msg = 'Incorrect username/password!'
            return render_template('login.html')
    return render_template('login.html')

@app.route('/register', methods=["GET","POST"])
def register():
    print("register")
    if request.method == "POST":
        try:
            name = request.form.get("name")
            address = request.form.get("address")
            mailid = request.form.get("mailid")
            mobile = request.form.get("mobile")
            pass1 = request.form.get("pass1")
            con = dbConnection()
            cursor = con.cursor()
            cursor.execute('SELECT * FROM userdetails WHERE mobile = %s', (mobile))
            res = cursor.fetchone()
            if not res:
                sql = "INSERT INTO userdetails (name, address, email, mobile, password) VALUES (%s, %s, %s, %s, %s)"
                val = (name, address, mailid, mobile, pass1)
                cursor.execute(sql, val)
                con.commit()
                
                sql1 = "INSERT INTO readingcount (uid, ht_count, toi_count, ie_count) VALUES (%s, %s, %s, %s)"
                val1 = (mobile,int(0),int(0),int(0))
                cursor.execute(sql1, val1)
                con.commit()
                status= "success"
                return redirect(url_for('index'))
            else:
                status = "Already available"
            return status
        except Exception as inst:
            print(inst)
            print("Exception occured at user registration")
            return redirect(url_for('index'))
        finally:
            dbClose()
    return render_template('register.html')



#logout code
@app.route('/logout')
def logout():
    session.pop('user')
    return redirect(url_for('index'))

#getImagesAndLabels()
if __name__ == '__main__':
    app.run('0.0.0.0')
    #app.run()
