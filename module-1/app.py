from flask import Flask, render_template, request, session, url_for, redirect, jsonify,make_response
import pymysql
from pyresparser import ResumeParser
from werkzeug.utils import secure_filename
from models.keras_first_go import KerasFirstGoModel
from clear_bash import clear_bash
import os
from FacebookPostsScraper import FacebookPostsScraper as Fps
from pprint import pprint as pp
from selenium import webdriver 
from time import sleep 
from webdriver_manager.chrome import ChromeDriverManager 
from selenium.webdriver.chrome.options import Options  
from selenium.common.exceptions import NoSuchElementException  
import utils
from bs4 import BeautifulSoup
from selenium.webdriver.common.keys import Keys
import io
from bs4 import BeautifulSoup
import pickle
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
#from withoutui import startcamera 
import cv2
from PIL import Image, ImageTk
import os
import numpy as np

import numpy as np
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import backend as K
import tensorflow as tf


import random
filename_keras='my_model_s.h5'
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
def getrecommendedjob(skills):
    #first_go_model = KerasFirstGoModel()
    #print(skills)
    data=pd.read_csv('data/25_cleaned_job_descriptions.csv',names = ['Query', 'Description'],header = 0)
    train, test = train_test_split(data, test_size=0.2)
    test_labels = test['Query']
    lst=skills
    print(lst)
    from keras.preprocessing.sequence import pad_sequences
    from keras.preprocessing.text import one_hot
    encoded_docs = [one_hot(lst,500,)]
        # pad documents to a max length
    padded_text = pad_sequences(encoded_docs, maxlen=500, padding='post')
        # Prediction based on model
    processed_text=''
    
    loaded_model_nn = tf.keras.models.load_model(filename_keras)
    processed_text = loaded_model_nn.predict(padded_text)  
    encoder = LabelBinarizer()
    encoder.fit(test_labels)
    result = encoder.inverse_transform(processed_text)
    #result[0]
    result = {'Job': result[0]}
    print(result)
    return result





import os
import cv2
import numpy as np
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing import image
#cv2.ocl.setUseOpenCL(False)

emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

emotionlistforadding=['Angry','Disgust','Fearful','Happy','Neutral','Sad','Surprise']
emoji_dist={0:"./emojis/angry.png",2:"./emojis/disgusted.png",2:"./emojis/fearful.png",3:"./emojis/happy.png",4:"./emojis/neutral.png",5:"./emojis/sad.png",6:"./emojis/surpriced.png"}
global last_frame1                                    
last_frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
global cap1
show_text=[0]
global listofemotion
listofclasses=['ENFJ',
 'ENFP',
 'ENTJ',
 'ENTP',
 'ESFJ',
 'ESFP',
 'ESTP',
 'INFJ',
 'INFP',
 'INTJ',
 'INTP',
 'ISFJ',
 'ISFP',
 'ISTJ',
 'ISTP']
Corpus = pd.read_csv(r"processed_data.csv",encoding='latin-1',nrows=10,error_bad_lines=False)
Tfidf_vect = TfidfVectorizer(max_features=5000)
Tfidf_vect.fit(Corpus['text_final'])



app = Flask(__name__)
app.secret_key = 'random string'
cleaner=clear_bash()

#app.config[‘UPLOAD_FOLDER’]=
app.config['UPLOADED_FILE'] = 'static/ResumeFiles'

filename='naive_bayes_model.sav'
loaded_model = pickle.load(open(filename, 'rb'))

usr='+917350242165'#input('Enter Email Id:')  
pwd='Ajay6478'#input('Enter Password:')  
usrlnk='ajgavde@acpce.ac.in'#input('Enter Email Id:')  
pwdlnk='ajay6478'#input('Enter Password:')  

fbpostdata=''
linkedindata=''

#train_model()
#processed_text = first_go_model.prediction("Oracle Soap Sdlc C Engineering Opencv Architecture Android Sql Java Html Database Agile Technical")
#result = {'Job': processed_text}
#print(result)


import pyaudio
import wave


     
     
#Database Connection
def dbConnection():
    connection = pymysql.connect(host="localhost", user="root", password="", database="resumeverification")
    return connection


#close DB connection
def dbClose():
    dbConnection().close()
    return


@app.route('/index')
@app.route('/')
def index():
    return render_template('index.html')

#logout code
@app.route('/logout')
def logout():
    session.pop('user')
    return redirect(url_for('index'))


#login code
@app.route('/login', methods=["GET","POST"])
def login():
    msg = ''
    if request.method == "POST":
        session.pop('user',None)
        mailid = request.form.get("mailid")
        password = request.form.get("pas")
        #print(mobno+password)
        con = dbConnection()
        cursor = con.cursor()
        result_count = cursor.execute('SELECT * FROM userdetails WHERE emailid = %s AND password = %s', (mailid, password))
        #a= 'SELECT * FROM userdetails WHERE mobile ='+mobno+'  AND password = '+ password
        #print(a)
        #result_count=cursor.execute(a)
        # result = cursor.fetchone()
        if result_count>0:
            print(result_count)
            session['user'] = mailid
            return redirect(url_for('home'))
        else:
            print(result_count)
            msg = 'Incorrect username/password!'
            return msg
        #dbClose()
    return redirect(url_for('index'))

@app.route('/FBLinkuploader', methods = ['GET', 'POST'])
def FBLinkuploader():
    if 'user' in session:
        if request.method == 'POST':
            fblink = request.form.get("fblink")
            #fbposts = FB_post_fetch(fblink) 
            fbposts=fetchingscrapinguserdata(fblink)
            
            return render_template('FetchedPost.html', data=fbposts, user=session['user'])
        return render_template('home.html', user=session['user'])
    return redirect(url_for('index'))

@app.route('/LinkedinLinkuploader', methods = ['GET', 'POST'])
def LinkedinLinkuploader():
    if 'user' in session:
        if request.method == 'POST':
            link = request.form.get("link")
            print('link is',link)
            #fbposts = FB_post_fetch(fblink) 
            fbposts=LinkedInfetchingscrapinguserdata(link)
            
            return render_template('FetchedPost.html', data=fbposts, user=session['user'])
        return render_template('home.html', user=session['user'])
    return redirect(url_for('index'))

#user register code
@app.route('/userRegister', methods=["GET","POST"])
def userRegister():
    if request.method == "POST":
        try:
            status=""
            name = request.form.get("name")
            address = request.form.get("address")
            mailid = request.form.get("mailid")
            mobile = request.form.get("mobile")
            pass1 = request.form.get("pass1")
            con = dbConnection()
            cursor = con.cursor()
            cursor.execute('SELECT * FROM userdetails WHERE emailid = %s', (mailid))
            res = cursor.fetchone()
            #res = 0
            if not res:
                sql = "INSERT INTO userdetails (name, address, emailid, mobileno, password) VALUES (%s, %s, %s, %s, %s)"
                val = (name, address, mailid, mobile, pass1)
                print(sql," ",val)
                cursor.execute(sql, val)
                con.commit()
                status= "success"
                return redirect(url_for('index'))
            else:
                status = "Already available"
            #return status
            return redirect(url_for('index'))
        except:
            print("Exception occured at user registration")
            return redirect(url_for('index'))
        finally:
            dbClose()
    return redirect(url_for('index'))


@app.route('/home')
def home():
    if 'user' in session:

        return render_template('home.html', user=session['user'])
    return redirect(url_for('index'))

#import os

@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
    global fbpostdata,linkedindata
    if 'user' in session:
        if request.method == 'POST':
            f = request.files['file']
            print(f)
            filename = secure_filename(f.filename)
            f.save(os.path.join(app.config['UPLOADED_FILE'], filename))

            filename = os.path.abspath(app.config['UPLOADED_FILE']+"//"+filename)#os.path.abspath(filename)
            
            filename = filename#'ningesh.pdf'
 #           print("=======")
 #           print(filename)
 #           print("=======")
            listofcontent = ['pdf']
            splitfilename = filename.split(".")
            if splitfilename[1] in listofcontent:
                data = ResumeParser(filename).get_extracted_data()
                print(data)
                #print(type(['Soap', 'Agile', 'Sdlc', 'Java', 'Sql', 'C', 'Android', 'Engineering', 'Html', 'Architecture', 'Opencv', 'Database', 'Oracle', 'Technical']))
                                
                skillset=data['skills']
                print(skillset)
                dataop=''
                for ik in skillset:
                    dataop=dataop+ik+" "
                
                print('=======')
                print(dataop)
                #global first_go_model
                #processed_text1 = first_go_model.prediction("Oracle Soap Sdlc C Engineering Opencv Architecture Android Sql Java Html Database Agile Technical")
                #processed_text1 = first_go_model.prediction(dataop)
                
                processed_text1 = getrecommendedjob(dataop)
                result1 = {'Job type suited for this ': processed_text1}
                
                #result1 = {'Job type suited for this ': processed_text1}
                print(result1)
                
                processed_text12 = getrecommendedjob(linkedindata)
                resultlinkedin = {'Job type suited for this from linked in post': processed_text12}
                
                #result1 = {'Job type suited for this ': processed_text1}
                print('resultlinkedin',resultlinkedin)
                predictions_RF = loaded_model.predict(Tfidf_vect.transform([fbpostdata]))
                #print(listofclasses[predictions_RF[0]-1])
                #predictions_RF = loaded_model.predict(Tfidf_vect.transform([fbpostdata]))
                processed_text11 =listofclasses[predictions_RF[0]-1]#getpersonalityprediction(fbpostdata)
                resultfacebookpersonality = {'Personality prediction from post ': processed_text11}
                
                #result1 = {'Job type suited for this ': processed_text1}
                print('resultfacebook',resultfacebookpersonality)
                
                
                
                
                return render_template('recommend.html', user=session['user'], data=data, job=processed_text1,linkedininfo=processed_text12,fbpersonality=processed_text11)
            else:
                print('Not able to scan data please provide pdf format')



            #return 'file uploaded successfully'
        return render_template('home.html', user=session['user'])
    return redirect(url_for('index'))

    







def LinkedInfetchingscrapinguserdata(links):
    global usrlnk
    global pwdlnk,linkedindata
  
    driver = webdriver.Chrome(ChromeDriverManager().install()) 
    driver.get('https://www.linkedin.com') 
    print ("Opened linkedin") 
    sleep(1) 

    username_box = driver.find_element_by_id('session_key') 
    username_box.send_keys(usrlnk) 
    print ("Email Id entered") 
    sleep(1) 
  
    password_box = driver.find_element_by_id('session_password') 
    password_box.send_keys(pwdlnk) 
    print ("Password entered") 
  
#login_box = driver.find_element_by_id('loginbutton') 
#login_box.click() 

    try:
            # clicking on login button
            driver.find_element_by_class_name("sign-in-form__submit-button").click()
    except NoSuchElementException:
            # Facebook new design
            driver.find_element_by_name("Sign in").click()
  
    print ("Done") 

    sleep(10) 
    #links='https://www.linkedin.com/in/dipti-mhatre-2a04b452/'
    driver.get(links)
    sleep(5) 

    try:
            # clicking on login button
            driver.find_element_by_class_name("pv-skills-section__chevron-icon").click()
    except NoSuchElementException:
            # Facebook new design
            pass



    html = driver.find_element_by_tag_name('html')  
    SCROLL_PAUSE_TIME = 1

# Get scroll height
    last_height = driver.execute_script("return document.body.scrollHeight")

    ik=0
    while ik<5:
        ik+=1
    
        html.send_keys(Keys.END)
        sleep(1)
    
    sleep(10)
    page_text = (driver.page_source).encode('utf-8')



#soup = BeautifulSoup(page_text)

#parse_data = soup.get_text()
#soup = BeautifulSoup(page_text, 'html.parser')
    
    with io.open("outputlinkedin.html", "w", encoding="utf-8") as f:
        f.write(str(page_text))
    

#print(soup)
#reviews_selector = soup.find_all('div', class_='_5msj')
#print(reviews_selector)
#row = soup.find('div._5msj') 
#print(row)


    person = {}



    print('data is ',person)
#sleep(5)     
#sleep(5) 
#input('Press anything to quit') 
    driver.quit() 
    print("Finished") 
    with open("outputlinkedin.html") as html_file:
        html = html_file.read()
    
## creating a BeautifulSoup object
    soup = BeautifulSoup(html, "html.parser")
#soup = BeautifulSoup(htmltxt, 'lxml')
    p_tags = soup.find_all("h3")  
    abbr_tags = soup.find_all("h4") 
    
    listofalltext=[]
    listofalldate=[]
    alldata=[]
#print(soup.find_all('p'))
    for tag in p_tags:
        listofalltext.append(tag.text)
        #print(tag.text)
    for tag in abbr_tags:
        listofalldate.append(tag.text)
    

    for i in range(len(listofalldate)):  
        linkedindata=linkedindata+listofalltext[i]+' '
        alldata.append(listofalltext[i]+"@"+listofalldate[i])
    #print(listofall[i]+"@"+listofalldate[i])

    print(alldata)
    return alldata

def FB_post_fetch(link):
    # Enter your Facebook email and password
    email = 'YOUR_EMAIL'
    password = 'YOUR_PASSWORD'
    # Instantiate an object
    fps = Fps(email, password, post_url_text='Full Story')
    # Example with single profile
    #single_profile = 'https://www.facebook.com/BillGates'
    data = fps.get_posts_from_profile(link)
    pp(data)
    return data

def fetchingscrapinguserdata(link):
    global usr
    global  pwd,fbpostdata
    driver = webdriver.Chrome(ChromeDriverManager().install()) 
    driver.get('https://m.facebook.com/') 
    print ("Opened facebook") 
    sleep(1) 

    username_box = driver.find_element_by_id('m_login_email') 
    username_box.send_keys(usr) 
    print ("Email Id entered") 
    sleep(1) 
  
    password_box = driver.find_element_by_id('m_login_password') 
    password_box.send_keys(pwd) 
    print ("Password entered") 
  
#login_box = driver.find_element_by_id('loginbutton') 
#login_box.click() 

    try:
            # clicking on login button
            driver.find_element_by_id("loginbutton").click()
    except NoSuchElementException:
            # Facebook new design
            driver.find_element_by_name("login").click()
  
    print ("Done") 

    sleep(15) 
    #link='https://m.facebook.com/ningeshkumar.kharatmol/'
    driver.get(link)
    sleep(5) 
    html = driver.find_element_by_tag_name('html')  
    SCROLL_PAUSE_TIME = 1

# Get scroll height
    last_height = driver.execute_script("return document.body.scrollHeight")

    ik=0
    while ik<25:
        ik+=1
    
        html.send_keys(Keys.END)
        sleep(1)
    
    sleep(10)
    page_text = (driver.page_source).encode('utf-8')
#soup = BeautifulSoup(page_text)

#parse_data = soup.get_text()
#soup = BeautifulSoup(page_text, 'html.parser')

    with io.open("output1.html", "w", encoding="utf-8") as f:
        f.write(str(page_text))
    
    
    

#print(soup)
#reviews_selector = soup.find_all('div', class_='_5msj')
#print(reviews_selector)
#row = soup.find('div._5msj') 
#print(row)


    person = {}



    print('data is ',person)
#sleep(5)     
#sleep(5) 
#input('Press anything to quit') 
    driver.quit() 
    print("Finished") 
    
    with open("output1.html") as html_file:
        html = html_file.read()
    
## creating a BeautifulSoup object
    soup = BeautifulSoup(html, "html.parser")
#soup = BeautifulSoup(htmltxt, 'lxml')
    p_tags = soup.find_all("p")  
    abbr_tags = soup.find_all("abbr") 
    listofalltext=[]
    listofalldate=[]
    alldata=[]
#print(soup.find_all('p'))
    for tag in p_tags:
        listofalltext.append(tag.text)
        #print(tag.text)
    for tag in abbr_tags:
        listofalldate.append(tag.text)
    

    for i in range(len(listofalltext)):  
        #alldata.append(listofalltext[i])
        fbpostdata=fbpostdata+listofalltext[i]+' '
        alldata.append(listofalltext[i]+"@"+listofalldate[i])
    #print(listofall[i]+"@"+listofalldate[i])

    print(alldata)
    return alldata







if __name__ == '__main__':
    #app.run(debug="True")
    app.run('0.0.0.0',threaded=False)