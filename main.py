import io
import threading
import time
from base64 import b64decode
import pandas as pd
import requests
import my_tf_mod
from flask import Flask, request, render_template, session
import firebase_admin
import random
from firebase_admin import credentials, firestore
cred = credentials.Certificate("key.json")
firebase_admin.initialize_app(cred)
import tensorflow as tf
import numpy as np
from io import BytesIO
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import base64
import os
from bs4 import BeautifulSoup

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.secret_key = "OnlineFruitQualityRecognition@123"
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

def imdecode_image(image_file):
    return cv2.imdecode(
        np.frombuffer(image_file.read(), np.uint8),
        cv2.IMREAD_UNCHANGED
    )

def fetch_calories(prediction):
    try:
        url = 'https://www.google.com/search?&q=calories in ' + prediction
        req = requests.get(url).text
        scrap = BeautifulSoup(req, 'html.parser')
        calories = scrap.find("div", class_="BNeawe iBp4i AP7Wnd").text
        return calories
    except Exception as e:
        print("Can't able to fetch the Calories")
        print(e)

@app.route('/usermakeprediction', methods=['POST','GET'])
def usermakeprediction():
    try:
        data={}
        fruit_dict=""
        rotten=""
        plot_url=""
        filename=""
        calorie=0.0
        if(request.method=="POST"):
            #ptype=request.form['ptype']
            #image = request.files['image']
            #image = imdecode_image(request.files["image"])
            #imagedata = read_file_as_image(image)
            #file_data = io.BytesIO(b64decode(image))
            #file = request.files['image']
            #print("File : ", file)
            #org_img, img = my_tf_mod.preprocess(file_data)
            #print("Img : ", img)
            #print("Image Shape : " ,img.shape)
            #fruit_dict = my_tf_mod.classify_fruit(img)
            #rotten = my_tf_mod.check_rotten(img)
            #img_x = BytesIO()
            #plt.imshow(org_img / 255.0)
            #plt.savefig(img_x, format='png')
            #plt.close()
            #img_x.seek(0)
            #plot_url = base64.b64encode(img_x.getvalue()).decode('utf8')
            file = request.files['image']
            """
            # Save image
            filename = secure_filename(file.filename)
            path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(path)

            prediction = predict.predict(path)
            print("Prediction : ", prediction)
            # Save image in static for website
            file.seek(0)
            path2 = os.path.join('./static/uploads', filename)
            file.save(path2)
            """
            org_img, img = my_tf_mod.preprocess(file)
            print(img.shape)
            fruit_dict = my_tf_mod.classify_fruit(img)
            rotten = my_tf_mod.check_rotten(img)

            print("Fruit Dict : ", fruit_dict)
            print("Rotten     : ", rotten)

            fruitname = None
            value1 = fruit_dict['apple']
            value2 = fruit_dict['banana']
            value3 = fruit_dict['orange']

            if(value1>value2 and value1>value3):
                fruitname='Apple'
            elif(value2>value3):
                fruitname='Banana'
            else:
                fruitname = 'Orange'

            cal = fetch_calories(fruitname)
            print("Cal : ", cal)
            calorie = (float(rotten[0]) * int(cal.split(" ")[0]))/100.0
            print("Calorie : ", calorie)
            img_x = BytesIO()
            plt.imshow(org_img / 255.0)
            plt.savefig(img_x, format='png')
            plt.close()
            img_x.seek(0)
            plot_url = base64.b64encode(img_x.getvalue()).decode('utf8')
            """
            filename = secure_filename(file.filename)
            path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(path)

            prediction = predict.predict(path)
            print("Prediction : ", prediction)

            img = io.imread(path)
            size = resize(img, (32, 32))
            ravel = size.ravel()
            ravel = size.ravel()
            ########### MNB Loaded Goods ######################
            pred_mnb = loaded_model_nb.predict([ravel])[0]
            print('pred_mnb*******************', pred_mnb)
            ########### End OF MNB Loaded Goods ###############
            ########### CNN Loaded Goods ######################
            vals = ['Pear', 'Tomato']
            cnn_size = resize(img, (32, 32, 32, 3))
            pred_cnn = load_model_cnn.predict([cnn_size])[0]
            print('pred_cnn*******************', pred_cnn)
            new_pred = np.argmax(pred_cnn)
            print('new_pred**************', new_pred)
            # new_pred_cnn = vals[new_pred] ## not working yet, getting a shape error
            final = pd.DataFrame({'name': np.array(vals), 'probability': pred_cnn[0]})
            final, new_pred_cnn = final.sort_values(by='probability', ascending=False), vals[new_pred]
            print('new_pred_cnn*****************', new_pred_cnn)
            print('final', final)
            ########### End Of CNN Loaded Goods ###############
            ########### Loaded Data Frame #####################
            nutri_facts_filename = 'data/nutri_facts_name.csv'
            df = get_nutri_facts(nutri_facts_filename)
            nf_mnb = df[df['Fruits_Vegetables_Name'] == pred_mnb]['Nutrition_Facts']
            print(nf_mnb)
            mnb_nf = nf_mnb.iloc[0]
            nf_cnn = df[df['Fruits_Vegetables_Name'] == new_pred_cnn]['Nutrition_Facts']
            cnn_nf = nf_cnn.iloc[0]
            print(cnn_nf)
            ########### End of Loaded Data Frame ##############

            # Save image in static for website
            #file.seek(0)
            #path2 = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            #file.save(path2)
        """
        print("Data : ", data)
        return render_template("usermakeprediction.html", fruit_dict=fruit_dict,
                               rotten=rotten, plot_url=plot_url, data=data,
                               path='uploads/'+filename,
                               filename=filename, cal=calorie)
    except Exception as e:
        return str(e)

@app.route('/')
def index():
    try:
        return render_template("index.html")
    except Exception as e:
        return str(e)

@app.route('/usermainpage')
def usermainpage():
    try:
        return render_template("usermainpage.html")
    except Exception as e:
        return str(e)

@app.route('/staffforgotpassword')
def staffforgotpassword():
    try:
        return render_template("staffforgotpassword.html")
    except Exception as e:
        return str(e)

@app.route('/staffenterotppage')
def staffenterotppage():
    try:
        return render_template("staffenterotppage.html")
    except Exception as e:
        return str(e)

@app.route('/staffchecking', methods=['POST'])
def staffchecking():
    try:
        if request.method == 'POST':
            uname = request.form['uname']
            email = request.form['email']
        print("Uname : ", uname, " Email : ", email);
        db = firestore.client()
        dbref = db.collection('newstaff')
        userdata = dbref.get()
        data = []
        for doc in userdata:
            print(doc.to_dict())
            print(f'{doc.id} => {doc.to_dict()}')
            data.append(doc.to_dict())
        flag = False
        for temp in data:
            if uname == temp['UserName'] and email == temp['EmailId']:
                session['username'] = uname
                session['emailid'] = email
                session['userid'] = temp['id']
                flag = True
                break
        if (flag):
            otp = random.randint(1000, 9999)
            print("OTP : ", otp)
            session['toemail'] = email
            session['uname'] = uname
            session['otp'] = otp
            print("User Id : ", session['userid'])
            return render_template("staffgenerateotp.html", uname=uname, toemail=email, otp=otp,
                                                        redirecturl= 'http://127.0.0.1:5000/staffenterotppage')
        else:
            return render_template("staffforgotpassword.html", msg="UserName/EmailId is Invalid")
    except Exception as e:
        return str(e)

@app.route('/staffcheckotppage', methods=['POST'])
def staffcheckotppage():
    if request.method == 'POST':
        storedotp=session['otp']
        enteredotp = request.form['otp']
        print("Entered OTP : ", enteredotp, " Stored OTP : ", storedotp)
        if(int(storedotp)==int(enteredotp)):
            return render_template("staffpasswordchangepage.html", msg="You can update your password")
        else:
            return render_template("staffenterotppage.html", msg="Incorrect OTP")
    return render_template("staffenterotppage.html", msg="Incorrect OTP")

@app.route('/staffpasswordchangepage', methods=['POST'])
def staffpasswordchangepage():
    print("Password Change Page")
    if request.method == 'POST':
        uname = request.form['uname']
        pwd = request.form['pwd']

        db = firestore.client()
        newstaff_ref = db.collection('newstaff')
        staffdata = newstaff_ref.get()
        data = []
        for doc in staffdata:
            print(doc.to_dict())
            print(f'{doc.id} => {doc.to_dict()}')
            data.append(doc.to_dict())
        id=""
        for doc in data:
            print("Document : ", doc)
            if(doc['UserName']==uname):
                id=doc['id']
        db = firestore.client()
        data_ref = db.collection(u'newstaff').document(id)
        data_ref.update({u'Password': pwd})
        print("Password Updated Success")
        return render_template("stafflogin.html", msg="Password Updated Success")
    return render_template("stafflogin.html", msg="Password Not Updated")

@app.route('/index')
def indexpage():
    try:
        return render_template("index.html")
    except Exception as e:
        return str(e)

@app.route('/logout')
def logoutpage():
    try:
        session['id']=None
        return render_template("index.html")
    except Exception as e:
        return str(e)

@app.route('/about')
def aboutpage():
    try:
        return render_template("about.html")
    except Exception as e:
        return str(e)

@app.route('/services')
def servicespage():
    try:
        return render_template("services.html")
    except Exception as e:
        return str(e)

@app.route('/gallery')
def gallerypage():
    try:
        return render_template("gallery.html")
    except Exception as e:
        return str(e)

@app.route('/adminlogin', methods=['GET','POST'])
def adminloginpage():
    msg=""
    if request.method == 'POST':
        uname = request.form['uname'].lower()
        pwd = request.form['pwd'].lower()
        print("Uname : ", uname, " Pwd : ", pwd)
        if uname == "admin" and pwd == "admin":
            return render_template("adminmainpage.html")
        else:
            msg = "UserName/Password is Invalid"
    return render_template("adminlogin.html", msg=msg)

@app.route('/userlogin', methods=['GET','POST'])
def userloginpage():
    msg=""
    if request.method == 'POST':
        uname = request.form['uname']
        pwd = request.form['pwd']

        db = firestore.client()
        dbref = db.collection('newuser')
        userdata = dbref.get()
        data = []
        for doc in userdata:
            print(doc.to_dict())
            print(f'{doc.id} => {doc.to_dict()}')
            data.append(doc.to_dict())
        flag = False
        for temp in data:
            if uname == temp['UserName'] and pwd == temp['Password']:
                session['userid'] = temp['id']
                flag = True
                break
        if (flag):
            return render_template("usermainpage.html")
        else:
            msg = "UserName/Password is Invalid"
    return render_template("userlogin.html", msg=msg)

@app.route('/stafflogin', methods=['GET','POST'])
def staffloginpage():
    msg=""
    if request.method == 'POST':
        uname = request.form['uname']
        pwd = request.form['pwd']
        db = firestore.client()
        dbref = db.collection('newstaff')
        userdata = dbref.get()
        data = []
        for doc in userdata:
            print(doc.to_dict())
            print(f'{doc.id} => {doc.to_dict()}')
            data.append(doc.to_dict())
        flag = False
        for temp in data:
            if uname == temp['UserName'] and pwd == temp['Password']:
                session['userid'] = temp['id']
                flag = True
                break
        if (flag):
            return render_template("staffmainpage.html")
        else:
            msg = "UserName/Password is Invalid"
    return render_template("stafflogin.html", msg=msg)

@app.route('/staffviewprofile')
def staffviewprofile():
    try:
        id = session['userid']
        db = firestore.client()
        dbref = db.collection('newstaff')
        userdata = dbref.get()
        data={}
        for doc in userdata:
            temp = doc.to_dict()
            if(id==temp['id']):
                data = {'id':temp['id'],
                    'FirstName':temp['FirstName'],
                    'LastName':temp['LastName'],
                    'EmailId':temp['EmailId'],
                    'UserName': temp['UserName'],
                    'PhoneNumber':temp['PhoneNumber']}
                break
        print("User Data ", data)
        return render_template("staffviewprofile.html", data=data)
    except Exception as e:
        return str(e)

@app.route('/userviewprofile')
def userviewprofile():
    try:
        id=session['userid']
        db = firestore.client()
        dbref = db.collection('newuser')
        userdata = dbref.get()
        data={}
        for doc in userdata:
            temp = doc.to_dict()
            if(id==temp['id']):
                data = {'id':temp['id'],
                    'FirstName':temp['FirstName'],
                    'LastName':temp['LastName'],
                    'EmailId':temp['EmailId'],
                    'PhoneNumber':temp['PhoneNumber'],
                    'UserName':temp['UserName']}
                break
        print("User Data ", data)
        return render_template("userviewprofile.html", data=data)
    except Exception as e:
        return str(e)

@app.route('/newuser', methods=['POST','GET'])
def newuser():
    try:
        msg=""
        print("Add New User page")
        if request.method == 'POST':
            fname = request.form['fname']
            lname = request.form['lname']
            uname = request.form['uname']
            pwd = request.form['pwd']
            email = request.form['email']
            phnum = request.form['phnum']
            address = request.form['address']
            id = str(random.randint(1000, 9999))
            json = {'id': id,
                        'FirstName': fname, 'LastName': lname,
                        'UserName': uname, 'Password': pwd,
                        'EmailId': email, 'PhoneNumber': phnum,
                        'Address': address}
            db = firestore.client()
            newuser_ref = db.collection('newuser')
            newuser_ref.document(id).set(json)
            print("User Inserted Success")
            msg = "New User Added Success"
        return render_template("newuser.html", msg=msg)
    except Exception as e:
        return str(e)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/adminaddstaff', methods=['POST','GET'])
def adminaddstaff():
    try:
        print("Add New Staff page")
        msg=""
        if request.method == 'POST':
            fname = request.form['fname']
            lname = request.form['lname']
            uname = request.form['uname']
            pwd = request.form['pwd']
            email = request.form['email']
            phnum = request.form['phnum']
            address = request.form['address']
            id = str(random.randint(1000, 9999))
            json = {'id': id,
                    'FirstName': fname,'LastName':lname,
                    'UserName': uname,'Password':pwd,
                    'EmailId': email,'PhoneNumber':phnum,
                    'Address': address}
            db = firestore.client()
            newdb_ref = db.collection('newstaff')
            id = json['id']
            newdb_ref.document(id).set(json)
            msg="New Staff Added Success"
        return render_template("adminaddstaff.html", msg=msg)
    except Exception as e:
        return str(e)

@app.route('/contact', methods=['POST','GET'])
def contactpage():
    try:
        msg=""
        if request.method == 'POST':
            cname = str(request.form['fname']) + " " + str(request.form['lname'])
            subject = request.form['subject']
            message = request.form['message']
            email = request.form['email']
            id = str(random.randint(1000, 9999))
            json = {'id': id,
                    'ContactName': cname, 'Subject': subject,
                    'Message': message,
                    'EmailId': email}
            db = firestore.client()
            newdb_ref = db.collection('newcontact')
            id = json['id']
            newdb_ref.document(id).set(json)
            msg = "New Contact Added Success"
        return render_template("contact.html", msg=msg)
    except Exception as e:
        return str(e)

@app.route('/adminviewusers')
def adminviewusers():
    try:
        db = firestore.client()
        newdata_ref = db.collection('newuser')
        newdata = newdata_ref.get()
        data=[]
        for doc in newdata:
            data.append(doc.to_dict())
        print("Users Data " , data)
        return render_template("adminviewusers.html", data=data)
    except Exception as e:
        return str(e)

@app.route('/adminviewinfos')
def adminviewinfos():
    try:
        db = firestore.client()
        newdata_ref = db.collection('newinfo')
        newdata = newdata_ref.get()
        data=[]
        for doc in newdata:
            data.append(doc.to_dict())
        print("Users Data " , data)
        return render_template("adminviewinfos.html", data=data)
    except Exception as e:
        return str(e)

@app.route('/staffviewusers')
def staffviewusers():
    try:
        db = firestore.client()
        newdata_ref = db.collection('newuser')
        newdata = newdata_ref.get()
        data=[]
        for doc in newdata:
            data.append(doc.to_dict())
        print("Users Data " , data)
        return render_template("staffviewusers.html", data=data)
    except Exception as e:
        return str(e)

@app.route('/staffviewinfos')
def staffviewinfos():
    try:
        db = firestore.client()
        newdata_ref = db.collection('newinfo')
        newdata = newdata_ref.get()
        data=[]
        for doc in newdata:
            data.append(doc.to_dict())
        print("Users Data " , data)
        return render_template("staffviewinfos.html", data=data)
    except Exception as e:
        return str(e)

@app.route('/adminviewstaffs')
def adminviewstaffs():
    try:
        db = firestore.client()
        newdata_ref = db.collection('newstaff')
        newdata = newdata_ref.get()
        data=[]
        for doc in newdata:
            data.append(doc.to_dict())
        print("Users Data " , data)
        return render_template("adminviewstaffs.html", data=data)
    except Exception as e:
        return str(e)

@app.route('/adminviewcontacts')
def adminviewcontacts():
    try:
        db = firestore.client()
        newdata_ref = db.collection('newcontact')
        newdata = newdata_ref.get()
        data=[]
        for doc in newdata:
            data.append(doc.to_dict())
        print("Contact Data " , data)
        return render_template("adminviewcontacts.html", data=data)
    except Exception as e:
        return str(e)

@app.route('/adminviewreports')
def adminviewreports():
    try:
        db = firestore.client()
        newdata_ref = db.collection('newquery')
        newdata = newdata_ref.get()
        data=[]
        for doc in newdata:
            data.append(doc.to_dict())
        print("Report Data " , data)
        return render_template("adminviewreports.html", data=data)
    except Exception as e:
        return str(e)

@app.route('/adminmainpage')
def adminmainpage():
    try:
        return render_template("adminmainpage.html")
    except Exception as e:
        return str(e)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    app.debug = True
    app.run()