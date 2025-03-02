from django.shortcuts import render
from django.template import RequestContext
from django.contrib import messages
from django.http import HttpResponse
import os
import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from numpy import dot
from numpy.linalg import norm
from django.core.files.storage import FileSystemStorage
from datetime import date
from string import punctuation
from nltk.corpus import stopwords
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer #loading tfidf vector
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import io
import base64

global uname, tfidf_vectorizer, scaler
global X_train, X_test, y_train, y_test
accuracy, precision, recall, fscore = [], [], [], []

#define object to remove stop words and other text processing
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
ps = PorterStemmer()

#define function to clean text by removing stop words and other special symbols
def cleanText(doc):
    tokens = doc.split()
    #table = str.maketrans('', '', punctuation)
    #tokens = [w.translate(table) for w in tokens]
    #tokens = [word for word in tokens if word.isalpha()]
    tokens = [w for w in tokens if not w in stop_words]
    tokens = [word for word in tokens if len(word) > 1]
    tokens = [ps.stem(token) for token in tokens]
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    tokens = ' '.join(tokens)
    return tokens

X = []
Y = []
labels = []
filenames = []

dataset = pd.read_csv("input-1300.csv")
labels, count = np.unique(dataset['Pattern'], return_counts=True)
dataset = dataset.values

def getLabel(name):
    label = -1
    for i in range(len(labels)):
        if labels[i] == name:
            label = i
            break
    return label

if os.path.exists("model/X.npy"):
    X = np.load("model/X.npy")
    Y = np.load("model/Y.npy")
    filenames = np.load("model/file.npy")
else:
    for i in range(len(dataset)):
        name = dataset[i,1]
        label = getLabel(dataset[i,2])
        if os.path.exists("DesignPatternsCode/"+name+".java") == True:
            with open("DesignPatternsCode/"+name+".java", "r") as file:
                data = file.read()
            file.close()
            data = data.strip("\n").strip().lower()
            data = cleanText(data)#clean description
            X.append(data)
            Y.append(label)
            filenames.append(name+".java")
            print(str(i)+" "+str(label))
    X = np.asarray(X)
    Y = np.asarray(Y)
    filenames = np.asarray(filenames)
    np.save("model/X", X)
    np.save("model/Y", Y)
    np.save("model/file", filenames)
tfidf_vectorizer = TfidfVectorizer(stop_words=stop_words, use_idf=True, smooth_idf=False, norm=None, decode_error='replace', max_features=2000)
X = tfidf_vectorizer.fit_transform(X).toarray()

Y = np.asarray(Y)
sc = StandardScaler()
X = sc.fit_transform(X)

indices = np.arange(X.shape[0])
np.random.shuffle(indices)
X = X[indices]
Y = Y[indices]
filenames = filenames[indices]

print(np.unique(Y, return_counts=True))

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2) #split dataset into train and test
X_train, X_test1, y_train, y_test1 = train_test_split(X, Y, test_size=0.1) #split dataset into train and test
def calculateMetrics(algorithm, predict, y_test):
    global accuracy, precision, recall, fscore
    a = accuracy_score(y_test,predict)*100
    p = precision_score(y_test, predict,average='macro') * 100
    r = recall_score(y_test, predict,average='macro') * 100
    f = f1_score(y_test, predict,average='macro') * 100
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)

dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
predict = dt.predict(X_test)
calculateMetrics("Decision Tree", predict, y_test)

rf = RandomForestClassifier()
rf.fit(X_train, y_train)
predict = rf.predict(X_test)
calculateMetrics("Random Forest", predict, y_test)

svm_cls = svm.SVC()
svm_cls.fit(X_train, y_train)
predict = svm_cls.predict(X_test)
calculateMetrics("SVM", predict, y_test)

def LoadDataset(request):
    if request.method == 'GET':
        dataset = pd.read_csv("input-1300.csv")
        dataset = dataset.values
        output = ''
        output+='<table border=1 align=center width=100%><tr><th><font size="" color="black">Project Name</th><th><font size="" color="black">Class Name</th>'
        output+='<th><font size="" color="black">Pattern Name</th></tr>'
        for i in range(len(dataset)):
            pname = dataset[i,0]
            cname = dataset[i,1]
            pattern = dataset[i,2]
            output+='<td><font size="" color="black">'+pname+'</td><td><font size="" color="black">'+cname+'</td><td><font size="" color="black">'+pattern+'</td></tr>'
        output+= "</table></br></br></br></br>"  
        context= {'data':output}
        return render(request, 'UserScreen.html', context)

def Vector(request):
    if request.method == 'GET':
        global X
        context= {'data':'<font size="" color="black">'+str(X)+'</font>'}
        return render(request, 'UserScreen.html', context)    

def TrainML(request):
    if request.method == 'GET':
        output = ''
        output+='<table border=1 align=center width=100%><tr><th><font size="" color="black">Algorithm Name</th><th><font size="" color="black">Accuracy</th><th><font size="" color="black">Precision</th>'
        output+='<th><font size="" color="black">Recall</th><th><font size="" color="black">FSCORE</th></tr>'
        global accuracy, precision, recall, fscore
        algorithms = ['Decision Tree', 'Random Forest', 'SVM']
        for i in range(len(algorithms)):
            output+='<td><font size="" color="black">'+algorithms[i]+'</td><td><font size="" color="black">'+str(accuracy[i])+'</td><td><font size="" color="black">'+str(precision[i])+'</td><td><font size="" color="black">'+str(recall[i])+'</td><td><font size="" color="black">'+str(fscore[i])+'</td></tr>'
        output+= "</table></br>"        
        df = pd.DataFrame([['Decision Tree','Precision',precision[0]],['Decision Tree','Recall',recall[0]],['Decision Tree','F1 Score',fscore[0]],['Decision Tree','Accuracy',accuracy[0]],
                           ['Random Forest','Precision',precision[1]],['Random Forest','Recall',recall[1]],['Random Forest','F1 Score',fscore[1]],['Random Forest','Accuracy',accuracy[1]],
                           ['SVM','Precision',precision[2]],['SVM','Recall',recall[2]],['SVM','F1 Score',fscore[2]],['SVM','Accuracy',accuracy[2]],
                          ],columns=['Algorithms','Metrics','Value'])
        df.pivot_table(index="Algorithms", columns="Metrics", values="Value").plot(kind='bar', figsize=(5, 3))
        plt.title("All Algorithms Performance Graph")
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        img_b64 = base64.b64encode(buf.getvalue()).decode()    
        context= {'data':output, 'img': img_b64}
        return render(request, 'ViewResult.html', context)  

def Predict(request):
    if request.method == 'GET':
       return render(request, 'Predict.html', {})     

def PredictAction(request):
    if request.method == 'POST':
        global rf, tfidf_vectorizer, X, sc, filenames, labels
        myfile = request.FILES['t1'].read()
        fname = request.FILES['t1'].name
        if os.path.exists("PatternApp/static/"+fname):
            os.remove("PatternApp/static/"+fname)
        with open("PatternApp/static/"+fname, "wb") as file:
            file.write(myfile)
        file.close()    
        with open("PatternApp/static/"+fname, "r") as file:
            content = file.read()
        file.close()
        data = content.strip("\n").strip().lower()
        data = cleanText(data)#clean description
        temp = tfidf_vectorizer.transform([data]).toarray()
        temp = sc.transform(temp)
        predict = rf.predict(temp)[0]
        predict = int(predict)
        print(str(predict)+" "+labels[predict])
        context= {'data':'<font size="4" color="blue">Uploaded Code Pattern Predicted As : '+labels[predict]+'</font>'}
        return render(request, 'Predict.html', context)

def index(request):
    if request.method == 'GET':
       return render(request, 'index.html', {})

def Login(request):
    if request.method == 'GET':
       return render(request, 'Login.html', {})
    
def UserLogin(request):
    if request.method == 'POST':
        global userid
        user = request.POST.get('t1', False)
        password = request.POST.get('t2', False)
        if user == "admin" and password == "admin":
            context= {'data':'Welcome '+user}
            return render(request, 'UserScreen.html', context)
        else:
            context= {'data':'Invalid Login'}
            return render(request, 'Login.html', context)

