

import os
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.model_selection import cross_val_score
import joblib
import matplotlib.pyplot as plt


def buildSampleFromPath (path1, path2):
    images =[]
    for element in os.listdir(path1):
        name=f"{path1}/{element}"
        initial_image = Image.open(name)
        resize=resized_image(initial_image,200 ,100 )
        image = {
            "name_path" : name,
            "resized_image" : resize,
            "X_histo" : computeHisto(resize),
            "y_true_class" : +1,
            "y_predicted_class" : None
        }
        images.append(image)
    for element in os.listdir(path2):
        name=f"{path2}/{element}"
        initial_image = Image.open(name)
        resize=resized_image(initial_image,200 ,100 )
        image = {
            "name_path" : name,
            "resized_image" : resize,
            "X_histo" : computeHisto(resize),
            "y_true_class" : -1,
            "y_predicted_class" : None
        }
        images.append(image)
    return images


def resized_image(i, l, h):
    i=i.convert("RGB")
    return i.resize((l,h))

def computeHisto(i):
    image = np.array(i.convert("RGB"))
    #histo = i.convert("RGB").histogram() #double conversion en RGB
    red, _ = np.histogram((image[:,:,0]).flatten(), bins=256, range=[0, 256])
    green, _ = np.histogram((image[:,:,1]).flatten(), bins=256, range=[0, 256])
    blue, _ = np.histogram((image[:,:,2]).flatten(), bins=256, range=[0, 256])

    histo=np.concatenate([red, green, blue]).astype(float)


    return histo 

algo_bayes = {
    "name" : GaussianNB,
    "hyper_param": {}
}
algo_SVC={
    "name" : SVC,
    "hyper_param" : {"C":1.0, "kernel":'poly', "degree":3, "gamma":'scale'}
}

def fitFromHisto(S, algo):
    X = S.data
    y = S.target
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30,shuffle=True, random_state=42)#ajout de random_state=42 car sinon taux de reussite varie en fonction des images choisies pas la fonction et non pas parles hyper paramètres
    classifieur = algo["name"](**algo["hyper_param"])
    #score = cross_val_score(classifieur, X, y, cv=5)
    #print("erreur par cross validation : ", 1-np.mean(score))
    classifieur.fit(X, y) #avec fit le model apprend
    return classifieur

def givePredict(model, image):
    X=np.array(image["X_histo"]).reshape(1,-1)
    return model.predict(X)[0]#le [0] permet d'avoir une plus belle syntaxe dans notre fichier texte, la prediction ne sera pas entre crochet mais on récuperera uniquement la valeur
        
def predictFromHisto(S, model):
    predictions=[]
    for image in S:
        y_predits = givePredict(model, image)#le [0] permet d'avoir une plus belle syntaxe dans notre fichier texte, la prediction ne sera pas entre crochet mais on récuperera uniquement la valeur
        image["y_predicted_class"]=y_predits
        predictions.append(y_predits)
    return predictions



def erreurEmpirique(S, model):
    erreur = 0
    for image in S:
        X=np.array(image["X_histo"]).reshape(1,-1)
        y_predits = model.predict(X) 
        if y_predits[0]!=image["y_true_class"]: # ajout de [0]
            erreur+=1
    err = erreur/len(S)
    print ("Erreur empirique: ", err)
    return err

def erreurReelle(set, model):
    X = set.data
    y = set.target

    scores = cross_val_score(model, X, y, cv=5)
    erreur = 1-np.mean(scores)
    print("Erreur réelle :", erreur)
    return erreur

class CustomImageDataSet:
    data = []
    target = []

    def __init__(self, path1, path2):
        s = buildSampleFromPath(path1, path2)
        X = []
        y = []
        for image in s:
            X.append(image["X_histo"])
            y.append(image["y_true_class"])
        self.data = np.array(X)
        self.target = np.array(y)


path1 = "Init/Mer"
path2= "Init/Ailleurs"
s = buildSampleFromPath(path1, path2)
set = CustomImageDataSet(path1, path2)
model = fitFromHisto(set, algo_SVC)
joblib.dump(model, "main.pkl")
print(givePredict(model,s[0]))
ee=erreurEmpirique(s, model)
er=erreurReelle(set, model)

