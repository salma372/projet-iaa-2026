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

from main import *




def buildSampleFromPathNouveau (path) :
    img = []
    for element in os.listdir(path) :
        name = f"{path}/{element}"
        initial_image = Image.open(name)
        resizd_image = resized_image(initial_image,200,100)
        image = {"name_path" : name,
        "resized_image" : resizd_image,
        "X_histo" : computeHisto(resizd_image),
        "y_true_class" : None,
        "y_predicted_class" : None}
        img.append(image)
    

    return img

def ecrire_fichier():
    fichier = open("Célia&Salma&Eliahou.txt", "w")
    fichier.write("#Auteurs : Célia, Eliahou, Salma \n")
    fichier.write("#Nom de l’algorithme d’apprentissage utilisé : SVC \n")
    fichier.write("#Valeurs des hyperparamètres : C=1.0, kernel='rbf', degree=3, gamma='scale' \n")
    fichier.write("#Résumé des descripteurs d'image considérés : Histogramms de couleurs \n")
    for image in s_nouveau:
        name__path = image["name_path"]
        split_name = name__path.split("/")
        name = split_name[-1]
        fichier.write(name + " "+str(image["y_predicted_class"]) + "\n")
    fichier.write("#EE = " + str(ee) + "\n")
    fichier.write("#ER = " + str(er) + "\n")
    fichier.close()


model = joblib.load("main.pkl")
path = "dataCC2Test"
s_nouveau = buildSampleFromPathNouveau(path)
predictFromHisto(s_nouveau, model)
ecrire_fichier()
print("marche")
