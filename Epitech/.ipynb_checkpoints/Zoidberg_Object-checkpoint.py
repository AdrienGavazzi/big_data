import os
import time
import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm, metrics
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB
from tqdm import tqdm


class ZOIDBERG():
    
    train_images = None
    train_labels = None
    test_images = None
    test_labels = None
    
    def load_data(self):
        """
            load_data() : cette méthode charge les images de la base de données d'entraînement et de test et les stocke dans les variables train_images,
            train_labels, test_images et test_labels. Les images sont redimensionnées en 64x64 pixels, converties en niveaux de gris et aplaties en vecteurs
            pour pouvoir être traitées par les modèles.
        """
        start_time = time.time()

        train_dir = "/jup/Epitech/Data/chest_Xray/train/"
        test_dir = "/jup/Epitech/Data/chest_Xray/test/"
        
        train_images = []
        train_labels = []
        for foldername in tqdm(["NORMAL", "PNEUMONIA"]): # os.listdir(train_dir)
            label = 0 if foldername == "NORMAL" else 1
            folderpath = os.path.join(train_dir, foldername)
            for filename in os.listdir(folderpath):
                if filename.endswith(".jpeg"):
                    imgpath = os.path.join(folderpath, filename)
                    img = cv2.imread(imgpath)
                    if img is None:
                        print('Wrong path:', imgpath)
                    else:
                        img = cv2.resize(img, (64, 64))
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        train_images.append(img.flatten())
                        train_labels.append(label)

        test_images = []
        test_labels = []
        for foldername in tqdm(["NORMAL", "PNEUMONIA"]): # os.listdir(test_dir)
            label = 0 if foldername == "NORMAL" else 1
            folderpath = os.path.join(test_dir, foldername)
            for filename in os.listdir(folderpath):
                if filename.endswith(".jpeg"):
                    imgpath = os.path.join(folderpath, filename)
                    img = cv2.imread(imgpath)
                    if img is None:
                        print('Wrong path:', imgpath)
                    else:
                        img = cv2.resize(img, (64, 64))
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        test_images.append(img.flatten())
                        test_labels.append(label)

        print("Finished in", round((time.time() - start_time), 1), "s")

        self.train_images = train_images
        self.train_labels = train_labels
        self.test_images = test_images
        self.test_labels = test_labels
        
        return train_images, train_labels, test_images, test_labels

    def KNN(self, n_neighbors = 5):
        """
            KNN(n_neighbors) : cette méthode entraîne un classificateur KNN (k plus proches voisins) avec le nombre de voisins spécifié (5 par défaut) et
            calcule sa précision sur les données de test.
        """
        start_time = time.time()

        knn = KNeighborsClassifier(n_neighbors=n_neighbors)
        knn.fit(self.train_images, self.train_labels)

        test_preds = knn.predict(self.test_images)

        accuracy = np.mean(test_preds == self.test_labels)
        print("Exactitude du modèle : {:.2f} %".format(accuracy*100))
        print("Finished in", round((time.time() - start_time), 1), "s")

    def SVC(self, linear: bool = False):
        """
            SVC(linear) : cette méthode entraîne un classificateur SVM (Support Vector Machine) avec un noyau gaussien par défaut, ou un classificateur
            linéaire SVM si l'argument linear est True. Elle calcule ensuite la précision du modèle sur les données de test.
        """
        start_time = time.time()

        if not linear:

            clf = svm.SVC(verbose=True)
            clf.fit(self.train_images, self.train_labels)
            
            predicted = clf.predict(self.test_images)

        else:

            clf = svm.LinearSVC(verbose=True)
            clf.fit(self.train_images, self.train_labels)
            
            predicted = clf.predict(self.test_images)

        print("Accuracy:", round(metrics.accuracy_score(self.test_labels, predicted)*100, 2), "%")
        print("Finished in", round((time.time() - start_time), 1), "s")

    def MLP_classifier(self):
        """
            MLP_classifier() : cette méthode entraîne un classificateur MLP (Multilayer Perceptron) avec une architecture de 784 neurones en entrée, 3
            neurones dans une couche cachée et une fonction d'activation de type sigmoïde. Elle calcule ensuite la précision du modèle sur les données de 
            test.
        """
        start_time = time.time()

        clf = MLPClassifier(verbose=True, solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(784, 3), random_state=1)
        clf.fit(self.train_images, self.train_labels)
        
        predicted = clf.predict(self.test_images)

        print("Accuracy:", round(metrics.accuracy_score(self.test_labels, predicted)*100, 2), "%")
        print("Finished in", round((time.time() - start_time), 1), "s")

    def NAIVE_bayes(self):
        """
            NAIVE_bayes() : cette méthode entraîne un classificateur Bayésien naïf gaussien et calcule ensuite sa précision sur les données de test.
        """
        start_time = time.time()

        model = GaussianNB()
        model.fit(self.train_images, self.train_labels)

        predicted = model.predict(self.test_images)

        print("Accuracy:", round(metrics.accuracy_score(self.test_labels, predicted)*100, 2), "%")
        print("Finished in", round((time.time() - start_time), 1), "s")

    def EXTREMELY_randomized_trees(self, estimators: int = 100, max_depth: int = 10):
        """
            EXTREMELY_randomized_trees(estimators, max_depth) : cette méthode entraîne un classificateur de forêt d'arbres extrêmement aléatoires avec le 
            nombre d'estimateurs et la profondeur maximale spécifiés (100 et 10 par défaut). Elle calcule ensuite la précision du modèle sur les données de 
            test.
        """
        start_time = time.time()

        clf = ExtraTreesClassifier(n_estimators=estimators, max_depth=max_depth, min_samples_split=2, random_state=0)
        clf = clf.fit(self.train_images, self.train_labels)

        predicted = clf.predict(self.test_images)

        print("Accuracy:", round(metrics.accuracy_score(self.test_labels, predicted)*100, 2), "%")
        print("Finished in", round((time.time() - start_time), 1), "s")
    
    def compare(self):
        print("\n=== Loading Data ===")
        self.load_data()
        print("\n=== KNN Model ===")
        self.KNN()
        print("\n=== SVC Model ===")
        self.SVC()
        print("\n=== MLP Classifier Model ===")
        self.MLP_classifier()
        print("\n=== NAIVE bayes Model ===")
        self.NAIVE_bayes()
        print("\n=== EXTREMELY randomized trees Model ===")
        self.EXTREMELY_randomized_trees()
        