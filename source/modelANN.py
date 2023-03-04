import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

from data_preprocess import DataPreprocessing

class ModelANN(DataPreprocessing):
    def __init__(self, *args, **kwargs):
        super(ModelANN, self).__init__(*args, **kwargs)

    def dt(self, X_train, X_test, y_train, y_test):
        #Create DT model
        MANN_classifier = MLPClassifier()

        #Train the model
        MANN_classifier.fit(X_train, y_train)

        #Test the model
        DT_predicted = MANN_classifier.predict(X_test)

        #get performance
        self.accuracy = accuracy_score(y_test, DT_predicted)

        return MANN_classifier