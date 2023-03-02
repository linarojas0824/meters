import sys
sys.path.append('/Users/linarojas/Desktop/GitHub/meters/source')
from source.data_preprocess import DataPreprocessing
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

class ModelANN(DataPreprocessing):
    def __init__(self, *args, **kwargs):
        super(ModelANN, self).__init__(*args, **kwargs)

    def dt(self, X_train, X_test, y_train, y_test):
        #Create DT model
        MANN_classifier = MLPClassifier(hidden_layer_sizes=(100,),
                                      activation='logistic',solver='lbfgs',batch_size='auto')

        #Train the model
        MANN_classifier.fit(X_train, y_train)

        #Test the model
        DT_predicted = MANN_classifier.predict(X_test)

        error = 0
        for i in range(len(y_test)):
            error += np.sum(DT_predicted != y_test)

        total_accuracy = 1 - error / len(y_test)

        #get performance
        self.accuracy = accuracy_score(y_test, DT_predicted)

        return MANN_classifier