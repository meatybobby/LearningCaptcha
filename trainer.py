from scipy.ndimage import convolve
from sklearn import datasets, linear_model, metrics
from sklearn.cross_validation import train_test_split
from sklearn.neural_network import BernoulliRBM, MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.externals import joblib
from sklearn.grid_search import GridSearchCV
import os
import numpy as np
import cv2

class CaptchaClassifier:
    def __init__(self):
        if os.path.exists('model/captcha.pkl'):
            self.classifier = joblib.load('model/captcha.pkl')
        else:
            self.classifier = MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=1000, alpha=1e-4,
                solver='sgd', verbose=10, tol=1e-4, random_state=1)




    def loadData(self):
        self.data = []
        self.target = []
        for directory in [f for f in os.listdir('templates') if not f.startswith('.')]:
                for png in [f for f in os.listdir('templates/' + directory) if not f.startswith('.')]:
                    ref = cv2.imread('templates/' + directory + '/' + png, cv2.IMREAD_GRAYSCALE)
                    self.data.append(ref)
                    self.target.append(directory)
        self.data = np.array(self.data).astype('float32')
        self.data = self.data.reshape(len(self.data),-1)
        self.target = np.array(self.target)
        self.data = (self.data - np.min(self.data, 0)) / (np.max(self.data, 0) + 0.0001)

    def trainData(self):
        self.loadData()
        self.data, self.target = self.nudge_dataset(self.data, self.target)
        self.classifier.fit(self.data, self.target)
        joblib.dump(self.classifier,'model/captcha.pkl')

    def produceCross(self):
        self.loadData()
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.data, self.target,
            test_size=0.2, random_state=0)
        self.X_train, self.Y_train = self.nudge_dataset(self.X_train, self.Y_train)
        self.X_test, self.Y_test = self.nudge_dataset(self.X_test, self.Y_test)

    def logTest(self):
        logistic_classifier = linear_model.LogisticRegression(C=100.0,verbose = True)
        logistic_classifier.fit(self.X_train,self.Y_train)
        predicted = logistic_classifier.predict(self.X_test)
        print("Classification report for classifier %s:\n%s\n"
            % (logistic_classifier, metrics.classification_report(self.Y_test, predicted)))
        print("Confusion matrix:\n%s"
            % metrics.confusion_matrix(self.Y_test, predicted))

    def RBMtest(self):
        logistic = linear_model.LogisticRegression(C = 10.0)
        rbm = BernoulliRBM(n_components = 100, n_iter = 80, learning_rate = 0.01, verbose = True)
        classifier = Pipeline(steps=[('rbm', rbm), ('logistic', logistic)])
        classifier.fit(self.X_train,self.Y_train)
        predicted = classifier.predict(self.X_test)
        print("Classification report for classifier %s:\n%s\n"
            % (classifier, metrics.classification_report(self.Y_test, predicted)))
        print("Confusion matrix:\n%s"
            % metrics.confusion_matrix(self.Y_test, predicted))

    def rbmGS(self):
        rbm = BernoulliRBM()
        logistic = linear_model.LogisticRegression()
        classifier = Pipeline([("rbm", rbm), ("logistic", logistic)])
        print ("SEARCHING RBM + LOGISTIC REGRESSION")
        params = {
            "rbm__learning_rate": [0.1, 0.01, 0.001],
            "rbm__n_iter": [20, 40, 80],
            "rbm__n_components": [50, 100, 200],
            "logistic__C": [1.0, 10.0, 100.0]}
        # perform a grid search over the parameter
        gs = GridSearchCV(classifier, params, n_jobs = -1, verbose = 1)
        gs.fit(self.data, self.target)
        print ("best score: %0.3f" % (gs.best_score_))
        print ("RBM + LOGISTIC REGRESSION PARAMETERS")
        bestParams = gs.best_estimator_.get_params()

        for p in sorted(params.keys()):
            print ("\t %s: %f" % (p, bestParams[p]))

    def mlpTest(self):
        mlp = MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=1000, alpha=1e-4,
            solver  ='sgd', verbose=10, tol=1e-4, random_state=1)
        mlp.fit(self.X_train,self.Y_train)
        predicted = mlp.predict(self.X_test)
        print("Classification report for classifier %s:\n%s\n"
            % (mlp, metrics.classification_report(self.Y_test, predicted)))
        print("Confusion matrix:\n%s"
            % metrics.confusion_matrix(self.Y_test, predicted))

    def test(self):
        self.loadData()
        expected = self.target
        predicted = self.classifier.predict(self.data)
        print("Classification report for classifier %s:\n%s\n"
            % (self.classifier, metrics.classification_report(expected, predicted)))
        print("Confusion matrix:\n%s"
            % metrics.confusion_matrix(expected, predicted))

    def decode(self, imageData):
        imageData = (imageData - np.min(imageData, 0)) / (np.max(imageData, 0) + 0.0001)
        predicted = self.classifier.predict(imageData)
        return ''.join(predicted)

    def nudge_dataset(self,X, Y):
        """
        This produces a dataset 5 times bigger than the original one,
        by moving the 8x8 images in X around by 1px to left, right, down, up
        """
        direction_vectors = [
            [[0, 1, 0],
            [0, 0, 0],
            [0, 0, 0]],

            [[0, 0, 0],
            [1, 0, 0],
            [0, 0, 0]],

            [[0, 0, 0],
            [0, 0, 1],
            [0, 0, 0]],

            [[0, 0, 0],
            [0, 0, 0],
            [0, 1, 0]]]

        shift = lambda x, w: convolve(x.reshape((20, 20)), mode='constant',
                                    weights=w).ravel()
        X = np.concatenate([X] +
                        [np.apply_along_axis(shift, 1, X, vector)
                            for vector in direction_vectors])
        Y = np.concatenate([Y for _ in range(5)], axis=0)
        return X, Y

if __name__ == '__main__':
    app = CaptchaClassifier()
