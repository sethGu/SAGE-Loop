
from sklearn.svm import SVC

class myclassifier:
    def __init__(self):
        self.model = SVC(probability=True)
    
    def fit(self, train_aug_x, train_aug_y):
        self.model.fit(train_aug_x, train_aug_y)
    
    def predict(self, test_aug_x):
        return self.model.predict(test_aug_x)
    
    def predict_proba(self, test_aug_x):
        return self.model.predict_proba(test_aug_x)[:, 1]