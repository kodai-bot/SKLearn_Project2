from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class SplitData:
    def __init__(self, features, label):
        self.features = features
        self.label = label
    
    def split(self, test_size=0.2, random_state=0):
        x_train, x_test, y_train, y_test = train_test_split(self.features, self.label, test_size=test_size, random_state=random_state)
        sc = StandardScaler()
        x_train = sc.fit_transform(x_train)
        x_test = sc.transform(x_test)
        return x_train, x_test, y_train, y_test