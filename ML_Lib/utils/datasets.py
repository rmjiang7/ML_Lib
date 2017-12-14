import numpy as np
import os

class Dataset(object):

    description = ""

    def data_dir(self, fname):
        script_dir = os.path.dirname(__file__)
        rel_path = 'data/' + fname
        abs_file_path = os.path.join(script_dir, rel_path)
        return abs_file_path

class BostonHousing(Dataset):
    
    def __init__(self):
        self.load_data()
        self.set_description()
   
    def set_description(self):
        self.description = """
        The Boston Housing dataset is a regression task to predict the
        price of a house (y) in Boston given a set of features about
        the house (X).
        """

    def load_data(self):
        data = np.loadtxt(self.data_dir('boston_housing/data.txt'))
        self.n_observations = data.shape[0]
        self.n_features = data.shape[1] - 1
        self.X = data[:,:self.n_features]
        self.normalized_X = (self.X - self.X.mean(axis = 0))/(self.X.std(axis = 0))
        self.y = data[:,self.n_features]
        self.normalized_y = (self.y - self.y.mean(axis = 0))/(self.y.std(axis = 0))

class BreastCancer(Dataset):

    def __init__(self):
        self.load_data()

    def load_data(self):
        data = np.loadtxt(self.data_dir('breast_cancer/breast_cancer.csv'), delimiter = ',')
        self.n_observations = data.shape[0]
        self.n_features = data.shape[1] - 1
        self.X = data[:,:self.n_features]
        self.normalized_X = (self.X - self.X.mean(axis = 0))/(self.X.std(axis = 0))
        self.y = data[:,self.n_features]

if __name__ == "__main__":
    from ML_Lib.models.glm import LinearRegression
    from ML_Lib.models.neural_network import NeuralNetwork
    from ML_Lib.inference.sampling import HMC
    
    b = BostonHousing()

    l = NeuralNetwork([b.n_features, 10, 1])
    l.set_data(b.normalized_X, b.normalized_y)
    
    m = HMC(l)
    samples = m.train(1000, step_size = 0.0001, integration_steps = 50)
    predictions = l.predict(samples, b.normalized_X)
    print(np.mean((np.mean(predictions, axis = 0) - b.normalized_y)**2))

    
