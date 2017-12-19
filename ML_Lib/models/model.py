class Model(object):

    def set_data(self, *args):
        raise NotImplementedException("Must be implemented in subclass!")

    def get_params(self):
        raise NotImplementedException("Must be implemented in subclass!")

    def set_params(self, params):
        raise NotImplementedException("Must be implemented in subclass!")

    def predict(self, params, input):
        raise NotImplementedException("Must be implemented in subclass!")

class ProbabilityModel(Model):

    def full_log_prob(self, params, *args):
        raise NotImplementedException("Must be implemented in subclass!")

    def full_grad_log_prob(self, params, *args):
        raise NotImplementedException("Must be implemented in subclass!")
    
    def log_prob(self, params):
        raise NotImplementedException("Must be implemented in subclass!")

    def grad_log_prob(self, params):
        raise NotImplementedException("Must be implemented in subclass!")
