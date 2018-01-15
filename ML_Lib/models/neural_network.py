import numpy as np
import autograd.numpy as agnp
import autograd.scipy as agsp
import autograd
from autograd.scipy.misc import logsumexp
from autograd.scipy.signal import convolve
from ML_Lib.models.model import Model

def sigmoid(x):
    return 0.5*(agnp.tanh(x) + 1.0)   # Output ranges from 0 to 1.

def relu(x):
    return x * (x > 0)

def linear(x):
    return x

class Layer(object):

    def __init__(self, input_dim, output_dim):
        self.num_weights = 0
        self.m = input_dim
        self.n = output_dim

    def forward(self, weights, inputs):
        raise NotImplementedException("Must define forward pass through the layer!")

    def get_params(self):
        raise NotImplementedException("Must define method to get params")

    def set_params(self, params):
        raise NotImplementedException("Must define method to set params")

class FCLayer(Layer):

    def __init__(self, input_dim, output_dim, nonlinearity = linear):
        super().__init__(input_dim, output_dim)
        self.nonlinearity = nonlinearity
        self.num_weights = (self.m+1)*self.n
        
        # Xavier Initialization
        self.params = np.random.normal(0, np.sqrt(2/(self.m + self.n)), size = (1,self.num_weights))
        
    def unpack_params(self,weights):
        num_weight_sets = len(weights)
        return weights[:, :self.m*self.n].reshape((num_weight_sets, self.m, self.n)),\
               weights[:, self.m*self.n:].reshape((num_weight_sets, 1, self.n))

    def forward(self, weights, inputs):
        if len(inputs.shape) > 3:
            inputs = inputs.reshape((inputs.shape[0],inputs.shape[1],-1))
        W, b = self.unpack_params(weights)
        return self.nonlinearity(agnp.einsum('mnd,mdo->mno', inputs, W) + b)

    def get_params(self):
        return self.params

    def set_params(self, params):
        self.params = params

class ConvLayer(Layer):

    def __init__(self, input_dims, kernel_shape, num_filters, nonlinearity = linear):
        depth = input_dims[0]
        y = input_dims[1]
        x = input_dims[2]
        
        self.kernel_shape = kernel_shape
        self.num_filters = num_filters
        self.num_filter_weights = depth * num_filters * kernel_shape[0] * kernel_shape[1]
        self.filter_weights_shape = (depth, self.num_filters, kernel_shape[0], kernel_shape[1])
        self.bias_shape = (1, num_filters, 1, 1)
        self.nonlinearity = nonlinearity
            
        self.output_dim = (self.num_filters,) + self.conv_output_shape(input_dims[1:], self.kernel_shape)
         
        super().__init__(np.prod(input_dims), np.prod(self.output_dim))
        self.num_weights = self.num_filter_weights + num_filters 
        # Xavier Initialization
        self.params = np.random.normal(0, np.sqrt(2/(self.m + self.n)), size = (1,self.num_weights))

    def conv_output_shape(self, A, B):
        return (A[0] - B[0] + 1, A[1] - B[1] + 1)

    def get_output_shape(self):
        return self.output_dim

    def unpack_params(self, weights):
        num_weight_sets = len(weights)
        return weights[:, :self.num_filter_weights].reshape((num_weight_sets,) + self.filter_weights_shape),\
               weights[:,self.num_filter_weights:].reshape((num_weight_sets,) + self.bias_shape)

    def forward(self, weights, inputs):
        # Input dims are [num_weight_sets, 
        w,b = self.unpack_params(weights)
        convs = []
        for i in range(len(w)):
            conv = convolve(inputs[i,:], w[i,:], axes=([2,3],[2,3]), dot_axes = ([1], [0]), mode = 'valid')
            conv = conv + b[i,:]
            convs.append(self.nonlinearity(conv))
        z = agnp.array(convs)
        return z

    def get_params(self):
        return self.params

    def set_params(self, params):
        self.params = params

class RNNLayer(Layer):

    def __init__(self, input_dim, hidden_dim, output_dim, hidden_nonlinearity = linear, output_nonlinearity = linear):
        super().__init__(input_dim, output_dim)
        self.hidden_dim = hidden_dim
        self.num_weights = (self.m + self.hidden_dim + 1) * self.hidden_dim + (self.hidden_dim + 1) * self.n
        self.hidden_nonlinearity = hidden_nonlinearity
        self.output_nonlinearity = output_nonlinearity

        self.hidden_init = agnp.random.randn(self.hidden_dim)
        self.params = np.random.normal(0, np.sqrt(2/(self.m + self.n)), size = (1,self.num_weights))

    def unpack_hidden_params(self, weights):
        W_hidden = weights[:,:(self.m + self.hidden_dim) * self.hidden_dim]
        b_hidden = weights[:,(self.m + self.hidden_dim) * self.hidden_dim: (self.m + self.hidden_dim + 1) * self.hidden_dim]
        return W_hidden.reshape((-1,self.m + self.hidden_dim, self.hidden_dim)), b_hidden.reshape((-1, 1, self.hidden_dim))

    def unpack_output_params(self, weights):
        W_output = weights[:,(self.m + self.hidden_dim + 1) * self.hidden_dim: (self.m + self.hidden_dim + 1) * self.hidden_dim + self.hidden_dim * self.n]
        b_output = weights[:,(self.m + self.hidden_dim + 1) * self.hidden_dim + self.hidden_dim * self.n:]
        return W_output.reshape((-1,self.hidden_dim, self.n)), b_output.reshape((-1, 1, self.n))
    
    def update_hidden(self, weights, input, hidden):
        concated_input = agnp.concatenate((input, hidden),axis = 2)
        W_hidden, b_hidden = self.unpack_hidden_params(weights)
        return self.hidden_nonlinearity(agnp.einsum('pdh,pnd->pnh', W_hidden, concated_input) + b_hidden)

    def get_output(self, weights, hidden):
        W_output, b_output = self.unpack_output_params(weights)
        return self.output_nonlinearity(agnp.einsum('pdh,pnd->pnh', W_output, hidden) + b_output)

    def forward(self, weights, inputs):
        n_param_sets = inputs.shape[0]
        sequence_length = inputs.shape[1]
        n_sequences = inputs.shape[2]
        
        hiddens = agnp.expand_dims(agnp.expand_dims(self.hidden_init, 0).repeat(n_sequences, 0),0).repeat(n_param_sets, 0)
        outputs = [self.get_output(weights, hiddens)]
        for idx in range(sequence_length):
            input = inputs[:,idx,:,:]
            hiddens = self.update_hidden(weights, input, hiddens)
            outputs.append(self.get_output(weights, hiddens))
        
        out = agnp.array(outputs).reshape((inputs.shape[0],inputs.shape[1] + 1, inputs.shape[2], inputs.shape[3]))
        return out

    def get_params(self):
        return self.params

    def set_params(self, params):
        self.params = params

class LSTMLayer(Layer):

    def __init__(self, input_dim, hidden_dim, output_dim, hidden_nonlinearity = linear, output_nonlinearity = linear):
        super().__init__(input_dim, output_dim)
        self.hidden_dim = hidden_dim
        self.num_h_weights = (self.m + self.hidden_dim) * self.hidden_dim
        self.num_b_weights = (self.m + self.hidden_dim + 1) * self.hidden_dim
        self.num_weights = self.num_b_weights * 4 + (self.hidden_dim + 1) * self.n
        self.hidden_nonlinearity = hidden_nonlinearity
        self.output_nonlinearity = output_nonlinearity

        self.hidden_init = agnp.random.randn(self.hidden_dim) * 0.01
        self.cell_init = agnp.random.randn(self.hidden_dim) * 0.01
        self.params = agnp.random.randn(1, self.num_weights)

    def unpack_change_params(self, weights):
        W_change = weights[:,:self.num_h_weights]
        b_change = weights[:,self.num_h_weights:self.num_b_weights]
        return W_change.reshape((-1,self.m + self.hidden_dim, self.hidden_dim)), b_change.reshape((-1, 1, self.hidden_dim))

    def unpack_forget_params(self, weights):
        W_forget = weights[:,self.num_b_weights:self.num_b_weights + self.num_h_weights]
        b_forget = weights[:,self.num_b_weights + self.num_h_weights:self.num_b_weights*2]
        return W_forget.reshape((-1,self.m + self.hidden_dim, self.hidden_dim)), b_forget.reshape((-1, 1, self.hidden_dim))

    def unpack_ingate_params(self, weights):
        W_ingate = weights[:,self.num_b_weights*2:self.num_b_weights*2 + self.num_h_weights]
        b_ingate = weights[:,self.num_b_weights*2 + self.num_h_weights: self.num_b_weights*3]
        return W_ingate.reshape((-1,self.m + self.hidden_dim, self.hidden_dim)), b_ingate.reshape((-1, 1, self.hidden_dim))
    
    def unpack_outgate_params(self, weights):
        W_outgate = weights[:,self.num_b_weights*3:self.num_b_weights*3 + self.num_h_weights]
        b_outgate = weights[:,self.num_b_weights*3 + self.num_h_weights: self.num_b_weights*4]
        return W_outgate.reshape((-1,self.m + self.hidden_dim, self.hidden_dim)), b_outgate.reshape((-1, 1, self.hidden_dim))

    def unpack_output_params(self, weights):
        W_output = weights[:,self.num_b_weights*4:self.num_b_weights*4 + self.hidden_dim * self.n]
        b_output = weights[:,self.num_b_weights*4 + self.hidden_dim * self.n:]
        return W_output.reshape((-1,self.hidden_dim, self.n)), b_output.reshape((-1, 1, self.n))

    def update_hidden(self, weights, input, hidden, cells):
        concated_input = agnp.concatenate((input, hidden),axis = 2)
        W_change, b_change = self.unpack_change_params(weights)
        change = agnp.tanh(agnp.einsum('pdh,pnd->pnh', W_change, concated_input) + b_change)
        W_forget, b_forget = self.unpack_forget_params(weights)
        forget = self.hidden_nonlinearity(agnp.einsum('pdh,pnd->pnh', W_forget, concated_input) + b_forget)
        W_ingate, b_ingate = self.unpack_ingate_params(weights)
        ingate = self.hidden_nonlinearity(agnp.einsum('pdh,pnd->pnh', W_ingate, concated_input) + b_ingate)
        W_outgate, b_outgate = self.unpack_outgate_params(weights)
        outgate = self.hidden_nonlinearity(agnp.einsum('pdh,pnd->pnh', W_outgate, concated_input) + b_outgate)
        cells = cells * forget + ingate * change
        hidden = outgate * agnp.tanh(cells)
        return hidden, cells

    def get_output(self, weights, hidden):
        W_output, b_output = self.unpack_output_params(weights)
        return self.output_nonlinearity(agnp.einsum('pdh,pnd->pnh', W_output, hidden) + b_output)

    def forward(self, weights, inputs):
        n_param_sets = inputs.shape[0]
        sequence_length = inputs.shape[1]
        n_sequences = inputs.shape[2]
        
        hiddens = agnp.expand_dims(agnp.expand_dims(self.hidden_init, 0).repeat(n_sequences, 0),0).repeat(n_param_sets, 0)
        cells = agnp.expand_dims(agnp.expand_dims(self.cell_init, 0).repeat(n_sequences, 0),0).repeat(n_param_sets, 0)
        
        outputs = [self.get_output(weights, hiddens)]
        for idx in range(sequence_length):
            input = inputs[:,idx,:,:]
            hiddens, cells = self.update_hidden(weights, input, hiddens, cells)
            outputs.append(self.get_output(weights, hiddens))
        
        out = agnp.array(outputs).reshape((inputs.shape[0],inputs.shape[1] + 1, inputs.shape[2], inputs.shape[3]))
        return out

    def get_params(self):
        return self.params

    def set_params(self, params):
        self.params = params

class BaseNeuralNetwork(Model):

    def __init__(self):

        self.num_weights = 0
        self.layers = []
    
    def unpack_layers(self, weights):
        num_weight_sets = len(weights)
        for layer in self.layers:
            yield weights[:, :layer.num_weights]
            weights = weights[:, layer.num_weights:]

    def predict(self, weights, inputs):
        t = len(weights)
        inputs = agnp.expand_dims(inputs, 0).repeat(t,0)
        for i, w in enumerate(self.unpack_layers(weights)):
            inputs = self.layers[i].forward(w, inputs)
        return inputs

    def add_layer(self, layer):
        self.layers.append(layer)
        self.num_weights += layer.num_weights

    def get_params(self):
        return np.hstack([layer.get_params() for layer in self.layers])

    def set_params(self, weights):
        for i, w in enumerate(self.unpack_layers(weights)):
            self.layers[i].set_params(w[0:1,:])

class DenseNeuralNetwork(BaseNeuralNetwork):

    def  __init__(self, layer_dims, nonlinearity = lambda x: linear):
        super().__init__()
        self.nonlinearity = nonlinearity
        
        shapes = list(zip(layer_dims[:-1], layer_dims[1:]))
        for m, n in shapes[:len(shapes)-1]:
            self.add_layer(FCLayer(m, n, self.nonlinearity))

        self.add_layer(FCLayer(shapes[-1][0], shapes[-1][1], nonlinearity = lambda x: x))

if __name__ == '__main__':
    from os.path import join, dirname
    from ML_Lib.models.model import DifferentiableProbabilityModel
    from ML_Lib.inference.optimization import MAP
    from ML_Lib.inference.variational_inference import *

    def string_to_one_hot(string, maxchar):
        """Converts an ASCII string to a one-of-k encoding."""
        ascii = np.array([ord(c) for c in string]).T
        return np.array(ascii[:,None] == np.arange(maxchar)[None, :], dtype=int)

    def one_hot_to_string(one_hot_matrix):
        return "".join([chr(np.argmax(c)) for c in one_hot_matrix])

    def build_dataset(filename, sequence_length, alphabet_size, max_lines=-1):
        """Loads a text file, and turns each line into an encoded sequence."""
        with open(filename) as f:
            content = f.readlines()
        content = content[:max_lines]
        content = [line for line in content if len(line) > 2]   # Remove blank lines
        seqs = np.zeros((sequence_length, len(content), alphabet_size))
        for ix, line in enumerate(content):
            padded_line = (line + " " * sequence_length)[:sequence_length]
            seqs[:, ix, :] = string_to_one_hot(padded_line, alphabet_size)
        return seqs

    num_chars = 128

    # Learn to predict our own source code.
    text_filename = join(dirname(__file__), 'neural_network.py')
    train_inputs = build_dataset(text_filename, sequence_length=30,
                                 alphabet_size=num_chars, max_lines=60)

    class SimpleRNNWordModel(DifferentiableProbabilityModel):

        def __init__(self):
            self.b = BaseNeuralNetwork()
            self.b.add_layer(RNNLayer(train_inputs.shape[2], 40, train_inputs.shape[2], hidden_nonlinearity = sigmoid))
            self.params = self.b.get_params()
        
            self.full_grad_log_prob = autograd.elementwise_grad(self.full_log_prob)

        def full_log_prob(self, params, X, y):
            loglik = np.zeros((len(params), 1))
            output = self.b.predict(params, X)
            # Output: n_param_sets, sequence_length, n_sequences, n_output_dim
            logprobs = output - logsumexp(output, axis = 3, keepdims = True)
            sequence_length, n_sequences, _ = X.shape
            for t in range(sequence_length):
                loglik += agnp.sum(logprobs[:,t,:,:] * y[t],axis = (1,2)).reshape((-1,1))
            log_prior = agnp.sum(agsp.stats.norm.logpdf(params, 0, 1000),axis = 1)
            return loglik/(sequence_length * n_sequences) + log_prior

        def predict(self, params, X):
            output = self.b.predict(params, X)
            # Output: n_param_sets, sequence_length, n_sequences, n_output_dim
            logprobs = output - logsumexp(output, axis = 3, keepdims = True)
            return logprobs

        def set_data(self, X, y):
            self.log_prob = lambda params: self.full_log_prob(params, X, y)
            self.grad_log_prob = lambda params: agnp.clip(self.full_grad_log_prob(params, X, y), -1000, 1000)

        def get_params(self):
            return self.params

        def set_params(self, params):
            self.params = params
            self.b.set_params(params)

    s = SimpleRNNWordModel()
    s.set_data(train_inputs, train_inputs)
    
    # Getting initial parameters
    m2 = MAP(s)
    for _ in range(5):
        m2.train(num_iters = 100)
    for t in range(20):
        text = ""
        for i in range(30):
            seqs = string_to_one_hot(text, 128)[:, np.newaxis, :]
            logprobs = s.predict(s.get_params(), seqs)[0,-1].ravel()
            text += chr(np.random.choice(len(logprobs), p = np.exp(logprobs)))
        print(text)
    
    m = BlackBoxKLQPScore(s)
    def callback(weights, iter, grads):
        if iter % 50 == 0:
            print("Iteration ============================= %d ========================" % iter)
            m.v_dist.set_params(weights)
            mean_params = m.v_dist.v_params[:m.v_dist.n_params].reshape((1,-1))
            for t in range(5):
                text = ""
                for i in range(30):
                    seqs = string_to_one_hot(text, 128)[:, np.newaxis, :]
                    logprobs = s.predict(mean_params, seqs)[0,-1].ravel()
                    text += chr(np.random.choice(len(logprobs), p = np.exp(logprobs)))
                print(text)

    m.train(1, n_elbo_samples = 5, step_size = 0.001, num_iters = 5000, callback = callback)
    
    param_samples = m.sample(1000)
    mean_params = agnp.mean(param_samples, axis = 0).reshape((1,-1))
    for t in range(30):
        text = ""
        for i in range(30):
            seqs = string_to_one_hot(text, 128)[:, np.newaxis, :]
            logprobs = s.predict(mean_params, seqs)[0,-1].ravel()
            text += chr(np.random.choice(len(logprobs), p = np.exp(logprobs)))
        print(text)
    """
    for i in range(5):
        m.train(num_iters = 1000)

    for t in range(20):
        text = ""
        for i in range(30):
            seqs = string_to_one_hot(text, 128)[:, np.newaxis, :]
            logprobs = s.predict(s.get_params(), seqs)[0,-1].ravel()
            text += chr(np.random.choice(len(logprobs), p = np.exp(logprobs)))
        print(text)
    """
