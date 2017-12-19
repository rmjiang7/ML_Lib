import numpy as np
import autograd as ag
import autograd.numpy as agnp
from ML_Lib.models.model import ProbabilityModel
from ML_Lib.models.neural_network import DenseNeuralNetwork, BaseNeuralNetwork, ConvLayer, FCLayer

class VariationalAutoencoderShared(ProbabilityModel):

    def __init__(self, encoder_dims, decoder_dims, nonlinearity = lambda x: (x > 0)*x):
        self.nonlinearity = nonlinearity
        self.encoder = DenseNeuralNetwork(encoder_dims, nonlinearity)
        self.decoder = DenseNeuralNetwork(decoder_dims, nonlinearity)
        self.latent_dim = decoder_dims[0]
        self.params = np.hstack([self.encoder.get_params(),self.decoder.get_params()])

        self.full_grad_log_prob = ag.elementwise_grad(self.full_log_prob)
        
    def unpack_params(self, params):
        n_encoder_params = self.encoder.get_params().shape[1]
        return params[:,:n_encoder_params], \
               params[:,n_encoder_params:]

    def full_log_prob(self, params, X):
        enc_params, dec_params = self.unpack_params(params)
        enc = self.encoder.predict(enc_params, X)
        enc_mu, enc_sig = enc[:,:,:self.latent_dim], enc[:,:,self.latent_dim:]
        z_samp = enc_mu + agnp.sqrt(agnp.exp(enc_sig) + 1e-10) * agnp.random.normal(0,1,size=(1,enc_mu.shape[1],enc_mu.shape[2]))
        pred = self.decoder.predict(dec_params,z_samp[0,:,:])
        prob = (1/(1 + agnp.exp(-agnp.clip(pred,-300,300))))[0,:,:]
        enc_loss = agnp.sum(X * agnp.log(prob + 1e-10) + (1 - X)*agnp.log(1 - prob + 1e-10),axis = 1)
        dec_loss = 0.5 * agnp.sum((1 + enc_sig - enc_mu**2 - agnp.exp(enc_sig))[0,:,:],axis=1)
        return agnp.mean(enc_loss + dec_loss)

    def set_data(self, X):
        self.log_prob = lambda params: self.full_log_prob(params, X)
        self.grad_log_prob = lambda params: agnp.clip(self.full_grad_log_prob(params, X), -1000, 1000)

    def get_params(self):
        return self.params

    def set_params(self, params):
        self.params = params

    def predict(self, n_samples = 10, latent_space = None):
        _,dec_params = self.unpack_params(self.params)
        if latent_space is None:
            latent_space = np.random.normal(0, 10, size = (n_samples,self.latent_dim))
        return (1/(1 + agnp.exp(-agnp.clip(self.decoder.predict(dec_params,latent_space),-300,300))))[0,:,:]

    def encode(self, samples):
        enc_params,_ = self.unpack_params(self.params)
        enc = self.encoder.predict(enc_params, samples)
        enc_mu, enc_sig = enc[:,:,:self.latent_dim], enc[:,:,self.latent_dim:]
        return enc_mu, enc_sig

class VariationalAutoencoderSharedConv(ProbabilityModel):

    def __init__(self, encoder_dims, decoder_dims, nonlinearity = lambda x: (x > 0)*x):
        self.nonlinearity = nonlinearity
        b = BaseNeuralNetwork()
        b.add_layer(ConvLayer([1,28,28],[5,5],10))
        #b.add_layer(ConvLayer([10,24,24],[5,5],5))
        b.add_layer(FCLayer(np.prod(b.layers[-1].get_output_shape()), 20, nonlinearity = lambda x: x * (x > 0)))
        b.add_layer(FCLayer(20, decoder_dims[0] * 2, nonlinearity = lambda x : x))
        self.encoder = b
        self.decoder = DenseNeuralNetwork(decoder_dims, nonlinearity)
        self.latent_dim = decoder_dims[0]
        self.params = np.hstack([self.encoder.get_params(),self.decoder.get_params()])

        self.full_grad_log_prob = ag.elementwise_grad(self.full_log_prob)
        
    def unpack_params(self, params):
        n_encoder_params = self.encoder.get_params().shape[1]
        return params[:,:n_encoder_params], \
               params[:,n_encoder_params:]

    def full_log_prob(self, params, X):
        enc_params, dec_params = self.unpack_params(params)
        enc = self.encoder.predict(enc_params, X)
        enc_mu, enc_sig = enc[:,:,:self.latent_dim], enc[:,:,self.latent_dim:]
        z_samp = enc_mu + agnp.sqrt(agnp.exp(enc_sig) + 1e-10) * agnp.random.normal(0,1,size=(1,enc_mu.shape[1],enc_mu.shape[2]))
        pred = self.decoder.predict(dec_params,z_samp[0,:,:])
        prob = (1/(1 + agnp.exp(-agnp.clip(pred,-300,300))))[0,:,:]
        enc_loss = agnp.sum(X.reshape((X.shape[0],-1)) * agnp.log(prob + 1e-10) + (1 - X.reshape((X.shape[0],-1)))*agnp.log(1 - prob + 1e-10),axis = 1)
        dec_loss = 0.5 * agnp.sum((1 + enc_sig - enc_mu**2 - agnp.exp(enc_sig))[0,:,:],axis=1)
        return agnp.mean(enc_loss + dec_loss)

    def set_data(self, X):
        self.log_prob = lambda params: self.full_log_prob(params, X)
        self.grad_log_prob = lambda params: agnp.clip(self.full_grad_log_prob(params, X), -1000, 1000)
    
    def get_params(self):
        return self.params

    def set_params(self, params):
        self.params = params

    def predict(self, n_samples = 10, latent_space = None):
        _,dec_params = self.unpack_params(self.params)
        if latent_space is None:
            latent_space = np.random.normal(0, 10, size = (n_samples,self.latent_dim))
        return (1/(1 + agnp.exp(-agnp.clip(self.decoder.predict(dec_params,latent_space),-300,300))))[0,:,:]

    def encode(self, samples):
        enc_params,_ = self.unpack_params(self.params)
        enc = self.encoder.predict(enc_params, samples)
        enc_mu, enc_sig = enc[:,:,:self.latent_dim], enc[:,:,self.latent_dim:]
        return enc_mu, enc_sig

if __name__ == "__main__":

    from ML_Lib.inference.map import MAP
    from tensorflow.examples.tutorials.mnist import input_data
    import scipy.misc
    import sys
    
    mnist = input_data.read_data_sets("/Users/richardjiang/GradMLearning/datasets/MNIST_data/", one_hot=True)
    vae = VariationalAutoencoderShared([784,500,500,40],[20,500,500,784])
    #vae = VariationalAutoencoderSharedConv([784,200,200,20],[20,200,200,784])
    if sys.argv[1] == 'train':
        print("Starting training!")
        try:
            i = 0
            while True:
                l = 0
                for j in range(int(mnist.train.num_examples/200)):
                    batch_xs, _ = mnist.train.next_batch(200)
                    vae.set_data(batch_xs)#.reshape((batch_xs.shape[0],1,28,28)))
                    m = MAP(vae)
                    m.train(step_size = 0.01,num_iters=1)
                    l += vae.log_prob(vae.get_params())
                
                print("Epoch %d" % (i))
                print("Average loss %f" % (l/mnist.train.num_examples * 200))
                i += 1
        except (KeyboardInterrupt, SystemExit):
            print("Saving VAE params!")
            np.save("/Users/richardjiang/Downloads/vae_params.npy", vae.get_params())
    elif sys.argv[1] == 'test':
        import matplotlib.pyplot as plt

        params = np.load("/Users/richardjiang/Downloads/vae_params.npy")
        vae.set_params(params)
        conv_layer = vae.encoder.layers[0]
        batch_xs, _ = mnist.train.next_batch(10)
        """
        filters = conv_layer.forward(conv_layer.get_params(), batch_xs.reshape((1, 1,1,28,28)))
        
        f, axarr = plt.subplots(2, 5, figsize=(11,4))
        for i in range(2):
            for j in range(5):
                axarr[i][j].imshow(filters[0,0,i * 2 + j,:,:])
                axarr[i][j].axis('off')
        plt.show()
       
        """
        enc_mu, enc_sig = vae.encode(batch_xs)#.reshape((10,1,28,28)))
        z_samp = enc_mu + agnp.sqrt(agnp.exp(enc_sig)) * agnp.random.normal(0,1,size=(1,enc_mu.shape[1],enc_mu.shape[2]))
        out = vae.predict(latent_space = z_samp[0,:,:])

        f, axarr = plt.subplots(4, 5, figsize=(11,4))
        for i in range(2):
            for j in range(5):
                axarr[i][j].imshow(out[2 * i + j,:].reshape((28,28)))
                axarr[i][j].axis('off')
                axarr[i + 2][j].imshow(batch_xs[2 * i + j,:].reshape((28,28)))              
                axarr[i + 2][j].axis('off')
        plt.show()


