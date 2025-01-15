import numpy as np

class RBMmodel:
    def __init__(self, p, q, X: np.ndarray, n_epochs: int, batch_size: int, lr: float):
        """
        Initialize RBM Network
        
        Args:
            a,b,W : learnable parameters of the model
            X: Input data matrix
            n_epochs: Number of training epochs
            batch_size: Size of mini-batches
            lr: Learning rate
        """
        ## initialize the model learnable parameters : 
        self.a = np.zeros(p)
        self.b = np.zeros(q)
        self.w = np.random.rand(p,q)

        #initialize model training parameters
        self.X = X
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr

    
    def input_output(X_np):
        """
        Does a forward pass from observed states x to the hidden state h
        takes a mini-batch of size n of vectors x[p] and returns the result of vectors of h[q]
        return : H_nq 
        """
        w = self.w
        b = self.b

        H_nq = 1/(1+np.exp(X_np @ w + b))
        return H_nq


    
    def output_input(H_nq):
        """
        Does a forward pass from hidden states h to the observed state x
        takes a mini-batch of size n of vectors h[q] and returns the result of vectors of x[p]
        return : H_nq 
        """
        w = self.w
        a = self.a

        X_np = 1/(1+np.exp(H_nq @ w.T + a))
        return X_np


    def suffle_dataset():
        pass


    def train():

        loss = []
        
        for i in range(self.n_epochs):
            X = self.shuffle(X)
            n = len(X)
            runs = n//self.batch_size +1

            for j in range(runs):
                X_batch = X[j:min(j+self.batch_size,n)]
                t_b = len(X_batch)

                v_0 = X_batch
                
                # here we do one step of Gibbs sampling to approxiamte an expectation term in the derivative d(log p_theta)/d w_ij:
                 
                # compute probability using sigmoid :
                p_h_v_0 = self.input_output(X_batch)   # of size : [t_b, p]
                #samlpe initial hidden vector h_0, supposing pixels iid and each follows bernouli of params p_h_v_0[idx]
                h_0 = (np.random.rand(t_b, self.q) <  p_h_v_0)*1.

                p_v_h_0 = self.output_input(h_0)
                v_1 = (np.random.rand(t_b, self.p) < p_v_h_0)*1.

                p_h_v_1 = self.input_output(v_1)


                # compute gradients using close-form found analyitically : 

                grad_a = np.sum(X_batch - v_1, axis=0)
                grad_b = np.sum(p_h_v_0 - p_h_v_1, axis=0)   # intuitively, after convergence, the two probability will collapse and gradients will get smaller.
                grad_w = X_batch.T*p_h_v_0 - v_1.T*p_h_v_1

                #update model parameters : 
                self.w += (self.lr/t_b)*grad_w
                self.a += (self.lr/t_b)*grad_a
                self.b += (self.lr/t_b)*grad_b
            
            epoch_error = self.reconstruction_loss(self.X)
            loss.append( epoch_error)


    def reconstruction_loss(x):
        p_h_v = self.input_output(x)
        h = (np.random.rand(self.q) <  p_h_v)*1.

        p_v_h = self.output_input(h)
        x_hat = (np.random.rand(self.p) < p_v_h)*1.

        error = np.sum((x-x_hat)**2) / len(x)


    def generate_image(L = 10):
        """ 
        function used to generate images after convergece:
        this methode uses Gibbs-sampling algorithme to generate an image that matches the learnt distribution of the dataset statrting from a random_image.
        see : https://en.wikipedia.org/wiki/Gibbs_sampling
        """                 
        # initial samlple supposed iid 
        X = (np.random.rand(self.p) <  np.random.rand(self.p))*1.  # generate vector of dimension p where each pixel follows iid bernouli of random parameter 

        for l in range(L):
            p_h_v = self.input_output(X)   # of size : [t_b, p]
            #samlpe initial hidden vector h_0, supposing pixels iid and each follows bernouli of params p_h_v_0[idx]
            h = (np.random.rand(self.q) <  p_h_v)*1.

            p_v_h = self.output_input(h)
            X = (np.random.rand(self.p) < p_v_h)*1.

        return X