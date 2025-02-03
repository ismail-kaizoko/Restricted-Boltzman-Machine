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
        ## init model parameters :
        self.p = p
        self.q = q



        ## initialize the model learnable parameters : 
        self.a = np.zeros(p)
        self.b = np.zeros(q)
        # self.w = np.random.rand(p,q)
        self.w = np.random.randn(p, q) * 0.01



        #initialize model training parameters
        self.X = X
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr

    
    def input_output(self, X_np):
        """
        Does a forward pass from observed states x to the hidden state h
        takes a mini-batch of size n of vectors x[p] and returns the result of vectors of h[q]
        return : H_nq 
        """
        w = self.w
        b = self.b

        H_nq = 1/(1+np.exp(-(X_np @ w + b)))
        return H_nq


    
    def output_input(self, H_nq):
        """
        Does a forward pass from hidden states h to the observed state x
        takes a mini-batch of size n of vectors h[q] and returns the result of vectors of x[p]
        return : H_nq 
        """
        w = self.w
        a = self.a

        X_np = 1/(1+np.exp(-(H_nq @ w.T + a)))
        return X_np


    def shuffle_dataset(self, X):
        np.random.shuffle(X)
        return X



    def train(self):

        loss = []
        
        for i in range(self.n_epochs):
            X = self.shuffle_dataset(self.X)
            n = len(X)
            runs = n//self.batch_size +1

            for j in range(runs):
                X_batch = X[j:min(j+self.batch_size,n)]
                t_b = len(X_batch)

                v_0 = X_batch
                
                # here we do one step of Gibbs sampling to approxiamte an expectation term in the derivative d(log p_theta)/d w_ij:
                 
                # compute probability using sigmoid :
                p_h_v_0 = self.input_output(v_0)   # of size : [t_b, p]
                #samlpe initial hidden vector h_0, supposing pixels iid and each follows bernouli of params p_h_v_0[idx]
                h_0 = (np.random.rand(t_b, self.q) <  p_h_v_0)*1.

                p_v_h_0 = self.output_input(h_0)
                v_1 = (np.random.rand(t_b, self.p) < p_v_h_0)*1.

                p_h_v_1 = self.input_output(v_1)


                # compute gradients using close-form found analyitically : 

                grad_a = np.sum(v_0 - v_1, axis=0)
                grad_b = np.sum(p_h_v_0 - p_h_v_1, axis=0)   # intuitively, after convergence, the two probability will collapse and gradients will get smaller.
                grad_w = (X_batch.T@p_h_v_0) - (v_1.T@p_h_v_1)

                #update model parameters : 
                self.w += (self.lr/t_b)*grad_w
                self.a += (self.lr/t_b)*grad_a
                self.b += (self.lr/t_b)*grad_b
            
            epoch_error = self.reconstruction_loss(self.X)
            loss.append( epoch_error)
        
        return loss


    def reconstruction_loss(self, x):
        p_h_v = self.input_output(x)
        h = (np.random.rand(self.q) <  p_h_v)*1.

        p_v_h = self.output_input(h)
        x_hat = (np.random.rand(self.p) < p_v_h)*1.

        error = np.sum((x-x_hat)**2) / len(x) 

        return error



    def generate_images_GibbsSampling(self, img_size = (20, 16), num_images=10, L=1000):
        """
        Generate multiple images using Gibbs Sampling.

        Args:
            num_images: Number of images to generate.
            L: Number of Gibbs sampling steps.

        Returns:
            A NumPy array of shape (num_images, height, width) containing the generated images.
        """
        generated_images = []

        for _ in range(num_images):
            X = self.X[np.random.randint(len(self.X))]

            for _ in range(L):
                p_h_v = self.input_output(X)
                h = (np.random.rand(self.q) < p_h_v).astype(float)

                p_v_h = self.output_input(h)
                X = (p_v_h > 0.5).astype(float)

            generated_images.append(X.reshape(img_size))
            #generated_images.append(X)

        return np.array(generated_images)
