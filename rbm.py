import numpy as np
from typing import Tuple

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
        ## initialize the model parameters : 
        self.a = np.zeros(p)
        self.b = np.zeros(q)
        self.w = np.random.normal(size = (p,q))


        self.X = X
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        
    def initialize_centers(self, X: np.ndarray) -> np.ndarray:
        """Initialize RBM centers using subset of training data"""
        indices = np.random.permutation(X.shape[0])[:self.batch_size]
        return X[indices]
    
    def RBM_kernel(self, X: np.ndarray, centers: np.ndarray) -> np.ndarray:
        """
        Compute RBM kernel between X and centers
        
        Args:
            X: Input matrix of shape (n_samples, n_features)
            centers: Centers matrix of shape (n_centers, n_features)
            
        Returns:
            Kernel matrix of shape (n_samples, n_centers)
        """
        n_samples = X.shape[0]
        n_centers = centers.shape[0]
        
        # Compute pairwise squared Euclidean distances
        X_squared = np.sum(X**2, axis=1).reshape(n_samples, 1)
        centers_squared = np.sum(centers**2, axis=1).reshape(1, n_centers)
        cross_term = np.dot(X, centers.T)
        distances = X_squared + centers_squared - 2 * cross_term
        
        # Apply RBM kernel
        gamma = 1.0 / X.shape[1]  # Default gamma value
        return np.exp(-gamma * distances)
    
    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        Forward pass through the network
        
        Args:
            X: Input data matrix
            
        Returns:
            output: Network output
            cache: Cache of intermediate values for backward pass
        """
        # X_batch computation
        X_batch = X.copy()
        
        # Edge computation
        edge = self.RBM_kernel(X_batch, self.centers)
        
        # Hidden layer computations
        h_0 = edge
        p_h_v_0 = self.entre_sortie(h_0)
        v_1 = (np.random.rand(self.batch_size) < p_h_v_0).astype(float)
        p_h_v_1 = self.sortie_entre(v_1)
        
        # Gradient computations
        grad_a = np.sum(X_batch - v_1, axis=0, keepdims=True)
        grad_b = np.sum(p_h_v_0 - p_h_v_1, axis=0, keepdims=True)
        grad_w = np.dot(X_batch.T, p_h_v_0) - np.dot(v_1.T, p_h_v_1)
        
        cache = {
            'X_batch': X_batch,
            'edge': edge,
            'h_0': h_0,
            'p_h_v_0': p_h_v_0,
            'v_1': v_1,
            'p_h_v_1': p_h_v_1
        }
        
        return grad_w, grad_a, grad_b, cache
    
    def backward(self, cache: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Backward pass through the network
        
        Args:
            cache: Cache of intermediate values from forward pass
            
        Returns:
            Gradients for weights, a, and b parameters
        """
        X_batch = cache['X_batch']
        p_h_v_0 = cache['p_h_v_0']
        v_1 = cache['v_1']
        p_h_v_1 = cache['p_h_v_1']
        
        # Compute gradients
        d_log_po_daj = X_batch - v_1
        d_log_po_dvi = p_h_v_0 - p_h_v_1
        d_log_po_dbj = p_h_v_0 - v_1
        
        return d_log_po_daj, d_log_po_dvi, d_log_po_dbj
    
    def entre_sortie(self, h: np.ndarray) -> np.ndarray:
        """Compute transition from hidden to visible layer"""
        return 1 / (1 + np.exp(-h))
    
    def sortie_entre(self, v: np.ndarray) -> np.ndarray:
        """Compute transition from visible to hidden layer"""
        return 1 / (1 + np.exp(-v))
    
    def train(self) -> list:
        """
        Train the RBM network
        
        Returns:
            List of errors during training
        """
        errors = []
        self.centers = self.initialize_centers(self.X)
        
        for epoch in range(self.n_epochs):
            # Mini-batch sampling
            indices = np.random.permutation(self.X.shape[0])
            
            for i in range(0, self.X.shape[0], self.batch_size):
                batch_indices = indices[i:i + self.batch_size]
                X_batch = self.X[batch_indices]
                
                # Forward and backward passes
                grad_w, grad_a, grad_b, cache = self.forward(X_batch)
                d_log_po_daj, d_log_po_dvi, d_log_po_dbj = self.backward(cache)
                
                # Update parameters
                self.centers -= self.lr * grad_w
                
                # Compute reconstruction error
                X_rec = self.reconstruct(X_batch)
                error = np.mean((X_batch - X_rec) ** 2)
                errors.append(error)
                
        return errors
    
    def reconstruct(self, X: np.ndarray) -> np.ndarray:
        """Reconstruct input data"""
        h = self.RBM_kernel(X, self.centers)
        return self.entre_sortie(h)