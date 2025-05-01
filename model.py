import numpy as np
from utils import *

class NeuralNet:
    def __init__(self, input_size=2048, hidden_size=128, output_size=1, 
                 num_epochs=200, max_learning_rate=0.5, warmup_epochs=10,
                 val_split=0.1, random_seed=42):
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_epochs = num_epochs
        self.max_learning_rate = max_learning_rate
        self.warmup_epochs = warmup_epochs
        self.val_split = val_split
        self.random_seed = random_seed

    def fit(self, X, y):

        ## Val Split
        np.random.seed(self.random_seed)
        X = np.random.permutation(X)
        np.random.seed(self.random_seed)
        y = np.random.permutation(y)

        num_samples = len(X)
        X_val = X[0:int(num_samples*self.val_split)]
        y_val = y[0:int(num_samples*self.val_split)]

        X = X[int(num_samples*self.val_split):]
        y = y[int(num_samples*self.val_split):]

        X = np.stack(X)
        X_val = np.stack(X_val)

        ## Initialize Weights
        # Xavier Initialization
        weights_1 = np.random.randn(self.input_size, self.hidden_size) * np.sqrt(2 / (self.input_size + self.hidden_size))
        bias_1 = np.zeros(self.hidden_size)

        # Initialize output layer weights and bias_2
        weights_2 = np.random.randn(self.hidden_size, self.output_size) * np.sqrt(2 / (self.hidden_size + self.output_size))
        bias_2 = 0
        cutoff = 0.5
        min_loss = 1e9

        for i in range(self.num_epochs):

            ## Forward pass
            z1, a1, z2, a2 = forward_pass(X, weights_1, bias_1, weights_2, bias_2)

            # Sigmoid activation function and classify data
            probs = a2
            preds = (a2 >= cutoff).astype(int)

            # Compute loss
            epoch_loss = np.mean(weighted_bce(y_true=y, y_pred=probs, pos_weight=2, neg_weight=1))

            # Validation forward pass
            z1_val, a1_val, z2_val, a2_val = forward_pass(X_val, weights_1, bias_1, weights_2, bias_2)  
            
            # Validation loss
            val_loss = np.mean(weighted_bce(y_true=y_val, y_pred=a2_val, pos_weight=2, neg_weight=1))

            ## Save Best Model
            if val_loss < min_loss:
                min_loss = val_loss
                best_epoch = i

                # Save best model weights
                self.best_weights_1 = weights_1.copy()
                self.best_weights_2 = weights_2.copy()
                self.best_bias_1 = bias_1.copy()
                self.best_bias_2 = bias_2

                # Save best predictions
                self.best_preds = preds.copy()
                self.best_probs = probs.copy()


            ## Backpropogation 
            dL_dweights_1, dL_dbias_1, dL_dweights_2, dL_dbias_2 = back_prop(a2, z2, a1, z1, weights_2, X, y)

            # Clip gradients
            [dL_dweights_1, dL_dbias_1, dL_dweights_2, dL_dbias_2] = clip_gradients(
                [dL_dweights_1, dL_dbias_1, dL_dweights_2, dL_dbias_2], max_norm=5.0
            )

            # Learning rate warm-up
            if i < self.warmup_epochs:
                learning_rate = self.max_learning_rate * (i+1) / self.warmup_epochs
            else:
                learning_rate = self.max_learning_rate

            # Update weights
            weights_2 -= learning_rate * dL_dweights_2
            bias_2 -= learning_rate * dL_dbias_2
            weights_1 -= learning_rate * dL_dweights_1
            bias_1 -= learning_rate * dL_dbias_1

            if (i+1)%10 == 0:
                print(f'{i+1} / {self.num_epochs}')

            if i+1 == self.num_epochs:
                print('Model training complete.')


    def predict(self, X):

        ## Forward pass
        z1, a1, z2, a2 = forward_pass(X, self.best_weights_1, self.best_bias_1, self.best_weights_2, self.best_bias_2)

        return a2



