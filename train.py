import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from multi_layer_funcs import *

## Load Data
# Load data
df = pd.read_csv('../nr-ar.smiles', sep='\t', header=None).dropna()

# Rename columns
df.columns = ['Smiles', 'ID', 'Active']

# Add fingerprints
mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize = 2048)
df['mol'] = df.Smiles.apply(Chem.MolFromSmiles)
df.dropna(inplace = True)
df['fingerprint'] = df.mol.apply(lambda x: np.array(mfpgen.GetFingerprint(x)))

#Shuffle dataframe rows
df = df.sample(frac=1, random_state=42).reset_index().drop(columns=['index', 'Smiles', 'mol'])

# Make train/test/val splits
num_samples = len(df)
train_cutoff = int(num_samples*0.8)
test_cutoff = int(num_samples*0.9)
df_train = df[0:train_cutoff]
df_test = df[train_cutoff:test_cutoff]
df_val = df[test_cutoff:]

## Initialize Weights
# Initialize hidden layer weight matrix and bias_2
input_size = 2048
hidden_size = 128
output_size = 1

# Xavier Initialization
weights_1 = np.random.randn(input_size, hidden_size) * np.sqrt(2 / (input_size + hidden_size))
bias_1 = np.zeros(hidden_size)

# Initialize output layer weights and bias_2
weights_2 = np.random.randn(hidden_size, output_size) * np.sqrt(2 / (hidden_size + output_size))
bias_2 = 0
num_epochs = 200
cutoff = 0.5

# Initialize leanring rate and warm-up
max_learning_rate = 0.5
warmup_epochs = 10

# Train data
X = np.stack(df_train['fingerprint'].to_list())
y = np.array(df_train['Active'].to_list())

# Val data
X_val = np.stack(df_val['fingerprint'].to_list())
y_val = np.array(df_val['Active'].to_list())

min_loss = 100

all_losses = []
all_losses_val = []


## Train Model
for i in range(num_epochs):

    ## Forward pass
    z1 = np.dot(X, weights_1) + bias_1
    a1 = relu(z1)

    z2 = np.dot(a1, weights_2) + bias_2
    a2 = sigmoid(z2)

    # Sigmoid activation function and classify data
    probs = a2
    preds = (a2 >= 0.5).astype(int)

    # Compute loss
    epoch_loss = np.mean(weighted_bce(y_true=y, y_pred=probs, pos_weight=2, neg_weight=1))
    all_losses.append(epoch_loss)

    # Validation forward pass
    z1_val = X_val @ weights_1 + bias_1        
    a1_val = relu(z1_val)                      
    z2_val = a1_val @ weights_2 + bias_2       
    a2_val = sigmoid(z2_val)  
    
    # Validation loss
    val_loss = np.mean(weighted_bce(y_true=y_val, y_pred=a2_val, pos_weight=2, neg_weight=1))
    all_losses_val.append(val_loss)  


    ## Save Best Model
    if val_loss < min_loss:
        min_loss = val_loss
        best_epoch = i

        # Save best model weights
        best_weights_1 = weights_1.copy()
        best_weights_2 = weights_2.copy()
        best_bias_1 = bias_1.copy()
        best_bias_2 = bias_2

        # Save best predictions
        best_preds = preds.copy()
        best_probs = probs.copy()


    ## Backpropogation 
    # Output layer
    dL_da2 = (a2 - y.reshape(-1,1)) / len(y)
    dL_dz2 = dL_da2

    dL_dweights_2 = a1.T @ dL_dz2
    dL_dbias_2 = np.sum(dL_dz2, axis=0)

    # Hidden layer
    dL_da1 = dL_dz2 @ (weights_2.T)
    dL_dz1 = dL_da1 * relu_prime(z1)

    dL_dweights_1 = X.T @ dL_dz1
    dL_dbias_1 = np.sum(dL_dz1, axis=0)

    # Clip gradients
    [dL_dweights_1, dL_dbias_1, dL_dweights_2, dL_dbias_2] = clip_gradients(
        [dL_dweights_1, dL_dbias_1, dL_dweights_2, dL_dbias_2], max_norm=5.0
    )

    # Learning rate warm-up
    if i < warmup_epochs:
        learning_rate = max_learning_rate * (i+1) / warmup_epochs
    else:
        learning_rate = max_learning_rate

    # Update weights
    weights_2 -= learning_rate * dL_dweights_2
    bias_2 -= learning_rate * dL_dbias_2
    weights_1 -= learning_rate * dL_dweights_1
    bias_1 -= learning_rate * dL_dbias_1