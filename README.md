# Tox21 Classification Neural Network From Scratch
This project implements a fully vectorized two-layer neural network from scratch using only **NumPy** to classify chemical compounds based on their molecular fingerprints and activity.

These models utilize data from the [Tox21](https://tripod.nih.gov/tox21/challenge/data.jsp#) public dataset.

## Key Features
Some techniques used in the model include:
- Fully vectorized forward and backward pass
- Hidden layer with ReLU activation
- Sigmoid output for binary classification
- **Weighted binary cross-entropy loss** to handle class imbalance
- **Gradient clipping** to prevent exploding gradients
- **Learning rate warm-up** for stable early training
- **Validation loss tracking** and best model saving
- Clean loss visualizations (train vs val) using `seaborn`

    

***The  multi-layer model is able to acheive an F1  > 0.6  and PRC-AUC > 0.5 on unseen test data:***

![alt text](plots/confusion_matrix.png)
![alt text](plots/prc_curve.png)


