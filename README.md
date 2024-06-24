# Seed Sorting RBF NN

This document presents the implementation of a Radial Basis Function (RBF) Neural Network model for the purpose of identifying different types of olive seeds. The RBF Neural Network is a type of artificial neural network that uses radial basis functions as activation functions, and is particularly effective for pattern recognition tasks. This implementation leverages Python and relevant libraries to achieve accurate classification of olive seed varieties.

# RBF NN

### Radial Basis Function Neural Network (RBF NN) 

#### Overview
A Radial Basis Function Neural Network (RBF NN) is a type of artificial neural network that uses radial basis functions as activation functions. It is particularly effective for pattern recognition, classification, and function approximation tasks.

#### Architecture
An RBF NN typically consists of three layers:

1. **Input Layer**: 
   - This layer consists of input neurons that pass the input features to the next layer without any transformation.

2. **Hidden Layer**: 
   - This layer contains neurons with radial basis functions as their activation functions. 
   - Each neuron computes the distance between the input and a center point, then applies the radial basis function to this distance. The Gaussian function is the most common choice for the radial basis function, defined as:
     \[
     \phi(\|x - c\|) = \exp\left(-\frac{\|x - c\|^2}{2\sigma^2}\right)
     \]
   - Here, \(x\) is the input vector, \(c\) is the center of the RBF neuron, and \(\sigma\) is the spread parameter.

3. **Output Layer**: 
   - This layer performs a linear combination of the hidden layer outputs and provides the final output of the network.
   - The output is typically a weighted sum of the radial basis functions.

#### Training Process
The training of an RBF NN involves two main steps:

1. **Determining the Centers and Spreads**:
   - **Centers**: The center points \(c_i\) of the RBF neurons are typically determined using clustering algorithms like K-Means.
   - **Spreads**: The spread \(\sigma\) can be fixed or computed based on the distances between centers.

2. **Training the Output Weights**:
   - Once the centers and spreads are fixed, the output weights are trained using a linear learning algorithm, such as least squares or gradient descent. The objective is to minimize the error between the predicted output and the actual target values.

#### Advantages
- **Local Approximation**: RBF neurons respond only to inputs close to their centers, providing good local approximation properties.
- **Flexibility**: The network can model complex, non-linear functions.
- **Simplicity**: Training the output layer is straightforward and computationally efficient.

#### Applications
- **Pattern Recognition**: Effective for classifying complex patterns.
- **Function Approximation**: Used for approximating unknown functions based on input-output pairs.
- **Time Series Prediction**: Applicable in forecasting future values based on historical data.

![image](https://github.com/dhananjay-ryu-jin-sama/Seed_Sorting_RBF_NN/assets/144810835/1aa5f190-b0d0-4884-81f9-18e81b5c978c)

# Result

In the training phase the model can predict up to 95% pass rate and with 0.0155 error.

![results](https://github.com/dhananjay-ryu-jin-sama/Seed_Sorting_RBF_NN/assets/144810835/091be4fb-480c-445f-8009-08827a49e8ba)


