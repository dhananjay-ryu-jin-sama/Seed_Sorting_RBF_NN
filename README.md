# Seed_Sorting_RBF_NN

This document presents the implementation of a Radial Basis Function (RBF) Neural Network model for the purpose of identifying different types of olive seeds. The RBF Neural Network is a type of artificial neural network that uses radial basis functions as activation functions, and is particularly effective for pattern recognition tasks. This implementation leverages Python and relevant libraries to achieve accurate classification of olive seed varieties.

# RBF NN

Radial basis function (RBF) networks typically have three layers: an input layer, a hidden layer with a non-linear RBF activation function and a linear output layer. The input can be modeled as a vector of real numbers 
𝑥
∈
𝑅
𝑛
{\displaystyle \mathbf {x} \in \mathbb {R} ^{n}}. The output of the network is then a scalar function of the input vector, 
𝜑
:
𝑅
𝑛
→
𝑅
{\displaystyle \varphi :\mathbb {R} ^{n}\to \mathbb {R} }, and is given by

𝜑
(
𝑥
)
=
∑
𝑖
=
1
𝑁
𝑎
𝑖
𝜌
(
|
|
𝑥
−
𝑐
𝑖
|
|
)
{\displaystyle \varphi (\mathbf {x} )=\sum _{i=1}^{N}a_{i}\rho (||\mathbf {x} -\mathbf {c} _{i}||)}
where 
𝑁
{\displaystyle N} is the number of neurons in the hidden layer, 
𝑐
𝑖
{\displaystyle \mathbf {c} _{i}} is the center vector for neuron 
𝑖
{\displaystyle i}, and 
𝑎
𝑖
{\displaystyle a_{i}} is the weight of neuron 
𝑖
{\displaystyle i} in the linear output neuron. Functions that depend only on the distance from a center vector are radially symmetric about that vector, hence the name radial basis function. In the basic form, all inputs are connected to each hidden neuron. The norm is typically taken to be the Euclidean distance (although the Mahalanobis distance appears to perform better with pattern recognition[4][5][editorializing]) and the radial basis function is commonly taken to be Gaussian

𝜌
(
‖
𝑥
−
𝑐
𝑖
‖
)
=
exp
⁡
[
−
𝛽
𝑖
‖
𝑥
−
𝑐
𝑖
‖
2
]
{\displaystyle \rho {\big (}\left\Vert \mathbf {x} -\mathbf {c} _{i}\right\Vert {\big )}=\exp \left[-\beta _{i}\left\Vert \mathbf {x} -\mathbf {c} _{i}\right\Vert ^{2}\right]}.
The Gaussian basis functions are local to the center vector in the sense that

lim
|
|
𝑥
|
|
→
∞
𝜌
(
‖
𝑥
−
𝑐
𝑖
‖
)
=
0
{\displaystyle \lim _{||x||\to \infty }\rho (\left\Vert \mathbf {x} -\mathbf {c} _{i}\right\Vert )=0}
i.e. changing parameters of one neuron has only a small effect for input values that are far away from the center of that neuron.

Given certain mild conditions on the shape of the activation function, RBF networks are universal approximators on a compact subset of 
𝑅
𝑛
{\displaystyle \mathbb {R} ^{n}}.[6] This means that an RBF network with enough hidden neurons can approximate any continuous function on a closed, bounded set with arbitrary precision.

The parameters 
𝑎
𝑖
{\displaystyle a_{i}}, 
𝑐
𝑖
{\displaystyle \mathbf {c} _{i}}, and 
𝛽
𝑖
{\displaystyle \beta _{i}} are determined in a manner that optimizes the fit between 
𝜑{\displaystyle \varphi } and the data.
![image](https://github.com/dhananjay-ryu-jin-sama/Seed_Sorting_RBF_NN/assets/144810835/1aa5f190-b0d0-4884-81f9-18e81b5c978c)
