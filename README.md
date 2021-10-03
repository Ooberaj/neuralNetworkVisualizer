# neuralNetworkVisualizer

The visualization's neural network is trained on the MNIST dataset to recognize hand-written digits. The network has 784 input neurons with activations loaded from 28x28 images, 30 neurons in 1 hidden layer, and 10 output neurons representing numbers 0 through 9.

As the visualization is ran, the training process is displayed. The input image for each training example is loaded onto the input layer as the network is trained real-time. Simultaneously, the thickness and color of each edge and node border is changed. Border thickness and color are used to represent biases while edge thickness and color are used to represent weights. These changes are also displayed in each node's tooltip and heatmap.
