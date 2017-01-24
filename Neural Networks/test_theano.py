"""
This file contains code for a neural network implemented in Theano.
It is based on the code and info at http://neuralnetworksanddeeplearning.com/chap6.html#the_code_for_our_convolutional_networks
but (as of writing this) it has been stripped to its essentials, reworked in places, and heavily commented to make sure that I
understand every single line.
"""
import numpy as np
import theano
import theano.tensor as T
from theano import pp
from theano import function
from theano.tensor.nnet import sigmoid, softmax

# When setting up numpy arrays for Theano we have to tell it to use theano datatypes.
floatX = theano.config.floatX

class Network():
    """ A basic neural network consists of multiple layers of neurons. Each neuron's output
    is a function of some or all of the outputs in the previous layer. When the network is trained,
    each neuron learns 'weights' to give each neuron in the previous layer.
    """
    def __init__(self, layers):
        """ 
        Construct a neural network
            layers should be a list of layer instances.

        The input to the network is the data itself, so the first layer essentially receives raw (or preprocessed)
        data values. For example, if we were processing 10x10 images, then the first layer needs 100 inputs.

        The final layer should give us some kind of meaningful output e.g. classifications with one yes/no node for each class.
        """
        self.layers = layers

        # x and y are Theano tensors which will represent variables that we can vary manually. x for input data, y for training/test output data.
        self.x = T.matrix("x")
        self.y = T.ivector("y")

        # We extract a list of every parameter matrix/vector to use for training.
        self.params = [param for layer in self.layers for param in layer.params]

    def rewire(self, mini_batch_size):
        """ This function builds one big Theano symbolic function which represents the entire feed-forwward behaviour
        of the network. For each layer it connects outputs to inputs, connecting the first layer to the 'x' Tensor.
        """
        self.mini_batch_size = mini_batch_size
        input_layer = self.layers[0]
        output_layer = self.layers[-1]

        # Connect the first layer to the input data
        input_layer.wire_inpt(self.x, mini_batch_size)

        # Connect each subsequent layer to the output of the previous layer.
        for l_in, l_out in zip(self.layers, self.layers[1:]):
            l_out.wire_inpt(l_in.output, mini_batch_size)
        self.output = output_layer.output

    def train(self, x_train, y_train, mini_batch_size, epochs, learning_rate):
        """ Use mini-batch gradient descent to train the network. 

        Gradient descent in a nutshell is this:
        - If you run all your training data through your network, you can get a 'cost'/error metric i.e. 
                "how wrong were we?" we want to minimize this cost.
        - If you could calculate - for each weight - approximately how much changing that weight would change the cost function (i.e. the gradient dC/dw)
                then you could move all of the weights a bit in the right direction to help reduce the error.

        And in fact, you CAN. The algorithm is called backpropagation, and Theano can do it for us in a single line of code. Despite that, a basic explanation follows:
        - Since the cost function (C) itself is differentiable e.g. ||net_out - correct_out||, and correct_out is a function of the weights i.e. a(x*w + b)
                then you can easily compute dC/dw for the last layer.
        - The previous layers don't have access to the cost so you cannot directly compute dC/dw, but you can use the 'chain rule' to compute it using dC/dw from layer in front.
        - Thus the gradient is propagated backwards through the network, each layer finding its dC/dw using the layer in front of it until all gradients are known

        Finally we need an actual training algorithm. You could run this on the entire dataset but for large datasets that becomes infeasable. Instead you can 
        do this for subsets of the data at a time - hence 'mini-batch gradient descent'. If mini_batch_size = 1, it is called stochastic gradient descent.
        """


        # First rewire the network to use the desired mini-batch size.
        self.rewire(mini_batch_size)

        num_batches = len(x_train.get_value()) / self.mini_batch_size


        # Extract the cost function: it is the last layer's cost function.
        cost = self.layers[-1].cost(self)

        # We want to get the rates of change of the cost function with respect to every single
        # weight and bias in the network. 
        # Theano can automagically compute a function which does this.
        grads = T.grad(cost, self.params)

        # We use the rates of change to update the parameters, moving each parameter in the direction
        # which helps it to minimise the cost function.
        updates = ([(param, param-learning_rate*grad) for param, grad in zip(self.params, grads)])
        
        # i is an index which specifies which mini-batch we are looking at.
        i = T.lscalar()

        # We construct the training function - it is a function of i, x and y, which computes the cost
        # given x[i] and y[i] and updates the weights. 
        # Using Theano's 'given' argument for the data apprently helps optimize GPU operation.
        train_func = theano.function([i], cost, updates=updates, 
            givens = {
            self.x: x_train[i*self.mini_batch_size: (i+1)*self.mini_batch_size], 
            self.y: y_train[i*self.mini_batch_size: (i+1)*self.mini_batch_size]
            })

        # We train over the full dataset 'epoch' times, and each epoch training on every mini-batch.
        for epoch in range(epochs):
            for batch_index in range(37):
                train_func(batch_index)

    def get_class_predictions(self, x_test):
        """ Get the class predictions for some given input """
        # For convenience we just rewrite the whole network to take all the input at once.
        self.rewire(len(x_test.get_value()))

        # the final layer's y_out gives us the class prediction for each input.
        y_out = self.layers[-1].y_out
        return theano.function([], y_out, givens={self.x: x_test})()


class FullyConnectedLayer():
    """ A FullyConnectedLayer is the standard neural network layer, connecting all inputs to all outputs. """
    def __init__(self, n_in, n_out, activation_func = sigmoid):
        """
        Construct a FullyConnectedLayer. It has
            n_out neurons (each with one output activation),
            n_in inputs per neuron (i.e. the previous layer has n_in neurons and the layers are fully connected),
            activation_function - a function to transform each node's output.
        
        Each neuron in this layer has a weight for each neuron in the previous layer, so the number of weights (w) is
        n_in * n_out, and there is also a single bias (b) for each output. 

        The layer's final outputs are also called activations and an 'activation function' (a) - e.g. a sigmoid - operates individually on each output.
        Thus the activation for a node looks like y = a(x * w + b) where w and x are vectors, x being the input activations.

        This can be extended to the entire layer and multiple input datapoints, so that Y = a(X * W + b) where X is a matrix of inputs
        and W is the weight matrix for the whole layer, and b is a vector of biases. This is how the FullyConnectedLayer stores its data: a weight
        matrix and a bias vector."""
        self.n_in = n_in
        self.n_out = n_out
        self.activation_func = activation_func
        # Initialise the weights and biases to small random variables

        # A common strategy is to divide a normal random distribution by sqrt(1/n_in), to reduce the chance that
        # w*x+b will be very large in the first training runs (which would 'saturate' the sigma function) i.e.
        # diff(sigma(large-number)) is very small and so it can't learn quickly.
        w_init = np.asarray(np.random.normal(loc=0.0, scale=np.sqrt(1.0/n_in), size=(n_in, n_out)), dtype=floatX)
        b_init = np.asarray(np.random.normal(loc=0.0, scale=1.0, size=(n_out,)), dtype=floatX)
        # Use theano 'shared' variables to store the layer's parameters
        # we use borrow = True to allow a shallow copies of w_init/b_init
        self.weights = theano.shared(w_init, name='W', borrow=True)
        self.biases = theano.shared(b_init, name='b', borrow=True)
        self.params = [self.weights, self.biases]

    def wire_inpt(self, inpt, mini_batch_size):
        """ 
        Wire up the input to this layer. 
            inpt: A Theano symbolic function or variable which will give a matrix of activations for each datapoint
            mini_batch_size specifies how many datapoints this layer will process at once.

        This function returns nothing, but sets self.output to be a Theano symbolic function which performs the layer's
        logic, i.e. a function which returns a(x*w + b)
        """
        # For the layer input, there is a row for each data sample, 
        # and element i of the row is the activation for the ith neuron for that sample
        self.inpt = inpt.reshape((mini_batch_size, self.n_in))
        
        # By taking the dot product of the input and weights, each sample row is multiplied
        # with the weight matrix resulting in n_out columns and mini_batch_size rows.
        # After this, the activation function operates on each element.
        self.output = self.activation_func(T.dot(self.inpt, self.weights) + self.biases)

        # y_out should give - for each datapoint - the index of the largest activation.
        # this is used when the layer is a final/classification layer and each activation
        # represents a different class.
        self.y_out = T.argmax(self.output, axis=1)

    def accuracy(self, y):
        """ Return the proportion of correctly classified samples """
        return T.mean(T.eq(y, self.y_out))

class SoftmaxLayer(FullyConnectedLayer):
    def __init__(self, n_in, n_out):
        super().__init__(n_in, n_out, softmax)

    def cost(self, net):
        # This needs to be used with the softmax activation function at the final layer.
        # Each column of the layer's output  will contain mostly 0, except for large values at
        # the index of the 'best guess' of that datapoint's class (this is what softmax does).
        
        # 'net.y' will contain the correct class labels for each datapoint.
        # The cost of a single datapoint is log(out[column, index_of_real_class])
        # if the net guessed correctly, out[...] will be close to 1, so log(out[...]) close to 0
        # if the net guessed wrong, out[...] will be close to 0, so log(out[...]) will be a large negative value
        # 0 - The mean of all these costs gives us the total cost. If most of our guesses were correct, the cost will be close to 0
        
        return -T.mean(T.log(self.output)[T.arange(net.y.shape[0]), net.y])


def main():
    # Create the training data:
    # classes are either 'low,high' for y=1 or 'high,low' for y=0
    N = 1000
    N_train = 750
    Y = np.random.randint(2, size=N)
    X = np.random.normal(size=(N, 2))
    X[:, 1] = X[:, 0] + np.abs(X[:, 1])*(Y*2-1)

    # Split into test/train
    x_train = theano.shared(np.asarray(X[:N_train], dtype=floatX), borrow=True)
    y_train = T.cast(theano.shared(np.asarray(Y[:N_train], dtype=floatX), borrow=True), "int32")
    x_test = theano.shared(np.asarray(X[N_train+1:], dtype=floatX), borrow=True)
    y_test = T.cast(theano.shared(np.asarray(Y[N_train+1:], dtype=floatX), borrow=True), "int32")

    # Build the network
    net = Network([
        FullyConnectedLayer(2, 3),
        SoftmaxLayer(3, 2),
        ])

    # Train it - I picked these numbers completely arbitrarily
    net.train(x_train, y_train, 2, 50, 0.1)

    # Get predictions, check the accuracy
    out = net.get_class_predictions(x_test)

    # 95+ percent! Not bad
    print("Accuracy: {:.4}%".format(np.mean(np.equal(out,y_test.eval()))*100))


if __name__ == '__main__':
    main()
