import numpy as np
    
######################################
#  Feedforward Neural network (FNN)
######################################

class FNN(object):
    def __init__(self, input_dim, output_dim, sizes, activ_funcs):
        """Feedforward Neural network for multi-class classification.

        The object holds a list of layer objects, each one
        implements a layer in the network, the specification
        of each layer is decided by input_dim, output_dim,
        sizes and activ_funcs. Note that an output layer
        (linear) and loss function (softmax and
        cross-entropy) would be automatically added.

        Input: 
          input_dim: dimension of input.
          output_dim: dimension of output (number of labels).
          sizes: a list of integers specifying the number of
            hidden units on each layer.
          activ_funcs: a list of function objects specifying
            the activation function of each layer.
        """
        # Last layer is linear and loss is mean_cross_entropy_softmax
        self.sizes = [input_dim] + sizes[:] + [output_dim]
        self.activ_funcs = activ_funcs[:] + [linear]
        self.shapes = []
        for i in xrange(len(self.sizes)-1):
            self.shapes.append((self.sizes[i], self.sizes[i+1]))

        self.layers = []
        for i, shape in enumerate(self.shapes):
            self.layers.append(Layer(shape, self.activ_funcs[i]))
            # print "i, shape: ", i, shape, self.activ_funcs[i]

    def forwardprop(self, data, labels=None):
        """Forward propagate the activations through the network.

        Iteratively propagate the activations (starting from
        input data) through each layer, and output a
        probability distribution among labels (probs), and
        if labels are given, also compute the loss. 
        """
        inputs = data
        for layer in self.layers:
            outputs = layer.forward(inputs)
            inputs = outputs
        probs = softmax(outputs)
        if labels is not None:
            return probs, self.loss(outputs, labels)
        else:
            return probs, None

    def backprop(self, labels):
        """Backward propagate the gradients through the network.

        Iteratively propagate the gradients (starting from
        outputs) through each layer, and save gradients of
        the loss w.r.t each parameter (weights or bias) in
        each layer.
        """
        d_outputs = self.d_loss(self.layers[-1].a, labels)
        for layer in self.layers[::-1]:
            d_inputs = layer.backward(d_outputs)
            d_outputs = d_inputs

    def loss(self, outputs, labels):
        "Compute the cross entropy softmax loss."
        return mean_cross_entropy_softmax(outputs, labels)

    def d_loss(self, outputs, labels):
        """Compute gradients of the cross entropy softmax
        loss w.r.t the outputs.
        """
        return d_mean_cross_entropy_softmax(outputs, labels)
        
    def predict(self, data):
        "Predict the labels of the data."
        probs, _ = self.forwardprop(data)
        return np.argmax(probs, axis=1)


class Layer(object):
    def __init__(self, shape, activ_func):
        "Implements a layer of a NN."
      
        # Initialize the weights and bias.
        self.w = np.random.uniform(-np.sqrt(2.0 / shape[0]),
                                   np.sqrt(2.0 / shape[0]),
                                   size=shape)
        self.b = np.zeros((1, shape[1]))

        # The activation function. For example, relu and
        # tanh. A function object that can be used like a
        # function.
        self.activate = activ_func

        # The gradient of the activation function. For
        # example, d_relu and d_tanh. Also a function object
        # that can be used like a function.
        self.d_activate = GRAD_DICT[activ_func]

    def forward(self, inputs):
        """Forward propagate the activation through the layer.
        
        Given the inputs (activation of previous layers),
        compute and save the activation of current layer,
        then return it as output.
        """

        ###################################
        # Question 1

        # Instructions

        # Use the linear and non-linear transformation to
        # compute the activation and cache it in a the field, self.a.

        # Functions you would use:
        # np.dot: numpy function to compute dot product of two matrix.
        # self.activate: the activation function of this layer,
        #                it takes in a matrix of scores (linear transformation)
        #                and compute the activations (non-linear transformation).
        # (plus the common arithmetic functions).

        # For all the numpy functions, use google and numpy manual for
        # more details and examples. 
        # Learn about numpy broadcasting at
        # http://docs.scipy.org/doc/numpy-1.10.1/user/basics.broadcasting.html#general-broadcasting-rules.
        # It will simplify the code you need to write.
        
        # Object fields you would use:
        # self.w:
        #     weight matrix, a matrix with shape (H_-1, H).
        #     H_-1 is the number of hidden units in previous layer
        #     H is the number of hidden units in this layer
        # self.b: bias, a matrix/vector with shape (1, H).
        # self.activate: the activation function of this layer.

        # Input:
        # inputs:
        #    a matrix with shape (N, H_-1),
        #    N is the number of data points in this batch.
        #    H_-1 is the number of hidden units in previous layer

        # Code you need to fill in: 2 lines.
        #########################################################
        # Modify the right hand side of the following code.
        
        # The linear transformation.
        # scores:
        #     weighted sum of inputs plus bias, a matrix of shape (N, H).
        #     N is the number of data points in this batch.
        #     H is the number of hidden units in this layer.
        scores = np.dot(inputs, self.w) + self.b

        # The non-linear transformation.
        # outputs:
        #     activations of this layer, a matrix of shape (N, H).
        #     N is the number of data points in this batch.
        #     H is the number of hidden units in this layer.
        activations = self.activate(scores)

        # End of the code to modify
        #########################################################

        # Cache the inputs, will be used by backforward
        # function during backprop.
        self.inputs = inputs

        # Cache the activations, to be used by backprop.
        self.a = activations
        outputs = activations
        return outputs

    def backward(self, d_outputs):
        """Backward propagate the gradients through the layer.
        
        Given the gradients of the loss w.r.t the output
        (same as the "deltas" in the backpropagation
        slides), compute gradients of the loss w.r.t the
        parameters (weights and bias) in this layer and
        return the gradients of the loss w.r.t the output
        of previous layer.
        """
        ###################################
        # Question 2
        
        # Instructions

        # Compute the gradients of the loss w.r.t the
        # weights and bias given the gradients of the loss
        # w.r.t outputs of this layer using chain rule.

        # Naming convention: use d_var to hold the
        # gradient of loss w.r.t the variable var, for
        # example, self.d_w holds the gradient of self.w.

        # Inputs:
        # d_outputs:
        #     gradients of the loss w.r.t the output/activation
        #     of this layer, a matrix of shape (N, H).
        #     N is the number of data points in this batch.
        #     H is the number of hidden units in current layer. 

        # Functions you would use:
        # np.dot (numpy.dot): numpy function to compute dot product of two matrix.
        # np.sum (numpy.sum):
        #     numpy function to compute the sum of a matrix,
        #     use keywords argument 'axis' to compute the
        #     sum along a particular axis, you might 
        #     find 'keepdims' argument useful.
        # self.d_activate:
        #     given the current activation (self.a) as input,
        #     compute gradient of the activation function,
        #     See d_relu as an example.
        # (plus the common arithmetic functions).
        # np.transpose or m.T (m is an numpy array): transpose a matrix.
        
        # For all the numpy functions, use google and numpy
        # manual for more details and examples. 
        # Learn about numpy broadcasting at
        # http://docs.scipy.org/doc/numpy-1.10.1/user/basics.broadcasting.html#general-broadcasting-rules.
        # It will simplify the code you need to write.

        # Object fields you would use:
        # self.w: weight matrix, a matrix with shape (H_-1, H).
        #         H_-1 is the number of hidden units in previous layer
        #         H is the number of hidden units in this layer
        # self.d_activate: compute gradient of the activation function.
        #                  This is a field holding a function object,
        #                  See d_relu as an example of the value. 
        # self.inputs: cached inputs of current layer (outputs/activations of
        #              the previous layer), a matrix of shape (N, H_-1).
        #              See forward method.

        # Code you need to write: 4 lines.
        ###################################
        # Modify the right hand side of the following code.

        # d_scores:
        #     Gradients of the loss w.r.t the scores (the result from
        #     linear transformation).
        #     A matrix of shape (N, H)
        #     N is the number of data points in this batch.
        #     H is the number of hidden units in this layer.
        d_scores = d_outputs * self.d_activate(self.a)

        # self.d_b:
        #     Gradient of the loss w.r.t the bias
        #     A matrix of shape (1, H)
        #     H is the number of hidden units in this layer.
        self.d_b = np.sum(d_scores, axis = 0, keepdims = True)

        # self.d_w:
        #     Gradient of the loss w.r.t weight matrix 
        #     A matrix of shape (H_-1, H)
        #     H_-1 is the number of hidden units in previous layer
        #     H is the number of hidden units in this layer.        
        self.d_w = np.dot(np.transpose(self.inputs),d_scores)

        # d_inputs:
        #     Gradients of the loss w.r.t the previous layer's activations/outputs.
        #     A matrix of shape (N, H_-1)
        #     N is the number of data points in this batch.
        #     H_-1 is the number of hidden units in the previous layer.
        d_inputs = np.dot(d_scores, np.transpose(self.w))

        # End of the code to modify
        ###################################

        # Compute the average value of the gradients, since
        # we are minimizing the average loss. 
        self.d_b /= d_scores.shape[0]
        self.d_w /= d_scores.shape[0]
        
        return d_inputs


class GradientDescentOptimizer(object):
    def __init__(self, learning_rate, decay_steps=1000,
                 decay_rate=1.0):
        "Gradient descent with staircase exponential decay."
        self.learning_rate = learning_rate
        self.steps = 0.0
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        
    def update(self, model):
        "Update model parameters."
        for layer in model.layers:
            layer.w -= layer.d_w * self.learning_rate
            layer.b -= layer.d_b * self.learning_rate
        self.steps += 1
        if (self.steps + 1) % self.decay_steps == 0:
            self.learning_rate *= self.decay_rate


# Utility functions.
def relu(x):
    "The rectified linear unit (RELU) activation function."
    return np.clip(x, 0.0, None)


def d_relu(a=None, x=None):
    "Compute the gradient of RELU given current activation (a) or input (x)."
    if a is not None:    
        d = np.zeros_like(a)
        d[np.where(a > 0.0)] = 1.0
        return d
    else:
        return d_relu(a=relu(x))


def tanh(x):
    "The tanh activation function."
    return np.tanh(x)


def d_tanh(a=None, x=None):
    "Compute the gradient of tanh given current activation (a) or input (x)."
    if a is not None:
        return 1 - a ** 2
    else:
        return d_tanh(a=tanh(x))


def softmax(x):
    shifted_x = x - np.max(x, axis=1, keepdims=True)
    f = np.exp(shifted_x)
    p = f / np.sum(f, axis=1, keepdims=True)
    return p
    

def mean_cross_entropy(outputs, labels):
    "Average cross entroy loss between outputs and labels."    
    n = labels.shape[0]
    return - np.sum(labels * np.log(outputs)) / n


def mean_cross_entropy_softmax(logits, labels):
    "Combine softmax and cross entropy loss."
    return mean_cross_entropy(softmax(logits), labels)


def d_mean_cross_entropy_softmax(logits, labels):
    "Comptue the gradient of the combined softmax and cross entropy loss."
    return softmax(logits) - labels


def linear(x):
    """Activation function for a linear layer.

    Although conceptually unnecessay (linear layer doesn't
    have non-linear transformantion), this is added so that
    the same forward and backwar method can be reused for
    linear layer.
    """
    return x


def d_linear(a=None, x=None):
    """Compute the gradient of the activation function for
    linear layer given current activation (a) or input
    (x).
    """
    if a is not None:    
        return np.ones_like(a, dtype=np.float64)
    else:
        return np.ones_like(x, dtype=np.float64)


# Mapping from activation functions to its gradients.
GRAD_DICT = {relu: d_relu, tanh: d_tanh, linear: d_linear}
