from builtins import range
from builtins import object
import numpy as np

from ..layers import *
from ..layer_utils import *


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(
        self,
        input_dim=3 * 32 * 32,
        hidden_dim=100,
        num_classes=10,
        weight_scale=1e-3,
        reg=0.0,
    ):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian centered at 0.0 with               #
        # standard deviation equal to weight_scale, and biases should be           #
        # initialized to zero. All weights and biases should be stored in the      #
        # dictionary self.params, with first layer weights                         #
        # and biases using the keys 'W1' and 'b1' and second layer                 #
        # weights and biases using the keys 'W2' and 'b2'.                         #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        W1 = np.random.normal(scale=weight_scale,size=(input_dim, hidden_dim))
        b1 = np.zeros(hidden_dim)

        W2 = np.random.normal(scale=weight_scale,size=(hidden_dim,num_classes))
        b2 = np.zeros(num_classes)

        self.params['W1'] = W1
        self.params['b1'] = b1
        self.params['W2'] = W2
        self.params['b2'] = b2

        pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        W1 = self.params['W1']
        b1 = self.params['b1']
        W2 = self.params['W2']
        b2 = self.params['b2']
        hidden_scores, hidden_cache = affine_forward(X,W1,b1)
        activation_one, relu_cache = relu_forward(hidden_scores)
        scores, fin_cache = affine_forward(activation_one,W2,b2)

        pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        loss, dscores = softmax_loss(scores, y)
        drelu, dW2, db2 = affine_backward(dscores, fin_cache)
        dhs = relu_backward(drelu, relu_cache)
        dX, dW1, db1 = affine_backward(dhs, hidden_cache)

        # Adding regularization
        loss += 0.5 * self.reg * (np.sum(W2 * W2) + np.sum(W1*W1))
        dW2 += self.reg * W2
        dW1 += self.reg * W1

        grads['W1'] = dW1
        grads['b1'] = db1
        grads['W2'] = dW2
        grads['b2'] = db2

        pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch/layer normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch/layer normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(
        self,
        hidden_dims,
        input_dim=3 * 32 * 32,
        num_classes=10,
        dropout=1,
        normalization=None,
        reg=0.0,
        weight_scale=1e-2,
        dtype=np.float32,
        seed=None,
    ):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=1 then
          the network should not use dropout at all.
        - normalization: What type of normalization the network should use. Valid values
          are "batchnorm", "layernorm", or None for no normalization (the default).
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.normalization = normalization
        self.use_dropout = dropout != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution centered at 0 with standard       #
        # deviation equal to weight_scale. Biases should be initialized to zero.   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to ones and shift     #
        # parameters should be initialized to zeros.                               #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        # Initialize weights and biases for hidden layers
        for index, size in enumerate(hidden_dims):
            if (index == 0):
                num_row = input_dim
            else:
                num_row = hidden_dims[index - 1]
            num_col = size
            weights = np.random.normal(scale=weight_scale,size=(num_row, num_col))
            biases = np.zeros(num_col)
            self.params.update({'W{number}'.format(number=index+1): weights})
            self.params.update({'b{number}'.format(number=index+1): biases})
            if self.normalization == "batchnorm" or self.normalization == "layernorm":
                gammas = np.ones(num_col)
                betas = np.zeros(num_col)
                self.params.update({'gamma{number}'.format(number=index+1): gammas})
                self.params.update({'beta{number}'.format(number=index+1): betas})

        # Initialize weights and biases for final layer
        fin_weight = np.random.normal(scale=weight_scale,size=(hidden_dims[-1], num_classes))
        fin_bias = np.zeros(num_classes)
        self.params.update({'W{number}'.format(number=self.num_layers): fin_weight})
        self.params.update({'b{number}'.format(number=self.num_layers): fin_bias})



        pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {"mode": "train", "p": dropout}
            if seed is not None:
                self.dropout_param["seed"] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.normalization == "batchnorm":
            self.bn_params = [{"mode": "train"} for i in range(self.num_layers - 1)]
        if self.normalization == "layernorm":
            self.bn_params = [{} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = "test" if y is None else "train"

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param["mode"] = mode
        if self.normalization == "batchnorm":
            for bn_param in self.bn_params:
                bn_param["mode"] = mode
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully-connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        affine_cache_list = []
        norm_cache_list = []
        relu_cache_list = []
        dropout_cache_list = []

        for i in range(self.num_layers - 1):
            if (i == 0):
                axon = X
            else:
                if self.use_dropout:
                    axon = dropout_score
                else:
                    axon = relu_score
            weight = self.params['W{number}'.format(number=i+1)]
            bias = self.params['b{number}'.format(number=i+1)]
            affine_score, affine_cache = affine_forward(axon, weight, bias)
            if self.normalization=="batchnorm":
                gamma = self.params['gamma{number}'.format(number=i+1)]
                beta = self.params['beta{number}'.format(number=i+1)]
                batchnorm_score, batchnorm_cache = batchnorm_forward(affine_score, gamma, beta, self.bn_params[i])
                # self.bn_params[i+1] = self.bn_params[i]
                relu_score, relu_cache = relu_forward(batchnorm_score)
                norm_cache_list.append(batchnorm_cache)
            elif self.normalization == "layernorm":
                gamma = self.params['gamma{number}'.format(number=i+1)]
                beta = self.params['beta{number}'.format(number=i+1)]
                layernorm_score, layernorm_cache = layernorm_forward(
                    affine_score, gamma, beta, self.bn_params[i])
                # self.bn_params[i+1] = self.bn_params[i]
                relu_score, relu_cache = relu_forward(layernorm_score)
                norm_cache_list.append(layernorm_cache)
            else:
                relu_score, relu_cache = relu_forward(affine_score)
            if self.use_dropout:
                dropout_score, dropout_cache = dropout_forward(relu_score, self.dropout_param)
                dropout_cache_list.append(dropout_cache)
            affine_cache_list.append(affine_cache)
            relu_cache_list.append(relu_cache)

        if self.use_dropout:
            fin_axon = dropout_score
        else:
            fin_axon = relu_score
        fin_W = self.params['W{number}'.format(number=self.num_layers)]
        fin_b = self.params['b{number}'.format(number=self.num_layers)]
        scores, fin_affine_cache = affine_forward(fin_axon, fin_W, fin_b)


        pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early
        if mode == "test":
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch/layer normalization, you don't need to regularize the scale   #
        # and shift parameters.                                                    #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        loss, dscores = softmax_loss(scores, y)
        last_weight = 'W{layer}'.format(layer=self.num_layers)
        last_bias = 'b{layer}'.format(layer=self.num_layers)
        loss += 0.5 * self.reg * np.sum( self.params[last_weight] * self.params[last_weight])
        # Compute gradients on lass layer
        if self.use_dropout:
            ddr, dW, db = affine_backward(dscores, fin_affine_cache)
        else:
            drelu, dW, db = affine_backward(dscores, fin_affine_cache)
        dW += self.reg * self.params[last_weight]
        grads[last_weight] = dW
        grads[last_bias] = db
        # Reverse the list for ease of keeping track of backprop
        affine_cache_list.reverse()
        relu_cache_list.reverse()
        norm_cache_list.reverse()
        dropout_cache_list.reverse()
        # Initialize an offset from the last layer
        offset = 1
        for index, ac in enumerate(affine_cache_list):
            cur_layer = self.num_layers - offset
            # print("Backprop on {i}th layer".format(i=cur_layer))
            if self.use_dropout:
                # print("\t\tBackproped Dropout Layer")
                dc = dropout_cache_list[index]
                drelu = dropout_backward(ddr, dc)
            # print("\t\tBackproped Relu Layer")
            rc = relu_cache_list[index]
            # Do backprop on relu layer
            dhs = relu_backward(drelu, rc)
            # Backprop on Batchnorm if it exists
            if self.normalization == "batchnorm":
                bc = norm_cache_list[index]
                dhs, dgamma, dbeta= batchnorm_backward_alt(dhs, bc)
                grads['gamma{l}'.format(l=cur_layer)] = dgamma
                grads['beta{l}'.format(l=cur_layer)] = dbeta
            if self.normalization == "layernorm":
                bc = norm_cache_list[index]
                dhs, dgamma, dbeta = layernorm_backward(dhs, bc)
                grads['gamma{l}'.format(l=cur_layer)] = dgamma
                grads['beta{l}'.format(l=cur_layer)] = dbeta
            # Do backprop on affine layer
            if self.use_dropout:
                # print("\t\tBackproped Affine Layer with Dropout")
                ddr, dW, db = affine_backward(dhs, ac)
            else:
                # print("\t\tBackproped Affine Layer without Dropout")
                drelu, dW, db = affine_backward(dhs, ac)
            cur_weight = 'W{layer}'.format(layer=cur_layer)
            cur_bias = 'b{layer}'.format(layer=cur_layer)

            #Add L2 regularization to the loss
            loss +=  0.5 * self.reg * np.sum(self.params[cur_weight] * self.params[cur_weight])
            dW += self.reg * self.params[cur_weight]

            #Update to grads dict
            # print('Adding grads in layer {i}th'.format(i=cur_layer))
            grads[cur_weight] = dW
            grads[cur_bias] = db

            # Increment the offset
            offset += 1
        pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
