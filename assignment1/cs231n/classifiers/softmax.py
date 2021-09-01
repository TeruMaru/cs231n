from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg, verbose=False):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    num_train = X.shape[0]
    num_classes = W.shape[1]

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    for i in range(num_train):
        scores = X[i].dot(W) #shape (1,C) or (C,)
        # Exponentiate every element of scores array
        exp_scores = np.exp(scores)
        # exp_scores[y[i]] = 0
        if i == 0 and verbose == True:
            print("Scores:", scores[:5])
            print("Exponentiated scores:", exp_scores[:5])
            print("Exponentiated score of the correct class:", exp_scores[y[i]])
        correct_class_score = scores[y[i]]
        sum_e = np.sum(exp_scores)
        loss += np.log(sum_e) - correct_class_score
        mask = (np.ones_like(W) * exp_scores) / sum_e
        if verbose == True:
            if y[i]==0:
                start = y[i]
                end = y[i] + 6
            elif y[i] == np.amax(y):
                start = y[i]-5
                end = y[i] + 1
            else:
                start = y[i]-2
                end = y[i]+3
            print("Sample %d (Correct class %d) Mask Before Replace:\n"%(i,y[i]), mask[:3,start:end])
        # mask[:,y[i]] = -1
        mask[:,y[i]] -= 1
        if verbose == True:
            print("Sample %d (Correct class %d) Mask After Replace:\n"%(i,y[i]), mask[:3,start:end])
        dW += (np.multiply(mask.T,X[i])).T

    loss /= num_train #Calculate avarage
    dW /= num_train #Scaled as loss did

    #Add generalization loss
    loss += reg * np.sum(W * W)
    dW += 2 * reg * W

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    num_train = X.shape[0]
    num_classes = W.shape[1]
    # X.shape = (N,D)
    # W.shape = (D,C)
    ## => scores.shape = (N,C)
    S = X.dot(W)
    ## exp_S.shape = (N,C)
    E = np.exp(S)
    # Store the score at the correct class column of each example row in S matrix.
    ## The correct column index, i.e. correct class, of the example at ith row is the value store in ith index of y vector
    ### scores_at_correct_class is a (N,) array whose elements are the score of each example row in X at the correct class column
    scores_at_correct_class = S[np.arange(num_train),y]

    # Calculate the sum of exponented scores of each row
    ## se.shape = (N,)
    se = np.sum(E, axis=1)

    # Take ln of se
    ls = np.log(se)

    loss_vector = ls - scores_at_correct_class

    # The ultimate loss is a scalar resulted from adding all elements of loss_vector divided by num_train
    loss = np.sum(loss_vector) / num_train

    ### Back propagation
    # dloss/dloss_vector
    dlv = np.ones_like(se) / num_train

    ## loss_vector = ls - scores_at_correct_class
    # dloss/dls = dloss/dlv * dlv/dls
    dls = dlv
    # dlos/dsc = dloss/dlv * dlv/dsc
    dsc = -dlv

    ## ls = np.log(se)
    # dloss/dse = dloss/dls * dls/dse
    se_inv = 1. / se
    dse_local = np.diag(se_inv)
    dse = dls.dot(dse_local)

    ## dloss/dE = dloss/dse * dse/dE
    # dse/dE results in a 3D array that is equivalent to a stack of N matrix with shape NxC (hence dse/dE is of shape NxNxC).
    # Every row of a sub-2D matrix is 0 except for the ith row correspond to dse_i_scalar/dE_i_vector.
    # Ultimately, dloss/dE is an NxC matrix whose values on the same ith row is equal to each other and equal the ith element of dse
    dE_trans = np.tile(dse,(num_classes,1))
    dE = dE_trans.T

    ## dloss/dS = dloss/dE * dE/dS + dloss/dsc
    dS = np.multiply(E,dE)
    dS[np.arange(num_train),y] += dsc
    # dloss/dW = dloss/dS * dS/dW
    dW = X.T.dot(dS)

    # Adding regularization term
    loss += reg * np.sum(W * W)
    dW += reg * 2 * W
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
