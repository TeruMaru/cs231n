from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c <= C-1.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather than first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # initialize the gradient as zero
    dW = np.zeros(W.shape)
    # Collect the number of classes in the training sets, i.e. C
    num_classes = W.shape[1]
    # Collect the number of training examples, i.e. N
    num_train = X.shape[0]
    # Initialize the loss to be 0 (float)
    loss = 0.0
    for i in range(num_train):
        # Compute the score of current weight when applied to example ith
        ## X[i].shape = (1,D) or (D,)
        ## W.shape = (D,C)
        ### => score_i.shape = (1,C) or (C,)
        ### => Each element at jth index in score_i array represents the score of jth class
        score_i = X[i].dot(W)

        # For the ith example, correct class score should be the score at index y[i]
        correct_class_score = score_i[y[i]]
        #Initialize a gradient matrix of zeros having the same shape as W, i.e. shape of (D,C):
        grad_i = np.zeros(W.shape)
        num_fail = 0
        for j in range(num_classes):
            if j == y[i]:
                continue
            else:
                margin = score_i[j] - correct_class_score + 1 # Note: delta = 1
            if margin > 0:
                loss += margin
                num_fail += 1
                grad_i[:,j] += X[i]


        grad_i[:,y[i]] -= num_fail * X[i]
        dW += grad_i


    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead, hence we divide by num_train.
    loss /= num_train
    dW /= num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)
    dW += reg * 2 * W

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW



def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape) # initialize the gradient as zero

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    # num_train = N
    num_train = X.shape[0]
    # Compute the score of every examples against current weight matrix W
    ## X.shape = (N,D)
    ## W.shape = (D,C)
    ## => S.shape = (N,C), i.e. S[i,j] represents the score of example i versus class j
    S = X.dot(W)
    # Identify correct classes scores and store in C array of shape (N,), i.e.
    # 1 row and N columns. To understand the command used, refer to
    # https://cs231n.github.io/python-numpy-tutorial/#arrays, "mutating elements
    # of a Matrix row" section
    C = S[np.arange(num_train),y]
    # Calculate [M]argin condition matrix before cutting off negative values,i.e.
    # M = S_wrong - S_correct + delta. (delta = 1). The matrices have been transposed
    # to cope with each other's dimensions and produce M sharing the same shape with
    # S, which is (N x C)
    M = (S.T - C + 1).T

    # Use boolean indexing to create a rank 1 array of values from M that satisfy
    # M[i,j] > 0. Note that this solution will have correct classes scores accumulated
    # to the overall loss (each correct class score of a sample will accumulate to the
    # overall loss by 1, hence num_train unit in total). Not setting the condition
    # to be > 1 since there might be similar scores between incorrect classes on correct
    # ones, thus it is safer (my opinion) to sum all and substract by num_train
    loss_contributed_scores = M[M > 0]
    ovr_loss = np.sum(loss_contributed_scores)

    # Calculate the number of classes accumulate to the lost, i.e. failed the margin
    # condition: if M[i,j] > 0 => num_lost[i]+=1 => sum by columns (axis=1).
    # The result's shape would be (N,), where each ith index represents the total
    # number of loss-contributed classes for ith sample (ith row in X)
    # Note that each row of M (represents a sample) contains 1 correct class (whose score = 1),
    # hence the results vector must be substracted by 1 at each colum
    num_contributed = np.sum(M > 0, axis=1) - 1

    # We want to relocate this array in to a mapping matrix, say U to immitate Greek letter for M,
    # of shape N x C, i.e. same shape as S and M, where each index is the scaling factor
    # of sample Xi with regard to class Cj
    # 1/ If sample Xi's score with regard to class Cj is below 0 (<0), U[i,j] = 0
    # 2/ If the score is more than > 0 but not at the correct class Cy[i], U[i,j] = 1
    # 3/ If the score is at the correct class, U[i,j] = - num_contributed[i]
    # This mapping will be dotted with X matrix to become dW
    U = np.zeros(M.shape)
    U[ M > 0] = 1 # Satisfied 1 and 2
#     print(M,"\n",U)
    U[np.arange(num_train),y] = - num_contributed.T #Tranpose a 1D array will take no effect, but it states that we understand the dimension matching. Note that this is a replace opreation
#     print(y,"\n",U)

    # Now U is of shape N x C, we want our ouput dW to be of shape D x C. Therefore, X must be transposed
    dW = X.T.dot(U)



    ## Equivalent to loss = (ovr_loss - num_train) / num_train
    loss = ovr_loss/num_train - 1
    dW /= num_train
    # Add regularization
    loss += reg * np.sum(W * W)
    dW += reg * 2 * W

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
