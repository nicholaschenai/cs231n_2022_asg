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
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1  # note delta = 1
            if margin > 0:
                loss += margin

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    dW += reg * 2 * W # regularization
    # ill just copy the code frm above here so i dont modify the above.
    # anyway later ill implement a more efficient ver
    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1  # note delta = 1
            if margin > 0:
                dW[:,j] += 1/num_train * X[i] # grad of the scores[j] term
                dW[:,y[i]] -= 1/num_train * X[i] # grad of the -correct_class_score term


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    num_train = X.shape[0]

    scores_mat = X @ W # N x C
    correct_class_scores = np.take_along_axis(scores_mat, y[..., None], axis=1)
    margin = np.maximum(0, scores_mat - correct_class_scores + 1)
    margin[np.arange(num_train), y] = 0
    loss = margin.sum() / num_train +  reg * np.sum(W * W)

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
    
    # use idx notation to get ans
    # suppose we had a 'mask' M which indicates if gradient should be included (this is just margin>0)
    # then we work out the gradient assuming no max fn, and then apply the mask
    # ignoring the reg and num_train denominator,
    # loss = margin.sum() =  ((scores_mat - correct_class_scores + 1) * M).sum()
    # = \Sigma_{ijk} X_ik * (W_kj - W_ky[i] )* M_ij (ignore the +1 as itll be differentiated away)
    # taking the derivative wrt W_lm,
    # dW_lm = \Sigma_{ijk} X_ik * (\delta_lk * \delta_mj - \delta_lk * \delta_my[i]) * M_ij
    # = \Sigma_{ij} X_il * (\delta_mj - \delta_my[i]) * M_ij # reduce across k
    # = \Sigma_{ij} X.T_li * \delta_mj * M_ij - \Sigma_{ij} X.T_li * \delta_my[i] * M_ij
    # = \Sigma_{i} X.T_li * M_im - \Sigma_{i} X.T_li * \delta_y[i]m * M_reduced_i where M_reduced is the contraction over j
    # = X.T @ M - X.T @ (M_reduced * one_hot) where one_hot represents the \delta_y[i]m term  

    M = margin>0
    num_classes = W.shape[1]
    one_hot = np.zeros((num_train, num_classes))
    one_hot[np.arange(num_train), y] = 1
    M_reduced = M.sum(axis=1).reshape((-1,1))
    
    dW += X.T @ (M - M_reduced * one_hot) / num_train
    dW += reg * 2 * W # regularization

    # cleaner soln from mantasu
    # dW = (margins > 0).astype(int)    # initial gradient with   respect to Y_hat
    # dW[range(N), y] -= dW.sum(axis=1) # update gradient to include correct labels
    # dW = X.T @ dW / N + 2 * reg * W   # gradient with respect to W
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
