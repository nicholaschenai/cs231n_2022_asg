from builtins import range
import numpy as np


def affine_forward(x, w, b):
    """Computes the forward pass for an affine (fully connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    # print('x shape',x.shape)
    # print('w shape',w.shape)
    N = x.shape[0]
    x_reshaped = x.reshape(N, -1) # reshape to (N, D)
    # print('x reshaped shape',x_reshaped.shape)
    out = x_reshaped @ w + b

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """Computes the backward pass for an affine (fully connected) layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: Biases, of shape (M,)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # see 2022 L4 slide 135 on for rough idea
    x_shape = x.shape
    N = x_shape[0]

    dx = dout @ w.T
    dx = dx.reshape(x_shape)
    
    x_reshaped = x.reshape(N, -1) # reshape to (N, D)
    dw = x_reshaped.T @ dout

    # for this, imagine the b 'matrix' is just a ones vector (N x 1) times 
    # b vector (1 x M ) i.e. it copies the rows of b
    # then using the logic for differentiating loss wrt to the second term of 
    # a prodt eg the case for w in x @ w, we get the result of ones.T @ dout
    db = np.ones(N) @ dout

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def relu_forward(x):
    """Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    out = np.maximum(0, x.copy())

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    dx = dout.copy() * (x.copy() > 0)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def softmax_loss(x, y):
    """Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    loss, dx = None, None

    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    def softmax(x_in):
      # trick for numerical stability. 
      exp_x = np.exp(x_in - np.max(x_in, axis = 1).reshape((x_in.shape[0],-1)))
      return exp_x / np.sum(exp_x, axis = 1).reshape(-1,1)

    # use take_along_axis to extract the relevant scores for each row
    softmax_x = softmax(x)

    y_reshaped = y.copy().reshape(-1,1)
    extracted_scores = np.take_along_axis(softmax_x, y_reshaped, axis=1)
    loss = np.average(-np.log(extracted_scores))
    # instead of extracting, can use trick i used below softmax_x[range(N), y]
  

    # the softmax gradient is just the softmax as usual, minus 1 at areas of ground truth
    dx = softmax_x
    delta = np.zeros_like(dx)
    delta[np.arange(dx.shape[0]).reshape(-1,1),y_reshaped] = -1
    # can just do dx[range(N), y] -= 1
    dx += delta
    dx /= y_reshaped.size # cos loss is the average of samples

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return loss, dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param["mode"]
    eps = bn_param.get("eps", 1e-5)
    momentum = bn_param.get("momentum", 0.9)

    N, D = x.shape
    running_mean = bn_param.get("running_mean", np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get("running_var", np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == "train":
        #######################################################################
        # TODO: Implement the training-time forward pass for batch norm.      #
        # Use minibatch statistics to compute the mean and variance, use      #
        # these statistics to normalize the incoming data, and scale and      #
        # shift the normalized data using gamma and beta.                     #
        #                                                                     #
        # You should store the output in the variable out. Any intermediates  #
        # that you need for the backward pass should be stored in the cache   #
        # variable.                                                           #
        #                                                                     #
        # You should also use your computed sample mean and variance together #
        # with the momentum variable to update the running mean and running   #
        # variance, storing your result in the running_mean and running_var   #
        # variables.                                                          #
        #                                                                     #
        # Note that though you should be keeping track of the running         #
        # variance, you should normalize the data based on the standard       #
        # deviation (square root of variance) instead!                        #
        # Referencing the original paper (https://arxiv.org/abs/1502.03167)   #
        # might prove to be helpful.                                          #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        sample_mean = x.mean(axis=0)
        sample_var = x.var(axis=0)
        if 'isLN' not in bn_param:
          running_mean = momentum * running_mean + (1 - momentum) * sample_mean
          running_var = momentum * running_var + (1 - momentum) * sample_var

        x_norm = (x - sample_mean) / np.sqrt(sample_var + eps)
        if len(gamma) == N:
          gammaMat = gamma.reshape((N,1)) @ np.ones((1, D))
          betaMat = beta.reshape((N,1)) @ np.ones((1, D))
        elif len(gamma) == D:
          gammaMat = np.ones((N, 1)) @ gamma.reshape((1, D))
          betaMat = np.ones((N, 1)) @ beta.reshape((1, D))
        out = gammaMat * x_norm + betaMat
        
        cache = (x, x_norm, gamma, beta, sample_mean, sample_var, eps, out, bn_param)
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == "test":
        #######################################################################
        # TODO: Implement the test-time forward pass for batch normalization. #
        # Use the running mean and variance to normalize the incoming data,   #
        # then scale and shift the normalized data using gamma and beta.      #
        # Store the result in the out variable.                               #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        x_norm = (x - running_mean) / np.sqrt(running_var + eps)
        if len(gamma) == N:
          gammaMat = gamma.reshape((N,1)) @ np.ones((1, D))
          betaMat = beta.reshape((N,1)) @ np.ones((1, D))
        elif len(gamma) == D:
          gammaMat = np.ones((N, 1)) @ gamma.reshape((1, D))
          betaMat = np.ones((N, 1)) @ beta.reshape((1, D))
        out = gammaMat * x_norm + betaMat

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param["running_mean"] = running_mean
    bn_param["running_var"] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    # Referencing the original paper (https://arxiv.org/abs/1502.03167)       #
    # might prove to be helpful.                                              #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    x, x_norm, gamma, beta, sample_mean, sample_var, eps, out, bn_param = cache
    (N, D) = x.shape

    # use the same technique as the one for affine backward. beta is copied
    # many times so it has the same shape as N, D. for normal case, the ans is
    # np.ones(shape = (1,N)) @ dout. 
    # however to future proof for layer norm, we see that in general, beta is really
    # np.ones((N, 1)) @ beta_vec and in the layer norm case, these 2 vectors
    # are swapped. the ans for the normal case can also be seen as
    # dout.sum(axis=0) and for layer norm, its dout @ np.ones(shape = (D,1))
    # or dout.sum(axis=1). so we use the bn_param flag to signify which axis 
    # to sum
    myAxis = int('isLN' in bn_param)
    dbeta = dout.sum(axis=myAxis)

    # actually, just use index notation and we can see its (x_norm*dout).sum(axis=0)
    dgamma = (x_norm*dout).sum(axis=myAxis)

    # similar to above, we have x_hat @ Diag(gamma) and we want to get loss wrt
    # x_hat, so we just use formula for differentiating first term in a product
    # which is dout @ second term transpose. but Diag(gamma) is symmetric
    # which is effectively multiplying dout with gamma row wise
    # or just use index notation
    if len(gamma) == N:
      gammaMat = gamma.reshape((N,1)) @ np.ones((1, D))
    elif len(gamma) == D:
      gammaMat = np.ones((N, 1)) @ gamma.reshape((1, D))
    dx_hat = dout * gammaMat

    # now with dx_hat, we want to find dx. see paper for expressions after taking  
    # derivatives wrt sigma^2, mu and x. we directly implement the expressions here
    d_sigma_sq = (dx_hat * (x-sample_mean)).sum(axis=0) * (-1/2) * (sample_var + eps) ** (-3/2)
    d_mu = (dx_hat * -1/np.sqrt(sample_var + eps)).sum(axis=0) + d_sigma_sq * (-2/N) * (x - sample_mean).sum(axis=0)
    dx = dx_hat * 1/np.sqrt(sample_var + eps) + d_sigma_sq * 2/N * (x-sample_mean) + d_mu/N
    # Note: see others soln for step by step
    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass.
    See the jupyter notebook for more hints.

    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    #                                                                         #
    # After computing the gradient with respect to the centered inputs, you   #
    # should be able to compute gradients with respect to the inputs in a     #
    # single statement; our implementation fits on a single 80-character line.#
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    x, x_norm, gamma, beta, sample_mean, sample_var, eps, out, bn_param = cache
    (N, D) = x.shape

    myAxis = int('isLN' in bn_param)
    dbeta = dout.sum(axis=myAxis)
    dgamma = (x_norm*dout).sum(axis=myAxis)
    if len(gamma) == N:
      gammaMat = gamma.reshape((N,1)) @ np.ones((1, D))
    elif len(gamma) == D:
      gammaMat = np.ones((N, 1)) @ gamma.reshape((1, D))
    dx_hat = dout * gammaMat

    # not sure if this is what the answer wants. its just my prev ans but i removed 
    # the 2nd term in d_mu cos it equates to zero. also rearrange the (x_i - mu_b)
    # to get x_hat
    sq = np.sqrt(sample_var + eps)
    dx = dx_hat/sq - 1/(N*sq) * dx_hat.sum(axis=0) - x_norm/(N*sq) * (dx_hat*x_norm).sum(axis=0)
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def layernorm_forward(x, gamma, beta, ln_param):
    """Forward pass for layer normalization.

    During both training and test-time, the incoming data is normalized per data-point,
    before being scaled by gamma and beta parameters identical to that of batch normalization.

    Note that in contrast to batch normalization, the behavior during train and test-time for
    layer normalization are identical, and we do not need to keep track of running averages
    of any sort.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - ln_param: Dictionary with the following keys:
        - eps: Constant for numeric stability

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    out, cache = None, None
    eps = ln_param.get("eps", 1e-5)
    ###########################################################################
    # TODO: Implement the training-time forward pass for layer norm.          #
    # Normalize the incoming data, and scale and  shift the normalized data   #
    #  using gamma and beta.                                                  #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of  batch normalization, and inserting a line or two of  #
    # well-placed code. In particular, can you think of any matrix            #
    # transformations you could perform, that would enable you to copy over   #
    # the batch norm code and leave it almost unchanged?                      #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    ln_param['isLN'] = True
    ln_param['mode'] = 'train'
    out, cache = batchnorm_forward(x.T, gamma, beta, ln_param)
    out = out.T

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def layernorm_backward(dout, cache):
    """Backward pass for layer normalization.

    For this implementation, you can heavily rely on the work you've done already
    for batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from layernorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for layer norm.                       #
    #                                                                         #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of batch normalization. The hints to the forward pass    #
    # still apply!                                                            #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    dx, dgamma, dbeta = batchnorm_backward(dout.T, cache)
    dx = dx.T
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """Forward pass for inverted dropout.

    Note that this is different from the vanilla version of dropout.
    Here, p is the probability of keeping a neuron output, as opposed to
    the probability of dropping a neuron output.
    See http://cs231n.github.io/neural-networks-2/#reg for more details.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We keep each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.
    """
    p, mode = dropout_param["p"], dropout_param["mode"]
    if "seed" in dropout_param:
        np.random.seed(dropout_param["seed"])

    mask = None
    out = None

    if mode == "train":
        #######################################################################
        # TODO: Implement training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                        #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        mask = (np.random.rand(*x.shape) < p) / p
        out = x*mask

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == "test":
        #######################################################################
        # TODO: Implement the test phase forward pass for inverted dropout.   #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        out = x

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """Backward pass for inverted dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param["mode"]

    dx = None
    if mode == "train":
        #######################################################################
        # TODO: Implement training phase backward pass for inverted dropout   #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        dx = dout*mask

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    elif mode == "test":
        dx = dout
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width WW.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.

    During padding, 'pad' zeros should be placed symmetrically (i.e equally on both sides)
    along the height and width axes of the input. Be careful not to modfiy the original
    input x directly.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    x_pad = x
    pad = conv_param['pad']
    x_pad = np.pad(x_pad, ((0,0),(0,0),(pad,pad),(pad,pad)))
    # print('x_pad.shape', x_pad.shape)
    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    stride = conv_param['stride']
    h_prime = 1 + (H + 2 * pad - HH) // stride
    w_prime = 1 + (W + 2 * pad - WW) // stride
    out = np.zeros((N, F, h_prime, w_prime))

    ## long winded way
    # for n_idx in range(N):
    #   for f_idx in range(F):
    #     for i in range(0, h_prime):
    #       ii = i*stride
    #       # print('ii',ii)
    #       for j in range(0, w_prime):
    #         jj = j*stride
    #         # print('jj',jj)
    #         x_patch = x_pad[n_idx,:,ii:ii+HH,jj:jj+WW]
    #         out[n_idx,f_idx, i, j] = (x_patch * w[f_idx]).sum()
    
    for i in range(h_prime):
      for j in range(w_prime):
        x_patch = x_pad[:, :, i*stride:i*stride+HH, j*stride:j*stride+WW]
        out[:,:,i,j] = np.einsum('NChw,FChw->NF', x_patch, w)

    out+=b.reshape((1,-1,1,1))

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    (x, w, b, conv_param) = cache
    db = dout.sum(axis=(0,2,3))
    dw = np.zeros_like(w)
    pad = conv_param['pad']
    stride = conv_param['stride']
    x_pad = np.pad(x, ((0,0),(0,0),(pad,pad),(pad,pad)))
    dx_pad = np.zeros_like(x_pad)

    # xpad has dims N, C, H+2*pad, W+2*pad
    # N, C, H, W = x.shape
    F, C, L, M = dw.shape
    N, _, I, J = dout.shape

    ## long winded way
    # we get regular intervals of x_pad_rearr that starts from ll/mm, and denote that in the
    # extra dims so we can use matrix product
    # x_pad_extended = np.zeros((N,C,I,J,L,M))
    # for ll in range(L):
    #   for mm in range(M):
    #     x_pad_extended[:,:,:,:,ll,mm] = x_pad[:,:,ll:ll+I*stride:stride,mm:mm+J*stride:stride]
    # dw = np.einsum('nfij,ncijlm->fclm',dout, x_pad_extended)
    # for ii in range(I):
    #   for jj in range(J):
    #     for ll in range(L):
    #       for mm in range(M):
    #         dx_pad[:,:,ii*stride+ll,jj*stride+mm] += dout[:,:,ii,jj] @ w[:,:,ll,mm]

    for i in range(I):
      for j in range(J):
        dw += np.einsum('NF,NCLM->FCLM', dout[:,:,i,j], x_pad[:, :, i*stride:i*stride+L, j*stride:j*stride+M])
        dx_pad[:, :, i*stride:i*stride+L, j*stride:j*stride+M] += np.einsum('NF,FCLM->NCLM', dout[:,:,i,j], w)

    # alternate way: if we compress CLM dimensions into 1, becomes usual matrix multiplication
    # dout[:,:,i,j] has shape NF, patches have shape ND, w has shape FD
    # dout[:,:,i,j] = x_patch @ w.T
    # so usuing the usual matrix stuff, dx_patch = dout @ w, dw.T = x_patch.T @ dout[::ij] or dw = dout[::ij].T @ x_patch 

    dx = dx_pad[:,:,pad:-pad, pad:-pad]
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """A naive implementation of the forward pass for a max-pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    No padding is necessary here, eg you can assume:
      - (H - pool_height) % stride == 0
      - (W - pool_width) % stride == 0

    Returns a tuple of:
    - out: Output data, of shape (N, C, H', W') where H' and W' are given by
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the max-pooling forward pass                            #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    N, C, H, W = x.shape
    pool_height = pool_param['pool_height']
    pool_width = pool_param['pool_width']
    stride = pool_param['stride']
    h_prime = 1 + (H - pool_height) // stride
    w_prime = 1 + (W - pool_width) // stride
    out = np.zeros((N,C,h_prime,w_prime))
    for hh in range(h_prime):
      for ww in range(w_prime):
        x_patch = x[:,:, hh*stride:hh*stride+pool_height, ww*stride:ww*stride+pool_width]
        out[:,:,hh,ww] = np.amax(x_patch,axis = (2,3))

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """A naive implementation of the backward pass for a max-pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    ###########################################################################
    # TODO: Implement the max-pooling backward pass                           #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    x, pool_param = cache
    N, C, h_prime, w_prime = dout.shape
    N, C, H, W = x.shape
    pool_height = pool_param['pool_height']
    pool_width = pool_param['pool_width']
    stride = pool_param['stride']
    dx = np.zeros_like(x)
    for nn in range(N):
      for cc in range(C):
        for hh in range(h_prime):
          for ww in range(w_prime):
            x_patch = x[nn,cc, hh*stride:hh*stride+pool_height, ww*stride:ww*stride+pool_width]
            h_idx, w_idx = np.unravel_index(np.argmax(x_patch, axis=None), x_patch.shape)
            dx[nn,cc,hh*stride + h_idx, ww*stride + w_idx] = dout[nn,cc,hh,ww]

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None

    ###########################################################################
    # TODO: Implement the forward pass for spatial batch normalization.       #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    N, C, H, W = x.shape
    xReshape = np.moveaxis(x,1,-1)
    xReshape = xReshape.reshape(-1,C)

    bOut, bCache = batchnorm_forward(xReshape, gamma, beta, bn_param)
    bOut = bOut.reshape(N, H, W, C)
    out = np.moveaxis(bOut,-1,1)
    cache = bCache
    # cache = (x, x_norm, gamma, beta, sample_mean, sample_var, eps, out)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial batch normalization.      #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    N, C, H, W = dout.shape
    doutReshape = np.moveaxis(dout,1,-1)
    doutReshape = doutReshape.reshape(-1,C)

    dxB, dgamma, dbeta = batchnorm_backward(doutReshape, cache)
    dxB = dxB.reshape(N, H, W, C)
    dx = np.moveaxis(dxB,-1,1)
    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def spatial_groupnorm_forward(x, gamma, beta, G, gn_param):
    """Computes the forward pass for spatial group normalization.
    
    In contrast to layer normalization, group normalization splits each entry in the data into G
    contiguous pieces, which it then normalizes independently. Per-feature shifting and scaling
    are then applied to the data, in a manner identical to that of batch normalization and layer
    normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (1, C, 1, 1)
    - beta: Shift parameter, of shape (1, C, 1, 1)
    - G: Integer mumber of groups to split into, should be a divisor of C
    - gn_param: Dictionary with the following keys:
      - eps: Constant for numeric stability

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None
    eps = gn_param.get("eps", 1e-5)
    ###########################################################################
    # TODO: Implement the forward pass for spatial group normalization.       #
    # This will be extremely similar to the layer norm implementation.        #
    # In particular, think about how you could transform the matrix so that   #
    # the bulk of the code is similar to both train-time batch normalization  #
    # and layer normalization!                                                #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    # NOTE: see soln for more elegant ones.
    # key idea is layer norm averages over D and preserves N for (N D) tensor
    # so if we transform (N C H W) tensor into (N*G C/G*H*W), can insert
    # directly into layernorm function without loops
    N, C, H, W = x.shape
    xReshape = np.moveaxis(x,1,-1)
    xReshape = xReshape.reshape(-1,C)

    gamma = np.squeeze(gamma)
    beta = np.squeeze(beta)

    cache = []
    lnStack = []
    for i in range(G):
      lnOut, lnCache = layernorm_forward(xReshape[:,i*C//N:(i+1)*C//N], 
      gamma[i*C//N:(i+1)*C//N], beta[i*C//N:(i+1)*C//N], gn_param)
      cache.append(lnCache)
      lnStack.append(lnOut)

    out = np.concatenate(lnStack, axis=1)
    out = out.reshape(N, H, W, C)
    out = np.moveaxis(out,-1,1)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def spatial_groupnorm_backward(dout, cache):
    """Computes the backward pass for spatial group normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (1, C, 1, 1)
    - dbeta: Gradient with respect to shift parameter, of shape (1, C, 1, 1)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial group normalization.      #
    # This will be extremely similar to the layer norm implementation.        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    N, C, H, W = dout.shape
    doutReshape = np.moveaxis(dout,1,-1)
    doutReshape = doutReshape.reshape(-1,C)

    dxStack = []
    dgammaStack = []
    dbetaStack = []
    for i, subcache in enumerate(cache):
      dxL, dgammaL, dbetaL = layernorm_backward(doutReshape[:,i*C//N:(i+1)*C//N], subcache)
      dxStack.append(dxL)
      dgammaStack.append(dgammaL)
      dbetaStack.append(dbetaL)


    dx = np.moveaxis(np.concatenate(dxStack, axis=1).reshape(N, H, W, C),-1,1)
    dgamma = np.concatenate(dgammaStack).reshape(1,C,1,1)
    dbeta = np.concatenate(dbetaStack).reshape(1,C,1,1)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta
