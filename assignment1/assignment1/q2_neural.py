#!/usr/bin/env python

import numpy as np
import random

from q1_softmax import softmax
from q2_sigmoid import sigmoid, sigmoid_grad
from q2_gradcheck import gradcheck_naive


def forward_backward_prop(data, labels, params, dimensions):
    """
    Forward and backward propagation for a two-layer sigmoidal network

    Compute the forward propagation and for the cross entropy cost,
    and backward propagation for the gradients for all parameters.

    Arguments:
    data -- M x Dx matrix, where each row is a training example.
    labels -- M x Dy matrix, where each row is a one-hot vector.
    params -- Model parameters, these are unpacked for you.
    dimensions -- A tuple of input dimension, number of hidden units
                  and output dimension
    """

    ### Unpack network parameters (do not modify)
    M = data.shape[0]
    ofs = 0
    Dx, H, Dy = (dimensions[0], dimensions[1], dimensions[2])
    # print(params)
    # print(dimensions)
    # print(params.shape)
    # print("*********")
    W1 = np.reshape(params[ofs:ofs+ Dx * H], (Dx, H))
    ofs += Dx * H
    b1 = np.reshape(params[ofs:ofs + H], (1, H))
    ofs += H
    W2 = np.reshape(params[ofs:ofs + H * Dy], (H, Dy))
    ofs += H * Dy
    b2 = np.reshape(params[ofs:ofs + Dy], (1, Dy))

    ### YOUR CODE HERE: forward propagation
    # raise NotImplementedError
    # if len(data.shape)==1:
    #     data = np.reshape(data,(1,len(data)))
    #     labels = np.reshape(labels,(1,len(labels)))
    hidden_input = np.dot(data,W1)+b1
    # for row in hidden_input:
    #     row += b1[0]
    hidden_output = sigmoid(hidden_input)

    yhead_input = np.dot(hidden_output,W2)+b2
    # for row in yhead_input:
    #     row += b2[0]
    yhead_output = softmax(yhead_input)

    CE = np.sum(-labels*np.log(yhead_output))
    cost = CE
    # CE = np.zeros(len(labels))
    # for i in range(len(cal_ce)):
    #     CE[i] = np.sum(cal_ce[i])
    # cost = CE

    ### END YOUR CODE

    ### YOUR CODE HERE: backward propagation
    # raise NotImplementedError
    delta1 = yhead_output-labels
    # print(delta1.shape)
    gradW2 = np.dot(hidden_output.T,delta1)
    gradb2 = np.array([np.sum(delta1[:,i]) for i in range(delta1.shape[1])])

    delta2 = np.dot(delta1,W2.T)
    delta3 = sigmoid_grad(hidden_output)*delta2
    gradW1 = np.dot(data.T,delta3)
    gradb1 = np.array([np.sum(delta3[:,i]) for i in range(delta3.shape[1])])



    # ### YOUR CODE HERE: forward propagation
    # z1 = np.dot(data,W1) + b1
    # a1 = sigmoid(z1)
    # z2 = np.dot(a1, W2) + b2
    # a2 = softmax(z2)
    # cost = np.sum(-1 * labels * np.log(a2))
    
    # ### END YOUR CODE
    
    # ### YOUR CODE HERE: backward propagation
    
    # d2 = (a2 - labels) # dim 20x10
    # gradW2 = np.dot(a1.T, d2) # dim 5x10
    # gradb2 = np.sum(d2, axis=0) # dim 1x10
    # d1 = np.multiply(sigmoid_grad(a1), np.dot(d2, W2.T)) # dim (20x10) x (10x5) = 20x5
    
    # gradW1 = np.dot(data.T, d1) # dim 10x5
    # gradb1 = np.sum(d1, axis=0)
    # grad = np.dot(delta3,W1.T)
    ### END YOUR CODE

    ### Stack gradients (do not modify)
    grad = np.concatenate((gradW1.flatten(), gradb1.flatten(),
        gradW2.flatten(), gradb2.flatten()))

    return cost, grad


def sanity_check():
    """
    Set up fake data and parameters for the neural network, and test using
    gradcheck.
    """
    print "Running sanity check..."

    N = 20
    dimensions = [10, 5, 10]
    data = np.random.randn(N, dimensions[0])   # each row will be a datum
    labels = np.zeros((N, dimensions[2]))
    for i in xrange(N):
        labels[i, random.randint(0,dimensions[2]-1)] = 1

    params = np.random.randn((dimensions[0] + 1) * dimensions[1] + (
        dimensions[1] + 1) * dimensions[2], )

    gradcheck_naive(lambda params:
        forward_backward_prop(data, labels, params, dimensions), params)
    print("End sanity check.")


def your_sanity_checks():
    """
    Use this space add any additional sanity checks by running:
        python q2_neural.py
    This function will not be called by the autograder, nor will
    your additional tests be graded.
    """
    print "Running your sanity checks..."
    ### YOUR CODE HERE
    # raise NotImplementedError
    ### END YOUR CODE


if __name__ == "__main__":
    sanity_check()
    your_sanity_checks()
