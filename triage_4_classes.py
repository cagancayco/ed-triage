#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  4 10:50:06 2018

@author: cag3fr
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 14:12:36 2018

@author: cag3fr
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import math
import tensorflow as tf
from tensorflow.python.framework import ops

#%% TensorFlow functions
# Function for encoding label vector as one hot matrix
def convert_to_one_hot(labels, C):
    C = tf.constant(C, name="C")
    one_hot_matrix = tf.one_hot(labels, C, axis = 0)
    with tf.Session() as sess:
        one_hot = sess.run(one_hot_matrix)
        sess.close()
    return one_hot


# Function for selecting random minibatches
def random_mini_batches(X, Y, mini_batch_size = 64):
    m = X.shape[1]                  # number of training examples
    mini_batches = []   
    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((Y.shape[0],m))
    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size : m]
        mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size : m]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)    
    return mini_batches

# Creating placeholders for inputs
def create_placeholders(n_x, n_y):
    X = tf.placeholder(shape = [n_x, None], dtype = "float")
    Y = tf.placeholder(shape = [n_y, None], dtype = "float")   
    return X, Y

# Initialize parameters
def initialize_parameters():
    W1 = tf.get_variable("W1", [16, 40], initializer = tf.contrib.layers.xavier_initializer())
    b1 = tf.get_variable("b1", [16, 1], initializer = tf.zeros_initializer())
    W2 = tf.get_variable("W2", [16, 16], initializer = tf.contrib.layers.xavier_initializer())
    b2 = tf.get_variable("b2", [16, 1], initializer = tf.zeros_initializer())
    W3 = tf.get_variable("W3", [16, 16], initializer = tf.contrib.layers.xavier_initializer())
    b3 = tf.get_variable("b3", [16, 1], initializer = tf.zeros_initializer())
    W4 = tf.get_variable("W4", [12, 16], initializer = tf.contrib.layers.xavier_initializer())
    b4 = tf.get_variable("b4", [12, 1], initializer = tf.zeros_initializer()) 
    W5 = tf.get_variable("W5", [12, 12], initializer = tf.contrib.layers.xavier_initializer())
    b5 = tf.get_variable("b5", [12, 1], initializer = tf.zeros_initializer()) 
    W6 = tf.get_variable("W6", [12, 12], initializer = tf.contrib.layers.xavier_initializer())
    b6 = tf.get_variable("b6", [12, 1], initializer = tf.zeros_initializer())
    W7 = tf.get_variable("W7", [8, 12], initializer = tf.contrib.layers.xavier_initializer())
    b7 = tf.get_variable("b7", [8, 1], initializer = tf.zeros_initializer()) 
    W8 = tf.get_variable("W8", [8, 8], initializer = tf.contrib.layers.xavier_initializer())
    b8 = tf.get_variable("b8", [8, 1], initializer = tf.zeros_initializer()) 
    W9 = tf.get_variable("W9", [8, 8], initializer = tf.contrib.layers.xavier_initializer())
    b9 = tf.get_variable("b9", [8, 1], initializer = tf.zeros_initializer())
    W10 = tf.get_variable("W10", [4, 8], initializer = tf.contrib.layers.xavier_initializer())
    b10 = tf.get_variable("b10", [4, 1], initializer = tf.zeros_initializer())
    W11 = tf.get_variable("W11", [4, 4], initializer = tf.contrib.layers.xavier_initializer())
    b11 = tf.get_variable("b11", [4, 1], initializer = tf.zeros_initializer())
    W12 = tf.get_variable("W12", [4, 4], initializer = tf.contrib.layers.xavier_initializer())
    b12 = tf.get_variable("b12", [4, 1], initializer = tf.zeros_initializer())
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3,
                  "W4": W4,
                  "b4": b4,
                  "W5": W5,
                  "b5": b5,
                  "W6": W6,
                  "b6": b6,
                  "W7": W7,
                  "b7": b7,
                  "W8": W8,
                  "b8": b8,
                  "W9": W9,
                  "b9": b9,
                  "W10": W10,
                  "b10": b10,
                  "W11": W11,
                  "b11": b11,
                  "W12": W12,
                  "b12": b12
                  }
    return parameters

# Forward prop
def forward_propagation(X, parameters):
    W1 = parameters["W1"]; b1 = parameters["b1"]
    W2 = parameters["W2"]; b2 = parameters["b2"]
    W3 = parameters["W3"]; b3 = parameters["b3"]
    W4 = parameters["W4"]; b4 = parameters["b4"]
    W5 = parameters["W5"]; b5 = parameters["b5"]
    W6 = parameters["W6"]; b6 = parameters["b6"]
    W7 = parameters["W7"]; b7 = parameters["b7"]
    W8 = parameters["W8"]; b8 = parameters["b8"]
    W9 = parameters["W9"]; b9 = parameters["b9"]
    W10 = parameters["W10"]; b10 = parameters["b10"]
    W11 = parameters["W11"]; b11 = parameters["b11"]
    W12 = parameters["W12"]; b12 = parameters["b12"]

    Z1 = tf.add(tf.matmul(W1, X), b1); A1 = tf.nn.relu(Z1)
    Z2 = tf.add(tf.matmul(W2, A1), b2); A2 = tf.nn.relu(Z2)
    Z3 = tf.add(tf.matmul(W3, A2), b3); A3 = tf.nn.relu(Z3)
    Z4 = tf.add(tf.matmul(W4, A3), b4); A4 = tf.nn.relu(Z4)
    Z5 = tf.add(tf.matmul(W5, A4), b5); A5 = tf.nn.relu(Z5)
    Z6 = tf.add(tf.matmul(W6, A5), b6); A6 = tf.nn.relu(Z6)
    Z7 = tf.add(tf.matmul(W7, A6), b7); A7 = tf.nn.relu(Z7)
    Z8 = tf.add(tf.matmul(W8, A7), b8); A8 = tf.nn.relu(Z8)
    Z9 = tf.add(tf.matmul(W9, A8), b9); A9 = tf.nn.relu(Z9)
    Z10 = tf.add(tf.matmul(W10, A9), b10); A10 = tf.nn.relu(Z10)
    Z11 = tf.add(tf.matmul(W11, A10), b11); A11 = tf.nn.relu(Z11)
    Z12 = tf.add(tf.matmul(W12, A11), b12)
    return Z12

def compute_cost(Z12, Y):
    logits = tf.transpose(Z12)
    labels = tf.transpose(Y)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = labels))
    return cost

# Model
def model(X_train, Y_train, X_test, Y_test, X_val, Y_val, learning_rate = 0.00001,
          num_epochs = 20000, minibatch_size = 5096, print_cost = True):
    ops.reset_default_graph()
    (n_x, m) = X_train.shape
    n_y = Y_train.shape[0]
    costs = []
    val_accs = []
    X,Y = create_placeholders(n_x, n_y)
    parameters = initialize_parameters()
    Z12 = forward_propagation(X, parameters)
    cost = compute_cost(Z12, Y)
#    W1_reg = tf.nn.l2_loss(parameters["W1"])
#    W2_reg = tf.nn.l2_loss(parameters["W2"])
#    W3_reg = tf.nn.l2_loss(parameters["W3"])
#    lambd = 0.009
#    cost = tf.reduce_mean(cost + lambd * (W1_reg+W2_reg+W3_reg))
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        saver = tf.train.Saver()
        for epoch in range(num_epochs):
            epoch_cost = 0.
            num_minibatches = int(m / minibatch_size)
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size)
            for minibatch in minibatches:
                minibatch_X, minibatch_Y = minibatch
                _ , minibatch_cost = sess.run([optimizer, cost], feed_dict = {X: minibatch_X, Y: minibatch_Y})
                epoch_cost += minibatch_cost / num_minibatches 
            if print_cost == True and epoch % 10 == 0:
                costs.append(epoch_cost)
            if print_cost == True and epoch % 100 == 0:
                val_correct_prediction = tf.equal(tf.argmax(Z12), tf.argmax(Y))
                val_accuracy = tf.reduce_mean(tf.cast(val_correct_prediction, "float")) 
                print("Cost after epoch %i: %f; Val Acc = %f" % (epoch, epoch_cost, val_accuracy.eval({X: X_val, Y: Y_val})))
                val_accs.append(val_accuracy.eval({X: X_val, Y: Y_val}))
                if epoch>=100 and val_accs[int(epoch/100)] == max(val_accs):
                        saver.save(sess, '/Users/cag3fr/Desktop/ML-Triage/triage_4_classes_snapshot')
            
                
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate = " + str(learning_rate))
        plt.show()
        parameters = sess.run(parameters)
        print("Parameters have been trained!")
        correct_prediction = tf.equal(tf.argmax(Z12), tf.argmax(Y))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))        
        print("Train Accuracy: ", accuracy.eval({X: X_train, Y: Y_train}))
        print("Test Accuracy: ", accuracy.eval({X: X_test, Y: Y_test}))
        
        train_pred = sess.run(tf.argmax(Z12), feed_dict={X: X_train})
        train_act = sess.run(tf.argmax(Y), feed_dict={Y: Y_train})
        
        val_pred = sess.run(tf.argmax(Z12), feed_dict={X: X_val})
        val_act = sess.run(tf.argmax(Y), feed_dict={Y: Y_val})
        
        test_pred = sess.run(tf.argmax(Z12), feed_dict={X: X_test})
        test_act = sess.run(tf.argmax(Y), feed_dict={Y: Y_test})
        
        return parameters, train_pred, train_act, val_pred, val_act, test_pred, test_act



#%% Data preprocessing

# Import imputed dataset
df = pd.read_csv("NHAMCS_2012-2015_2018-04-09_imp.csv")

# Drop rows with missing values (RFV1 only)
#df = df.dropna()

# Remove triage level 1
df = df[df.IMMEDR > 0]
df = df[df.IMMEDR < 7]
df = df[df.TRIAGELEVEL > 1]
df = df.reset_index()

# Set Triage Level as Labels
Y = df.TRIAGELEVEL
#Y[Y==5] = 4
#Y[Y==4] = 3

# Sample data, balance 2's and 3's
#Y4s = Y[Y==4]
#num4s = list(Y4s.shape); num4s = num4s[0]
#Y2s = Y[Y==2]
#selectedY2s = Y2s.sample(n=num4s)
#Y3s = Y[Y==3]
#selectedY3s = Y3s.sample(n=num4s)

#selected_idx = selectedY2s.index.tolist()+selectedY3s.index.tolist()+Y4s.index.tolist()
#Y = Y[selected_idx]

# Remove AbnormalVS and TRIAGELEVEL
df = df.drop(columns=["ABNORMALVS","TRIAGELEVEL","IMMEDR"])
#df = df.drop(columns=["ABNORMALVS", "TRIAGELEVEL", "INJR1", "INJR2", "INTENT", "INJDETR1", "INJDETR2"])
#df = df.iloc[selected_idx,:]
df = df.drop(columns="index")
# Scale features from 0 to 1
scaler = MinMaxScaler()
scaler.fit(df)
X = scaler.transform(df)


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
X_test, X_val, Y_test, Y_val = train_test_split(X_test, Y_test, test_size=0.5)

X_train = X_train.T
X_test = X_test.T
X_val = X_val.T

# One-hot encode label matrices
Y_train = convert_to_one_hot(Y_train-2, 4)
Y_test = convert_to_one_hot(Y_test-2, 4)
Y_val = convert_to_one_hot(Y_val-2, 4)
#Y_train = convert_to_one_hot(Y_train-2, 2)
#Y_test = convert_to_one_hot(Y_test-2, 2)

parameters, train_pred, train_act, val_pred, val_act, test_pred, test_act = model(X_train, Y_train, X_test, Y_test, X_val, Y_val)