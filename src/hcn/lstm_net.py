import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer as xav

import numpy as np

class LSTM_net():

    def __init__(self, obs_size, nb_hidden=128, action_size=16):

        self.obs_size = obs_size
        self.nb_hidden = nb_hidden
        self.action_size = action_size

        def __graph__():
            tf.reset_default_graph()

            # entry points
            features_ = tf.placeholder(tf.float32, [1, obs_size], name='input_features')
            init_state_c_, init_state_h_ = ( tf.placeholder(tf.float32, [1, nb_hidden]) for _ in range(2) )
            action_ = tf.placeholder(tf.int32, name='ground_truth_action')
            action_mask_ = tf.placeholder(tf.float32, [action_size], name='action_mask')

            # input projection
            Wi = tf.get_variable('Wi', [obs_size, nb_hidden], 
                    initializer=xav())
            bi = tf.get_variable('bi', [nb_hidden], 
                    initializer=tf.constant_initializer(0.))

            # add relu/tanh here if necessary
            projected_features = tf.matmul(features_, Wi) + bi 

            lstm_f = tf.contrib.rnn.LSTMCell(nb_hidden, state_is_tuple=True)

            lstm_op, state = lstm_f(inputs=projected_features, state=(init_state_c_, init_state_h_))

            # reshape LSTM's state tuple (2,128) -> (1,256)
            state_reshaped = tf.concat(axis=1, values=(state.c, state.h))

            # output projection
            Wo = tf.get_variable('Wo', [2*nb_hidden, action_size], 
                    initializer=xav())
            bo = tf.get_variable('bo', [action_size], 
                    initializer=tf.constant_initializer(0.))
            # get logits
            logits = tf.matmul(state_reshaped, Wo) + bo
            # probabilities
            #  normalization : elemwise multiply with action mask
            probs = tf.multiply(tf.squeeze(tf.nn.softmax(logits)), action_mask_)
            
            # prediction
            prediction = tf.arg_max(probs, dimension=0)

            # loss
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=action_)

            # train op
            train_op = tf.train.AdadeltaOptimizer().minimize(loss)

            # attach symbols to self
            self.loss = loss
            self.prediction = prediction
            self.probs = probs
            self.logits = logits
            self.state = state

            # attach placeholders
            self.features_ = features_
            self.init_state_c_ = init_state_c_
            self.init_state_h_ = init_state_h_
            self.action_ = action_
            self.action_mask_ = action_mask_

        # build graph
        __graph__()

        # start a session; attach to self
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        self.sess = sess
        # set init state to zeros
        self.init_state_c = np.zeros([1,self.nb_hidden], dtype=np.float32)
        self.init_state_h = np.zeros([1,self.nb_hidden], dtype=np.float32)


    # forward propagation
    def forward(self, features, action_mask):
        # forward
        probs, prediction, state_c, state_h = self.sess.run( [self.probs, self.prediction, self.state.c, self.state.h], 
                feed_dict = { 
                    self.features_ : features.reshape([1,self.obs_size]), 
                    self.init_state_c_ : self.init_state_c,
                    self.init_state_h_ : self.init_state_h,
                    self.action_mask_ : action_mask
                    })
        # maintain state
        self.init_state_c = state_c
        self.init_state_h = state_h
        # return argmax
        return prediction
