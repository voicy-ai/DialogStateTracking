import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer as xav

class LSTM_net():

    def __init__(self, obs_size, nb_hidden=128, action_size=16):

        self.obs_size = obs_size
        self.nb_hidden = nb_hidden
        self.action_size = action_size

        def __graph__():
            tf.reset_default_graph()

            # entry points
            features_ = tf.placeholder(tf.float32, [1, obs_size], name='input_features')
            init_state_ = ( tf.placeholder(tf.float32, [1, nb_hidden]) for _ in range(2) )
            action_ = tf.placeholder(tf.int32, name='ground_truth_action')
            accum_loss_ = tf.placeholder(tf.float32, name='accumulated_loss')

            # input projection
            Wi = tf.get_variable('Wi', [obs_size, nb_hidden], 
                    initializer=xav())
            bi = tf.get_variable('bi', [nb_hidden], 
                    initializer=tf.constant_initializer(0.))

            # add relu/tanh here if necessary
            projected_features = tf.matmul(features_, Wi) + bi 

            lstm_f = tf.contrib.rnn.LSTMCell(nb_hidden, state_is_tuple=True)

            lstm_op, state = lstm_f(inputs=projected_features, state=init_state_)

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
            probs = tf.nn.softmax(logits)
            
            # prediction
            prediction = tf.arg_max(tf.squeeze(probs), dimension=0)

            # loss
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=action_)

            # train op
            train_op = tf.train.AdadeltaOptimizer().minimize(loss)

            # attach symbols to self
            self.loss = loss
            self.prediction = prediction
            self.probs = prediction
            self.logits = prediction

        # build graph
        __graph__()
