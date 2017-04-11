import tensorflow as tf
import tensorflow.contrib.layers.xavier_initializer as xav

class LSTM_net():

    def __init__(self, input_size, nb_hidden, output_size):

        def __graph__():

            tf.reset_default_graph()

            # placeholders
            features_ = tf.placeholders(tf.float32, [None,], name='input_features')
            init_state_ = tf.placeholders(tf.float32, [state_size], name='init_state')
 

            # input projection
            Wi = tf.get_variable('Wi', [input_size, nb_hidden], 
                    initializer=xav())
            bi = tf.get_variable('bi', [nb_hidden], 
                    initaliazer=tf.constant_initializer(0.))
            # add relu/tanh here if necessary
            projected_features = tf.matmul(features_, Wi) + bi 

            lstm_f = tf.nn.rnn_cell.LSTMCell(state_size, state_is_tuple=True)
