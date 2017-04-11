import tensorflow as tf
import tensorflow.contrib.layers.xavier_initializer as xav

class LSTM_net():

    def __init__(self, input_size, nb_hidden, output_size):

        def __graph__():

            tf.reset_default_graph()

            # placeholders
            features_ = tf.placeholders(tf.float32, [None, input_size], name='input_features')
            init_state_ = tf.placeholders(tf.float32, [nb_hidden], name='init_state')
 

            # input projection
            Wi = tf.get_variable('Wi', [input_size, nb_hidden], 
                    initializer=xav())
            bi = tf.get_variable('bi', [nb_hidden], 
                    initaliazer=tf.constant_initializer(0.))
            # add relu/tanh here if necessary
            projected_features = tf.matmul(features_, Wi) + bi 

            lstm_f = tf.nn.rnn_cell.LSTMCell(nb_hidden, state_is_tuple=True)

            lstm_op, state = lstm_f(inputs=projected_features, state=init_state_)
