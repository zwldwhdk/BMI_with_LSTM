import tensorflow as tf
from tensorflow import flags

tf.random.set_random_seed(777) # for reproducibility

flags.DEFINE_integer('hidden_size', 128, 'number of hidden size in lstm layers')
flags.DEFINE_integer('seq_length', 141, 'sequence length for lstm')
flags.DEFINE_integer('output_dim', 3, '3 axis, [x, y, z]')
flags.DEFINE_integer('input_dim', 204, 'number of MEG channels, 204 grd whole channel')
#flags.DEFINE_integer('batch_size', 1803, 'number of MEG channels, 204 grd whole channel')
FLAGS = flags.FLAGS
hidden_size = 128
output_dim = FLAGS.output_dim
input_dim = FLAGS.input_dim
seq_length = FLAGS.seq_length
#batch_size = FLAGS.batch_size

class crnn():
    '''
    please set flags before build model.
    about epoch, learning_rate, etc
    '''
    def __init__(self, sess, name, learning_rate):
        '''
        set initial variables for classes
        :param sess: tensorflow session
        :param name: class name
        '''
        self.sess          = sess
        self.name          = name
        self.learning_rate = learning_rate
        self._crnn()

    def cnn_1d_layer(self, x):
        '''
        1D convolution neural Network layer. Convolution filtering through time and sub-sampling
        :param x: input layer; palceholder set to [num_channels, time steps, 1]
        :return: Convolution filtered output
        '''
        conv1 = tf.layers.conv1d(
            inputs=x, filters=4, kernel_size=128, strides=1, padding='same',
            activation=tf.nn.leaky_relu, name='conv1')
        pool1 = tf.layers.max_pooling1d(
            inputs=conv1, pool_size=3, strides=3, name='max_pool1')
        pool1 = tf.layers.batch_normalization(pool1)
        '''
        $ conv1 shape: [204, 1803, 4]; pool1 shape: [204, 601, 4]
        '''
        conv2 = tf.layers.conv1d(
            inputs=pool1, filters=6, kernel_size=16, strides=1, padding='same',
            activation=tf.nn.leaky_relu, name='conv2')
        pool2 = tf.layers.max_pooling1d(
            inputs=conv2, pool_size=2, strides=2, name='max_pool2', padding='same')
        pool2 = tf.layers.batch_normalization(pool2)
        '''
        $ conv2 shape: [204, 601, 6]; pool2 shape: [204, 300, 6]
        '''
        conv3 = tf.layers.conv1d(
            inputs=pool2, filters=8, kernel_size=4, strides=1, padding='same',
            activation=tf.nn.leaky_relu, name='conv3')
        pool3 = tf.layers.max_pooling1d(
            inputs=conv3, pool_size=2, strides=2, name='max_pool3', padding='same')
        pool3 = tf.layers.batch_normalization(pool3)
        '''
        $ conv2 shape: [204, 300, 8]; pool2 shape: [204, 151, 8]
        '''
        # squeezing dims
        transpose = tf.transpose(pool3, [2, 0, 1])
        self.conv_out = tf.reshape(transpose, [input_dim*8, -1])
        return self.conv_out

    def lstm_layer(self, x, multi=False):
        '''
        bidirection LSTM layer;
        :param x: input, must be sequenced
        :param multi: multi-layered is needed?
        :return: time sequenced output
        '''
        fw_cell = tf.contrib.cudnn_rnn.CudnnCompatibleGRUCell(
            num_units=hidden_size,
            kernel_initializer=tf.contrib.layers.xavier_initializer()
        )
        bw_cell = tf.contrib.cudnn_rnn.CudnnCompatibleGRUCell(
            num_units=hidden_size,
            kernel_initializer=tf.contrib.layers.xavier_initializer()
        )
        (fw_output, bw_output), _state = \
            tf.nn.bidirectional_dynamic_rnn(
                inputs =x,
                cell_fw=fw_cell, cell_bw=bw_cell,
                dtype  =tf.float32)
        outputs = tf.concat([fw_output[-1], bw_output[-1]], axis=1)
        return outputs

    def _crnn(self):
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            # input X; [1803 time x 204 channel] x 240 batch
            self.X = tf.placeholder(tf.float32, [None, input_dim])  # time length x input dimension
            self.Y = tf.placeholder(tf.float32, [None, output_dim]) # time length x output dimension
            # convolution neural network: signal feature extraction
            X_reshape    = tf.reshape(self.X, [-1, input_dim, 1])  # expending depth for 1d convolution
            X_transpose  = tf.transpose(X_reshape, [1, 0, 2])  # input dimension x time length(convolution target) x depth
            self.cnn_out = self.cnn_1d_layer(x=X_transpose)
            '''self.cnn_out; [input dimension*filters, ?(151)]'''

            # Sequencing cnn_output through time steps
            self.seq_cnn = tf.stack(
                [self.cnn_out[:, seq:seq+seq_length] for seq in range(0, 11)]
            )
            '''stack output; [data points(11), input dimension*filters, seq_length]'''

            seq_cnn_transose = tf.transpose(self.seq_cnn, [0, 2, 1])
            self.lstm_out = self.lstm_layer(x=seq_cnn_transose)
            '''blstm output; [hiddn_size*2, hidden_size*filters, seq_length]'''

            # Dense connected layer
            self.w = tf.get_variable(
               name='dense_weight', shape=[hidden_size*2, output_dim],
               initializer=tf.contrib.layers.xavier_initializer()
            )
            self.b = tf.get_variable(name='dense_bias', shape=[output_dim])

            self.prediction = tf.matmul(self.lstm_out, self.w) + self.b


            # LOSS FUNCTION and optimizer
            self.loss = tf.losses.mean_squared_error(
                labels=self.Y, predictions=self.prediction)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

    def validation(self, x_data, y_data):
        #corr = tf.contrib.metrics.streaming_pearson_correlation(
        #    predictions=self.prediction,
        #    labels=self.Y)
        #coeff = tf.square(corr)
        pred = self.prediction - tf.reduce_mean(self.prediction, axis=0)
        real = self.Y - tf.reduce_mean(self.Y, axis=0)
        pearson_a = tf.reduce_sum(pred*real, axis=0)
        pearson_b = tf.sqrt((tf.reduce_sum(tf.square(pred), axis=0) * tf.reduce_sum(tf.square(real), axis=0)))
        corr = pearson_a/pearson_b
        coeff = tf.square(corr)
        return self.sess.run(coeff, feed_dict={self.X:x_data, self.Y:y_data})

    def loss_function(self, x_data, y_data):
        return self.sess.run([self.loss, self.optimizer],
                             feed_dict={self.X:x_data, self.Y:y_data})
    def predict(self, x_data):
        return self.sess.run(self.prediction, feed_dict={self.X:x_data})
