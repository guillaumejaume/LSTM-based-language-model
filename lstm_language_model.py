import tensorflow as tf


class LSTMLanguageModel:
    """
    LSTM language model class definition
    """

    def build_lstm_graph(self, lstm_cell, input_batch, sentence_length, state_size):
        # cell: tensorflow LSTM object
        # batch: tensorflow of shape batch_size x sent_len x emb_dim

        # return value is a N-D tensor of shape [batch_size, state_size] filled with zeros.
        initial_state = lstm_cell.zero_state(tf.shape(input_batch)[0], tf.float32)

        # init state is the init one
        state = initial_state

        # where to store the cell_output after each time_step
        output = []
        for time_step in range(sentence_length-1):
            cell_output, state = lstm_cell(input_batch[:, time_step, :], state)
            output.append(cell_output)
        output = tf.reshape(output, [(sentence_length-1) * tf.shape(input_batch)[0], state_size])
        return output, state

    def __init__(self, sentence_length, vocabulary_size, embedding_dimensions, state_size):

        self.inputs = tf.placeholder(dtype=tf.int32,
                                     shape=[None, sentence_length],
                                     name='inputs')

        self.vocab_embedding = tf.placeholder(dtype=tf.float32,
                                              shape=[vocabulary_size, embedding_dimensions],
                                              name='vocab_embedding')

        self.labels = tf.placeholder(dtype=tf.int32,
                                     shape=[None, sentence_length-1],
                                     name='labels')

        with tf.device('/gpu:0'):

            with tf.variable_scope("Embedding", reuse=tf.AUTO_REUSE):
                self.embedded_inputs = tf.nn.embedding_lookup(self.vocab_embedding,
                                                              self.inputs,
                                                              name='embedded_inputs')

            with tf.variable_scope("LSTM_layer"):
                # construct basic LSTM cell
                lstm_cell = tf.contrib.rnn.BasicLSTMCell(state_size)
                # enroll the network (state = final_state)
                self.output, self.state = self.build_lstm_graph(lstm_cell,
                                                                self.embedded_inputs,
                                                                sentence_length,
                                                                state_size)

            with tf.variable_scope("softmax_layer"):
                # project state size on the vocab size dim = state_size x vocabulary_size
                weights = tf.get_variable("weights",
                                          [state_size, vocabulary_size],
                                          dtype=tf.float32,
                                          initializer=tf.contrib.layers.xavier_initializer())
                # add a bias dim = vocabulary_size
                bias = tf.get_variable("bias",
                                       [vocabulary_size],
                                       dtype=tf.float32,
                                       initializer=tf.contrib.layers.xavier_initializer())
                # compute the logits
                self.logits = tf.matmul(self.output, weights) + bias
                self.logits = tf.reshape(self.logits,
                                         [tf.shape(self.inputs)[0], sentence_length-1, vocabulary_size])
                self.probabilities = tf.nn.softmax(self.logits)

            with tf.variable_scope("loss"):
                # @TODO compute the loss only on the tokens that make sense (without all the padding)

                #end_of_sentence = tf.where(tf.equal(self.labels, val))

                self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels,
                                                                           logits=self.logits)
                self.loss = tf.reduce_mean(self.loss)

            with tf.variable_scope("perplexity"):
                self.perplexity = tf.exp(self.loss, name='perplexity')

            with tf.variable_scope("accuracy"):
                self.predictions = tf.argmax(self.probabilities, axis=2, output_type=tf.int32)
                # compute the average accuracy over the batch acc dim = batch_size
                self.is_equal = tf.equal(self.predictions, self.labels)
                self.accuracy = tf.reduce_mean(tf.cast(self.is_equal, tf.float32))

