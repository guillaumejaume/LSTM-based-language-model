import tensorflow as tf
from tensorflow.python.util import nest
from tensorflow.python.ops import array_ops

class LSTMLanguageModel:
    """
    LSTM language model class definition
    """

    def __init__(self, vocabulary_size, embedding_dimensions, state_size):

        self.vocabulary_size = vocabulary_size

        self.inputs = tf.placeholder(dtype=tf.int32,
                                     shape=[None, None],
                                     name='inputs')

        self.vocab_embedding = tf.placeholder(dtype=tf.float32,
                                              shape=[self.vocabulary_size, embedding_dimensions],
                                              name='vocab_embedding')

        self.labels = tf.placeholder(dtype=tf.int32,
                                     shape=[None, None],
                                     name='labels')


        with tf.device('/gpu:0'):

            with tf.variable_scope("Embedding", reuse=tf.AUTO_REUSE):
                self.embedded_inputs = tf.nn.embedding_lookup(self.vocab_embedding,
                                                              self.inputs,
                                                              name='embedded_inputs')

            with tf.variable_scope("LSTM_layer"):
                # construct basic LSTM cell
                self.lstm_cell = tf.contrib.rnn.BasicLSTMCell(state_size)
                # return value is a N-D tensor of shape [batch_size, state_size] filled with zeros.
                initial_state = self.lstm_cell.zero_state(tf.shape(self.embedded_inputs)[0], tf.float32)
                output, state = tf.nn.dynamic_rnn(self.lstm_cell,
                                                  self.embedded_inputs,
                                                  initial_state=initial_state,
                                                  dtype=tf.float32)

                output = output[:, :-1, :]

                # where to store the cell_output after each time_step
                # output = []
                # for time_step in range(self.sentence_length-1):
                #     try:
                #         cell_output, self.state = lstm_cell(self.embedded_inputs[:, time_step, :], self.state)
                #         output.append(cell_output)
                #     except:
                #         pass
                #         print('out of bounds...')
                output = tf.reshape(output,
                                    [(tf.shape(self.embedded_inputs)[1]-1) * tf.shape(self.embedded_inputs)[0],
                                     state_size])

                # tensor flow while loop ...
                # self.output = []
                # i0 = tf.constant(0)
                # condition = lambda i, a, b: i < (tf.shape(self.embedded_inputs)[1] - 1)
                #
                # def body(i, state, embedded_inputs):
                #     cell_output, state = lstm_cell(embedded_inputs[:, i, :], state)
                #     self.output.append(cell_output)
                #     i = tf.add(i, 1)
                #     return i, state, embedded_inputs
                #
                # i, state, embedded_inputs = tf.while_loop(condition,
                #                                           body,
                #                                           [i0, self.state, self.embedded_inputs])
                #
                # self.output = tf.reshape(self.output,
                #                          [(tf.shape(self.embedded_inputs)[1] - 1) * tf.shape(self.embedded_inputs)[0],
                #                           state_size])
                #
                # print(self.output.shape)

            with tf.variable_scope("softmax_layer"):
                self.weights = tf.get_variable("weights",
                                          [state_size, self.vocabulary_size],
                                          dtype=tf.float32,
                                          initializer=tf.contrib.layers.xavier_initializer())
                self.bias = tf.get_variable("bias",
                                       [self.vocabulary_size],
                                       dtype=tf.float32,
                                       initializer=tf.contrib.layers.xavier_initializer())
                self.logits = tf.matmul(output, self.weights) + self.bias
                self.logits = tf.reshape(self.logits,
                                         [tf.shape(self.inputs)[0], tf.shape(self.inputs)[1]-1, self.vocabulary_size])
                self.probabilities = tf.nn.softmax(self.logits)

            with tf.variable_scope("loss"):
                self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels,
                                                                           logits=self.logits)
                self.loss = tf.reduce_mean(self.loss)

            with tf.variable_scope("perplexity"):
                self.perplexity = tf.exp(self.loss,
                                         name='perplexity')

            with tf.variable_scope("accuracy"):
                self.predictions = tf.argmax(self.probabilities,
                                             axis=2,
                                             output_type=tf.int32,
                                             name='predictions')
                # compute the average accuracy over the batch acc dim = batch_size
                self.is_equal = tf.equal(self.predictions, self.labels)
                self.accuracy = tf.reduce_mean(tf.cast(self.is_equal, tf.float32),
                                               name='accuracy')

            with tf.variable_scope("continuation"):
                continuation_length = 30
                predicted_word = self.predictions[:, -1]
                predicted_sentence = tf.concat([self.inputs, tf.expand_dims(predicted_word, 1)], axis=1)
                embedded_predicted_word = tf.nn.embedding_lookup(self.vocab_embedding,
                                                                 predicted_word)
                for time_step in range(continuation_length):
                    predicted_word, state = self.predict_next_word(embedded_predicted_word, state)
                    embedded_predicted_word = tf.nn.embedding_lookup(self.vocab_embedding,
                                                                     predicted_word)
                    predicted_sentence = tf.concat([predicted_sentence, tf.expand_dims(predicted_word, 1)], axis=1)

                self.predicted_sentence = tf.identity(predicted_sentence, name='predicted_sentence')

    def predict_next_word(self, current_word, state):
        cell_output, state = self.lstm_cell(current_word, state)
        logits = tf.matmul(cell_output, self.weights) + self.bias
        probabilities = tf.nn.softmax(logits)
        predicted_word = tf.argmax(probabilities,
                                   axis=1,
                                   output_type=tf.int32,
                                   name='continuation_predictions')
        return predicted_word, state

