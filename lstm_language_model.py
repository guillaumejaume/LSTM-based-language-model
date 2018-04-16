import tensorflow as tf
from tensorflow.python.util import nest
from tensorflow.python.ops import array_ops


class LSTMLanguageModel:
    """
    LSTM language model
        Predict from a set of observations, probabilities
        of the next word given a fixed-size vocabulary
    """

    def __init__(self, vocabulary_size, embedding_dimensions, state_size):

        self.vocabulary_size = vocabulary_size

        self.state_size = state_size

        self.inputs = tf.placeholder(dtype=tf.int32,
                                     shape=[None, None],
                                     name='inputs')

        self.vocab_embedding = tf.placeholder(dtype=tf.float32,
                                              shape=[self.vocabulary_size, embedding_dimensions],
                                              name='vocab_embedding')

        self.labels = tf.placeholder(dtype=tf.int32,
                                     shape=[None, None],
                                     name='labels')

        self.discard_last_prediction = tf.placeholder(tf.bool)



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

                # With TensorFlow built-in function
                #output, state = tf.nn.dynamic_rnn(self.lstm_cell,
                #                                  self.embedded_inputs,
                #                                  initial_state=initial_state,
                #                                  dtype=tf.float32)

                # With our implementation
                output, state = self.rnn_dynamic(state=initial_state)

                # remove the last prediction from the output (irr because all the sent have a fixed-len of 30 words)
                output = tf.cond(tf.equal(self.discard_last_prediction, True), lambda: output[:, :-1, :], lambda: output)

                output_concat = tf.reshape(output,
                                    [tf.shape(output)[0] * tf.shape(output)[1],
                                     self.state_size])

            with tf.variable_scope("softmax_layer"):

                self.weights = tf.get_variable("weights",
                                          [self.state_size, self.vocabulary_size],
                                          dtype=tf.float32,
                                          initializer=tf.contrib.layers.xavier_initializer())

                self.bias = tf.get_variable("bias",
                                       [self.vocabulary_size],
                                       dtype=tf.float32,
                                       initializer=tf.contrib.layers.xavier_initializer())

                self.logits = tf.matmul(output_concat, self.weights) + self.bias
                self.logits = tf.reshape(self.logits,
                                         [tf.shape(output)[0], tf.shape(output)[1], self.vocabulary_size])
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
                # construct sentence of a fixed length of 30 words
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
        """ Given a word and a state, predict the next word
          Parameters:
          -----------
          current_word: tensor of shape [batch_size x embedding_dimension]
              input word of the LSTM cell
          state: tuple
              hidden state of the LSTM cell

          Returns:
          ----------
          predicted_words: tensor of shape [batch_size]
              predicted words represented as an integer ranging from 0 to |V|-1
          state: tuple
              updated hidden state of the LSTM
          """
        cell_output, state = self.lstm_cell(current_word, state)

        logits = tf.matmul(cell_output, self.weights) + self.bias
        probabilities = tf.nn.softmax(logits)
        predicted_word = tf.argmax(probabilities,
                                   axis=1,
                                   output_type=tf.int32,
                                   name='continuation_predictions')
        return predicted_word, state

    def rnn_dynamic(self, state):
        """ Enroll the LSTM network time_steps number of times. This function is
            a reimplementation of the tf.dynamic_rnn TensorFlow built-in function
          state:
          -----------
          state: tuple
              current hidden state of the LSTM cell (called here as the initial state)

          Returns:
          ----------
          output: tensor of shape [batch_size x time_steps x state_size]
              enrolled output of the LSTM cell
          final_state: tuple
              final hidden state of the LSTM cell
          """

        # extract time_steps from the input tensor
        embedded_inputs_shape = tf.shape(self.embedded_inputs)
        time_steps = embedded_inputs_shape[1]

        # define a TensorFlow time var (to increment at each loop)
        time = tf.constant(0, dtype=tf.int32, name="time")

        # from batch-major to time-major input
        time_major_input = tf.transpose(self.embedded_inputs, perm=[1, 0, 2])

        # Use TensorFlow while loop here. A classic for-loop over time_steps is not feasible as
        # it will be known only at runtime (Input placeholder has shape `None` for the sent len).

        # Define TensorArray where we will read from and write in at each loop
        output_as_ta = tf.TensorArray(size=time_steps, dtype=tf.float32)
        input_as_ta = tf.TensorArray(size=time_steps, dtype=tf.float32)
        input_as_ta = input_as_ta.unstack(time_major_input)

        def condition(time, output, state):
            """
                Stopping criteria: loop until time is equal to time_steps
            """
            return time < time_steps

        def process_time_step(time, output_ta_t, current_state):
            """
                Operation to process for each time step.
                    - Read the input at time t from the the input TensorArray
                    - Call the lstm cell
                    - Write the ouput
                    - Increment the time step
             """
            current_time_input = input_as_ta.read(time)
            new_output, new_state = self.lstm_cell(current_time_input, current_state)
            output_ta_t = output_ta_t.write(time, new_output)
            return time + 1, output_ta_t, new_state

        _, output, final_state = tf.while_loop(cond=condition,
                                               body=process_time_step,
                                               loop_vars=(time, output_as_ta, state))

        # stack the TensorArray to get a Tensor of shape [time_steps x batch_size x emb_dims]
        output = output.stack()
        # transpose to shape [batch_size x time_steps x emb_dims]
        output = tf.transpose(output, perm=[1, 0, 2])

        return output, final_state

