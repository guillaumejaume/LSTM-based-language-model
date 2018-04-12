import preprocess_helper

from lstm_language_model import LSTMLanguageModel

import tensorflow as tf
import numpy as np

import gensim

#  Parameters

# Data loading parameters
tf.flags.DEFINE_string("data_file_path", "data/sentences.continuation", "Path to the training data")
tf.flags.DEFINE_string("vocab_file_path", "data/k_frequent_words.word2vec", "Path to the vocabulary list")
tf.flags.DEFINE_string("checkpoint_dir", "/Users/aci/Documents/PhD/Courses/NLU/Development/project/LSTM-based-language-model/runs/1523450577/checkpoints", "Checkpoint directory from training run")

# Model parameters
tf.flags.DEFINE_integer("embedding_dimension", 100, "Dimensionality of word embeddings")
tf.flags.DEFINE_integer("vocabulary_size", 20000, "Size of the vocabulary")
tf.flags.DEFINE_integer("state_size", 512, "Size of the hidden LSTM state")
tf.flags.DEFINE_integer("sentence_length", 30, "Length of the sentence to create")

# Test parameters
tf.flags.DEFINE_integer("batch_size", 1, "Batch Size")

# Embedding parameters
tf.flags.DEFINE_boolean("use_word2vec_emb", True, "Use word2vec embedding")
tf.flags.DEFINE_string("path_to_word2vec", "wordembeddings-dim100.word2vec", "Path to the embedding file")

# Tensorflow Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS

# Prepare data

# Load data
print("Load vocabulary list \n")
generated_model = gensim.models.KeyedVectors.load_word2vec_format(FLAGS.vocab_file_path, binary=False)
vocab = {word: idx for idx, word in enumerate(generated_model.vocab)}

print("Loading and preprocessing test dataset \n")
x_test, y_test = preprocess_helper.load_and_process_data(FLAGS.data_file_path,
                                                         vocab,
                                                         FLAGS.sentence_length,
                                                         eos_token=False,
                                                         pad_sentence=False)
## EVALUATION ##

checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
        inputs = graph.get_operation_by_name("inputs").outputs[0]
        vocab_embedding = graph.get_operation_by_name("vocab_embedding").outputs[0]

        # Tensors we want to evaluate
        predicted_sentence = graph.get_operation_by_name("continuation/predicted_sentence").outputs[0]

        # Generate batches for one epoch
        batches = preprocess_helper.batch_iter(list(zip(x_test, y_test)), FLAGS.batch_size, 1, shuffle=False)

        # Construct the embedding matrix
        vocab_emb = np.zeros(shape=(FLAGS.vocabulary_size, FLAGS.embedding_dimension))
        w2v_model = gensim.models.KeyedVectors.load_word2vec_format(FLAGS.path_to_word2vec, binary=False)
        for idx, tok in enumerate(generated_model.vocab):
            if FLAGS.use_word2vec_emb and  tok in w2v_model.vocab:
                vocab_emb[idx] = w2v_model[tok]
            else:
                vocab_emb[idx] = generated_model[tok]

        # Collect the predictions here
        all_perplexity = []

        for batch_id, batch in enumerate(batches):
            x_batch, y_batch = zip(*batch)
            predicted_sentence_batch = sess.run([predicted_sentence], {inputs: x_batch,
                                                                       vocab_embedding: vocab_emb})

            # print('predicted sentence as indices: ', predicted_sentence_batch[0][0])
            prediction_sentence = ''
            ground_truth_sentence = ''
            y_batch = y_batch[0]
            for i in range(len(predicted_sentence_batch[0][0])):
                word = (list(vocab.keys())[list(vocab.values()).index(predicted_sentence_batch[0][0, i])])
                prediction_sentence += word
                prediction_sentence += ' '
            print('beginning of sentence: ', x_batch)
            print('predicted sentence as words: ', prediction_sentence)
            print('\n')
