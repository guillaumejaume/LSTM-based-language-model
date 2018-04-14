import preprocess_helper

from lstm_language_model import LSTMLanguageModel

import tensorflow as tf
import numpy as np

import gensim

import numpy as np

np.set_printoptions(threshold=np.nan)

#  Parameters

# Data loading parameters
tf.flags.DEFINE_string("data_file_path", "data/sentences.test", "Path to the training data")
tf.flags.DEFINE_string("vocab_with_emb_path", "data/vocab_with_emb.txt", "Path to the vocabulary list")
tf.flags.DEFINE_string("checkpoint_dir", "./runs/1523637670/checkpoints", "Checkpoint directory from training run")

# Model parameters
tf.flags.DEFINE_integer("embedding_dimension", 100, "Dimensionality of word embeddings")
tf.flags.DEFINE_integer("vocabulary_size", 20000, "Size of the vocabulary")
tf.flags.DEFINE_integer("sentence_length", 30, "Length of each sentence fed to the LSTM")

# Test parameters
tf.flags.DEFINE_integer("batch_size", 1, "Batch Size")

# Embedding parameters
tf.flags.DEFINE_boolean("use_word2vec_emb", True, "Use word2vec embedding")
tf.flags.DEFINE_string("path_to_word2vec", "wordembeddings-dim100.word2vec", "Path to the embedding file")

# Tensorflow Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
tf.flags.DEFINE_boolean("verbose_for_debugging", False, "Allow info to be printed to understand the behaviour of the network")
tf.flags.DEFINE_boolean("verbose_for_experiments", True, "Print only the perplexity")

FLAGS = tf.flags.FLAGS

# Prepare data

# Load data
print("Load vocabulary list \n")
vocab, generated_embeddings = preprocess_helper.load_frequent_words_and_embeddings(FLAGS.vocab_with_emb_path)

print("Loading and preprocessing test dataset \n")
x_test, y_test = preprocess_helper.load_and_process_data(FLAGS.data_file_path,
                                                         vocab,
                                                         FLAGS.sentence_length,
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
        labels = graph.get_operation_by_name("labels").outputs[0]
        vocab_embedding = graph.get_operation_by_name("vocab_embedding").outputs[0]

        # Tensors we want to evaluate
        predictions = graph.get_operation_by_name("accuracy/predictions").outputs[0]
        perplexity = graph.get_operation_by_name("perplexity/perplexity").outputs[0]

        # Generate batches for one epoch
        batches = preprocess_helper.batch_iter(list(zip(x_test, y_test)), FLAGS.batch_size, 1, shuffle=False)

        # Construct the embedding matrix
        vocab_emb = np.zeros(shape=(FLAGS.vocabulary_size, FLAGS.embedding_dimension))
        w2v_model = gensim.models.KeyedVectors.load_word2vec_format(FLAGS.path_to_word2vec, binary=False)
        for tok, idx in vocab.items():
            if FLAGS.use_word2vec_emb and tok in w2v_model.vocab:
                vocab_emb[idx] = w2v_model[tok]
            else:
                vocab_emb[idx] = generated_embeddings[tok]

        # Collect the predictions here
        all_perplexity = []

        if(FLAGS.verbose_for_experiments):
            print("verbose_for_experiments: Only the perplexity will be shown for each sentence")

        for batch_id, batch in enumerate(batches):
            x_batch, y_batch = zip(*batch)
            batch_predictions, batch_perplexity = sess.run([predictions, perplexity],
                                                           {inputs: x_batch,
                                                            labels: y_batch,
                                                            vocab_embedding: vocab_emb})
            all_perplexity.append(batch_perplexity)
            prediction_sentence = ''
            ground_truth_sentence = ''
            y_batch = y_batch[0]
            for i in range(len(batch_predictions[0])):
                word = (list(vocab.keys())[list(vocab.values()).index(batch_predictions[0, i])])
                #print("predicted %s \n"% word)
                prediction_sentence += word
                prediction_sentence += ' '

                word = (list(vocab.keys())[list(vocab.values()).index(y_batch[i])])
                #print("predicted %s \n" % word)
                ground_truth_sentence += word
                ground_truth_sentence += ' '

            if(FLAGS.verbose_for_debugging == True):
                print("y_b ", y_batch)
                print("b_pr", batch_predictions)
                print('ground truth: ', ground_truth_sentence)
                print('predictions: ', prediction_sentence)
                print('perplexity: ', batch_perplexity)
                print('\n')

            if(FLAGS.verbose_for_experiments == True):
                print(batch_perplexity)
                print('\n')
# Print average perplexity
average_perplexity = np.mean(np.asarray(all_perplexity))
print("Average: {:g}".format(average_perplexity))
