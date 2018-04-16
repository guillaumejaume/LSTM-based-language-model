import preprocess_helper

from lstm_language_model import LSTMLanguageModel

import tensorflow as tf
import numpy as np
import os
import time
import datetime

import gensim
import numpy as np

np.set_printoptions(threshold=np.nan)
#Training parameters

# Data loading parameters
tf.flags.DEFINE_float("val_sample_percentage", .001, "Percentage of the training data used for validation")
tf.flags.DEFINE_string("data_file_path", "data/sentences.train", "Path to the training data")
tf.flags.DEFINE_string("vocab_with_emb_path", "data/vocab_with_emb.txt", "Path to the vocabulary list")

# Model parameters
tf.flags.DEFINE_integer("embedding_dimension", 100, "Dimensionality of word embeddings")
tf.flags.DEFINE_integer("vocabulary_size", 20000, "Size of the vocabulary")
tf.flags.DEFINE_integer("state_size", 1024, "Size of the hidden LSTM state")
tf.flags.DEFINE_integer("output_size", 512, "Size of the downprojected output of the LSTM")
tf.flags.DEFINE_integer("sentence_length", 30, "Length of each sentence fed to the LSTM")

# Embedding parameters
tf.flags.DEFINE_boolean("use_word2vec_emb", True, "Use word2vec embedding")
tf.flags.DEFINE_string("path_to_word2vec", "wordembeddings-dim100.word2vec", "Path to the embedding file")

# Training parameters
tf.flags.DEFINE_integer("max_grad_norm", 5, "max norm of the gradient")
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size")
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on validation set after this many steps")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store")

# Tensorflow Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

# for running on EULER, adapt this
tf.flags.DEFINE_integer("inter_op_parallelism_threads", 0,
                        "TF nodes that perform blocking operations are enqueued on a pool of "
                        "inter_op_parallelism_threads available in each process.")
tf.flags.DEFINE_integer("intra_op_parallelism_threads", 0,
                        "The execution of an individual op (for some op types) can be parallelized"
                        " on a pool of intra_op_parallelism_threads.")

FLAGS = tf.flags.FLAGS

# Prepare the data
print("Load vocabulary list \n")
vocab, generated_embeddings = preprocess_helper.load_frequent_words_and_embeddings(FLAGS.vocab_with_emb_path)

print("Loading and preprocessing training and validation datasets \n")
data, labels = preprocess_helper.load_and_process_data(FLAGS.data_file_path,
                                                       vocab,
                                                       FLAGS.sentence_length)

# Randomly shuffle data
np.random.seed(10)
shuffled_indices = np.random.permutation(len(labels))
data = data[shuffled_indices]
labels = labels[shuffled_indices]

# Split train/dev sets
val_sample_index = -1 * int(FLAGS.val_sample_percentage * float(len(labels)))
x_train, x_val = data[:val_sample_index], data[val_sample_index:]
y_train, y_val = labels[:val_sample_index], labels[val_sample_index:]

# Summary of the loaded data
print('Loaded: ', len(x_train), ' samples for training')
print('Loaded: ', len(x_val), ' samples for validation')

print('Training input has shape: ', np.shape(x_train))
print('Validation input has shape: ', np.shape(x_val))

print('Training labels has shape: ', np.shape(y_train))
print('Validation labels has shape: ', np.shape(y_val))

# Generate training batches
batches = preprocess_helper.batch_iter(list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)

print("Loading and preprocessing done \n")

# Define the model and start training
with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        log_device_placement=FLAGS.log_device_placement,
        inter_op_parallelism_threads=FLAGS.inter_op_parallelism_threads,
        intra_op_parallelism_threads=FLAGS.intra_op_parallelism_threads)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Initialize model
        lstm_language_model = LSTMLanguageModel(vocabulary_size=FLAGS.vocabulary_size,
                                                embedding_dimensions=FLAGS.embedding_dimension,
                                                state_size=FLAGS.state_size)

        # Training step
        global_step = tf.Variable(0, name="global_step", trainable=False)
        # Define Adam optimizer
        learning_rate = 0.0002
        optimizer = tf.train.AdamOptimizer(learning_rate)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(lstm_language_model.loss, tvars),
                                          FLAGS.max_grad_norm)
        train_op = optimizer.apply_gradients(zip(grads, tvars),
                                             global_step=tf.train.get_or_create_global_step())

        # Output directory for models and summaries
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        print("Writing to {}\n".format(out_dir))

        # Summaries for loss, perplexity and accuracy
        loss_summary = tf.summary.scalar("loss", lstm_language_model.loss)
        perplexity_summary = tf.summary.scalar("perplexity", lstm_language_model.perplexity)
        acc_summary = tf.summary.scalar("accuracy", lstm_language_model.accuracy)

        # Train Summaries
        train_summary_op = tf.summary.merge([loss_summary, perplexity_summary, acc_summary])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        # Validation summaries
        val_summary_op = tf.summary.merge([loss_summary, perplexity_summary, acc_summary])
        val_summary_dir = os.path.join(out_dir, "summaries", "dev")
        val_summary_writer = tf.summary.FileWriter(val_summary_dir, sess.graph)

        # Checkpoint directory (Tensorflow assumes this directory already exists so we need to create it)
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

        # Initialize all variables
        sess.run(tf.global_variables_initializer())
        sess.graph.finalize()

        # Define training and validation steps (batch)
        def train_step(inputs, labels, vocab_emb):
            """
            A single training step
            """
            feed_dict = {
                lstm_language_model.inputs: inputs,
                lstm_language_model.labels: labels,
                lstm_language_model.vocab_embedding: vocab_emb,
                lstm_language_model.discard_last_prediction: True
            }
            _, step, summaries, loss, perplexity, accuracy = sess.run([train_op,
                                                                       global_step,
                                                                       train_summary_op,
                                                                       lstm_language_model.loss,
                                                                       lstm_language_model.perplexity,
                                                                       lstm_language_model.accuracy], feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print('\n\n')
            print("{}: step {}, perplexity {:g}, acc {:g}".format(time_str, step, perplexity, accuracy))
            train_summary_writer.add_summary(summaries, step)

        def dev_step(inputs, labels, vocab_emb, writer=None):
            """
            Evaluates model on the validation set
            """
            feed_dict = {
                lstm_language_model.inputs: inputs,
                lstm_language_model.labels: labels,
                lstm_language_model.vocab_embedding: vocab_emb
            }
            step, summaries, predictions, perplexity, accuracy = sess.run([global_step,
                                                                           val_summary_op,
                                                                           lstm_language_model.predictions,
                                                                           lstm_language_model.perplexity,
                                                                           lstm_language_model.accuracy], feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, perplexity {:g}, acc {:g}".format(time_str, step, perplexity, accuracy))
            print('Predictions: ', predictions)
            if writer:
                writer.add_summary(summaries, step)
        #  Embedding function use word2vec embedding option here
        vocab_emb = np.zeros(shape=(FLAGS.vocabulary_size, FLAGS.embedding_dimension))
        w2v_model = gensim.models.KeyedVectors.load_word2vec_format(FLAGS.path_to_word2vec, binary=False)
        for tok, idx in vocab.items():
            if FLAGS.use_word2vec_emb and tok in w2v_model.vocab:
                vocab_emb[idx] = w2v_model[tok]
            else:
                vocab_emb[idx] = generated_embeddings[tok]

        # TRAINING LOOP
        for batch in batches:
            x_batch, y_batch = zip(*batch)
            train_step(x_batch, y_batch, vocab_emb)
            current_step = tf.train.global_step(sess, global_step)
            if current_step % FLAGS.evaluate_every == 0:
                print("\nEvaluation:")
                dev_step(x_val, y_val, vocab_emb, writer=val_summary_writer)
                print("")
            if current_step % FLAGS.checkpoint_every == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))
