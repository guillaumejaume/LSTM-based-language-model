import numpy as np
import string
from collections import defaultdict
import random


def load_raw_data(filename):
    """ Load a file and read it line-by-line
    Parameters:
    -----------
    filename: string
    path to the file to load
    
    Returns:
    --------
    raw: list of sentences
    """
    file = open(filename, "r")
    raw_data = file.readlines()
    file.close()
    return raw_data


def argsort(seq):
    """ Argsort a list of string
    Parameters:
    -----------
    seq: list of words
    list of words to sort
    
    Returns:
    --------
    sorted_seq: list of int (indices)
    list of indices (same size as input seq) sorted
    (eg. seq=['foo', 'bar','foo','toto'] => out=[1,0,2,3])
    """
    return sorted(range(len(seq)), key=seq.__getitem__)


def add_tokens_to_sentences(raw_sentences, vocab, max_sent_length):
    """ Add the tokens <bos>, <eos>, <pad> to
    a list of sentences stored in raw_sentences
    Parameters:
    -----------
    raw_sentences: list of string
    list of sentences, where each word is already separated by a space char
    max_sent_lenght: int
    maximal size authorized for a sentence, if longer than it is discarded

    Returns:
    --------
    normalized_sentences: list of string
    sentences normalized following the process described in the handout task 1), 1a)
    """
    sentences_with_indices = []
    labels_with_indices = []
    for raw_sentence in raw_sentences:
        number_of_words = len(raw_sentence.split())
        if number_of_words <= max_sent_length-2:
            sentence = '<bos> ' + raw_sentence.rstrip() + ' <eos>'
            if max_sent_length-number_of_words-2 > 0:
                sentence += ' ' + '<pad> ' * (max_sent_length-number_of_words-3) + '<pad>'
            sentence = sentence.split(' ')
            # word2index
            sentence_with_indices = [vocab[word] for word in sentence]
            sentences_with_indices.append(sentence_with_indices)
            label_with_indices = sentence_with_indices[1:]
            labels_with_indices.append(label_with_indices)
            #print(sentence_with_indices)
            #print(label_with_indices)
            #print('\n')
    return np.asarray(sentences_with_indices), np.asarray(labels_with_indices)


def replace_unknown_words(input_sentences, vocab):
  """ replace all the words that don't belong to
    a list of known words with the token <unk>
    Parameters:
    -----------
    input_sentences: list of string
        list of sentences, where each word is already separated by a space char
    vocab: dict of words
        dict of the common words
    
    Returns:
    --------
    output_sentences: list of string
    sentences where each unknown word was replaced by <unk>
    """
  # argsort all the words from each sentence
  all_words = []
  for sentence in input_sentences:
    sentence = remove_punctuation_and_digits_from_line(sentence)
    if sentence and not sentence.isspace():
      all_words.extend(sentence.split())
      all_words.extend('\n')
  indices = argsort(all_words)

  # replace by <unk> when necessary
  current_word = ''
  replace = False
  for idx in indices:
    if not all_words[idx] == '\n':
      if current_word == all_words[idx]:
        if replace:
          all_words[idx] = '<unk>'
      else:
        replace = False
        current_word = all_words[idx]
        if not current_word in vocab.keys():
          all_words[idx] = '<unk>'
          replace = True

  # reconstruct the sentences
  output_sentences = []
  sentence = ''
  for word in all_words:
    if word == '\n':
      output_sentences.append(' '.join(sentence.split()))
      sentence = ''
    else:
      sentence += word
      sentence += ' '

  return output_sentences


def load_frequent_words(frequent_word_filename):
    frequent_words = load_raw_data(frequent_word_filename)
    frequent_words = [word.rstrip() for word in frequent_words]
    vocab = {word: i for i, word in enumerate(frequent_words)}
    return vocab


def load_and_process_data(filename, vocab, max_sent_length):
    raw_data = load_raw_data(filename)
    data = replace_unknown_words(raw_data, vocab)
    data, labels = add_tokens_to_sentences(data, vocab, max_sent_length)
    return data, labels


def remove_punctuation_and_digits_from_line(line):
    """ Removes the punctuation and the digits from line
    Parameters:
    -----------
    line: string
    line from which the punctuation and the digits should be removed
    
    Returns:
    line: string
    line with removed punctuation and digits
    
    """
    #https://stackoverflow.com/questions/265960/best-way-to-strip-punctuation-from-a-string-in-python
    line = line.translate(str.maketrans('', '', string.punctuation))
    return line


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


def generate_top_k_words(file_name, remove_punctuation, k):
    """ Generates k most frequent words
    Parameters:
    -----------
    filename: string
    path to the file to load
    
    remove_punctuation: bool
    defines whether the punctuation is removed or not
    
    k: int
    number of words 
    
    Returns:
    --------
    top_k_frequent_words: list 
    the most frequent k words
    """
    raw_lines_of_data = load_raw_data(file_name)
    lines = []
    if remove_punctuation:
        lines.extend(remove_punctuation_and_digits_from_line(line.rstrip('\n')) for line in raw_lines_of_data)
    else:
        lines.extend(line.rstrip('\n') for line in raw_lines_of_data)
    words = []
    for line in lines:
        words.extend(line.split())

    frequency_dictionary = defaultdict(int)
    for word in words:
        frequency_dictionary[word] = frequency_dictionary.get(word, 0) + 1
    
    frequency_dictionary = sorted(frequency_dictionary.items(), key=lambda item: item[1], reverse=True)
    top_k_frequent_key_pairs = frequency_dictionary[:k]
    top_k_frequent_words = [key_pair[0] for key_pair in top_k_frequent_key_pairs]
    
    return top_k_frequent_words


def write_list_to_file(string_list, filename):
    """ Writes list of items in a file, each item on a separated line
    Parameters:
    string_list: list
    list to be written
    
    filename: string
    file name
    """
    file = open(filename, "w")
    for item in string_list:
        file.write("%s\n" % item)
    file.close()
