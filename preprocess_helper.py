import numpy as np
import string
from collections import defaultdict

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


def add_tokens_to_sentences(raw_sentences, vocab, max_sent_length, eos_token=True, pad_sentence=True):
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
        sentences normalized as indices following the process described in the handout task 1), 1a)
    """
    sentences_with_indices = []
    labels_with_indices = []
    for raw_sentence in raw_sentences:
        number_of_words = len(raw_sentence.split())
        if number_of_words <= max_sent_length-2:
            sentence = '<bos> ' + raw_sentence.rstrip()
            if eos_token:
                sentence += ' <eos>'
            if max_sent_length-number_of_words-2 > 0 and pad_sentence:
                sentence += ' ' + '<pad> ' * (max_sent_length-number_of_words-3) + '<pad>'
            sentence = sentence.split(' ')
            # word2index
            sentence_with_indices = [vocab[word] for word in sentence]
            sentences_with_indices.append(sentence_with_indices)
            label_with_indices = sentence_with_indices[1:]
            labels_with_indices.append(label_with_indices)
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


def load_frequent_words_and_embeddings(frequent_word_filename):
    """ Load the list of frequent words and their embeddings
      Parameters:
      -----------
      frequent_word_filename: path to file
          text file where each line is a freq word to parse

      Returns:
      --------
      vocab: dict
        - the keys are the freq words
        - the values are the indices
      embeddings: dict
        - the keys are the freq words
        - the values are the embeddings
      """
    lines_of_data = load_raw_data(frequent_word_filename)
    lines_of_data = [line.rstrip() for line in lines_of_data]

    #Avoids the for loop but calls the split function 4 times
    #vocab = {line.split()[0]: line.split()[1] for line.split() in lines_of_data}
    #embeddings = {line.split()[0]: line.split()[2:] for line in lines_of_data}

    vocab={}
    embeddings={}
    for line in lines_of_data:
        line_as_array = line.split()
        vocab[line_as_array[0]]=int(line_as_array[1])
        embeddings[line_as_array[0]]=line_as_array[2:]
    return vocab, embeddings


def load_and_process_data(filename, vocab, max_sent_length, eos_token=True, pad_sentence=True):
    """ Load the list of frequent words
      Parameters:
      -----------
      filename: string
          path to file containing the raw data (ie. the sentences)
    vocab: dict
        - the keys are the freq words
        - the values are the indices
    max_sent_length: int
        max authorized length for a sentence
    eos_token: bool (default=True)
        if the token <eos> should be added at the end of each sentence
    pas_sentence: bool (default=True)
        if the sentences should be padded to match the max_sent_length

      Returns:
      --------
      vocab: dict
        - the keys are the freq words
        - the values are the indices
      """
    raw_data = load_raw_data(filename)
    data = replace_unknown_words(raw_data, vocab)
    data, labels = add_tokens_to_sentences(data, vocab, max_sent_length, eos_token, pad_sentence)
    return data, labels


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


def generate_top_k_words_and_their_emb(file_name, k, embedding_dimension):
    """ Generates k most frequent words
    Parameters:
    -----------
    filename: string
    path to the file to load
    
    k: int
    number of words 
    
    Returns:
    --------
    top_k_frequent_words_and_embedding_pairs: list
    the most frequent k words with their embeddings
    """
    extra_words = ['<unk>', '<pad>', '<bos>', '<eos>']
    raw_lines_of_data = load_raw_data(file_name)
    lines = []
    lines.extend(line.rstrip('\n') for line in raw_lines_of_data)
    words = []
    for line in lines:
        words.extend(line.split())

    frequency_dictionary = defaultdict(int)
    for word in words:
        frequency_dictionary[word] = frequency_dictionary.get(word, 0) + 1
    
    frequency_dictionary = sorted(frequency_dictionary.items(), key=lambda item: item[1], reverse=True)
    top_k_frequent_key_pairs = frequency_dictionary[:k]
    top_k_frequent_words_and_embedding_pairs = [(key_pair[0],i, np.random.uniform(low=-1, high=1, size=embedding_dimension)) for i, key_pair in enumerate(top_k_frequent_key_pairs)]
    top_k_frequent_words_and_embedding_pairs.extend((extra_word, k+i, np.random.uniform(low=-1, high=1, size=embedding_dimension)) for i,extra_word in enumerate(extra_words))

    return top_k_frequent_words_and_embedding_pairs


def write_list_to_file(tuple_map, filename):
    """ Writes list of items in a file, each item on a separated line
    Parameters:
    tuple_map: map
    tuple_map to be written
    
    filename: string
    file name
    """

    file = open(filename, "w")
    for item in tuple_map:
        file.write("%s %s " % (str(item[0]), str(item[1])))
        for element in item[2]:
            file.write("%s " % str(element))
        file.write("\n")
    file.close()


