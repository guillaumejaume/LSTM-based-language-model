import numpy as np

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
    if number_of_words <= max_sent_length:
      sentence = '<bos> ' + raw_sentence.rstrip() + ' ' + '<pad> ' * (max_sent_length-number_of_words) + '<eos>'
      sentence = sentence.split(' ')
      # word2index
      sentence_with_indices = [vocab[word] for word in sentence]
      sentences_with_indices.append(sentence_with_indices)
      label_with_indices = sentence_with_indices[1:]
      labels_with_indices.append(label_with_indices)
  return np.asarray(sentences_with_indices), np.asarray(labels_with_indices)

def replace_unknown_words(input_sentences, frequent_words):
  """ replace all the words that don't belong to
    a list of known words with the token <unk>
    Parameters:
    -----------
    input_sentences: list of string
    list of sentences, where each word is already separated by a space char
    frequent_words: list of string
    list of the common words
    
    Returns:
    --------
    output_sentences: list of string
    sentences where each unknown word was replaced by <unk>
    """
  # argsort all the words from each sentence
  all_words = []
  for sentence in input_sentences:
    all_words.extend(sentence.split())
    all_words.extend('\n')
  indices = argsort(all_words)

  # replace by <unk> when necessary
  current_word = ''
  sentence = ''
  replace = False
  for idx in indices:
    if not all_words[idx] == '\n':
      if current_word == all_words[idx]:
        if replace:
          all_words[idx] = '<unk>'
      else:
        replace = False
        current_word = all_words[idx]
        if not current_word in frequent_words:
          all_words[idx] = '<unk>'
          replace = True

  # reconstruct the sentences
  output_sentences = []
  sentence = ''
  for word in all_words:
    if word == '\n':
      output_sentences.append(sentence)
      sentence = ''
    else:
      sentence += word
      sentence += ' '

  return output_sentences

def load_frequent_words(frequent_word_filename):
  frequent_words = load_raw_data(frequent_word_filename)
  frequent_words = [word.rstrip() for word in frequent_words]
  frequent_words.extend(['<bos>','<eos>','<unk>','<pad>'])
  return frequent_words

def load_and_process_data(filename, vocab, frequent_words, max_sent_length):
  raw_data = load_raw_data(filename)
  data = replace_unknown_words(raw_data, frequent_words)
  data, labels = add_tokens_to_sentences(data, vocab, max_sent_length)
  print('- Number of sentences loaded: ', len(data))
  return data, labels
