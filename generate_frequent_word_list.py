import preprocess_helper


class Config:

    def __init__(self):
        # default constructor
        self.input_file_name = 'data/sentences.train'
        self.k_frequent_words = 20000 - 4
        self.frequent_words_file_name = 'data/k_frequent_words.txt'


def main():
    config = Config()
    top_k_words = preprocess_helper.generate_top_k_words(config.input_file_name, config.k_frequent_words)
    top_k_words.extend(['<unk>', '<pad>', '<bos>', '<eos>'])
    preprocess_helper.write_list_to_file(top_k_words, config.frequent_words_file_name)


if __name__ == "__main__":
    main()
