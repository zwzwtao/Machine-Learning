import numpy as np
import re
import random


def text_parse(input_string):
    """
    text preprocess: parse all the emails to a list of words
    """
    list_of_tokens = re.split(r'\W+', input_string)
    return [tok.lower() for tok in list_of_tokens if len(list_of_tokens) > 2]


def creat_vocabulary(doc_list):
    """
    return a vocabulary of all the unique words
    """
    vocab_set = set([])
    for doc in doc_list:
        vocab_set = vocab_set | set(doc)
    return list(vocab_set)


def set_of_word2vec(vocab_list, input_set):
    """
    construct a list that shows how many words from a input document appears in the vocabulary
    if appears, set the relevant position to 1, others 0
    """
    # the length should be the same as vocabulary, not the document
    returned_vec = [0] * len(vocab_list)
    for word in input_set:
        if word in vocab_list:
            returned_vec[vocab_list.index(word)] = 1

    return returned_vec


def train_nb(training_matrix, training_labels):
    """
    training a Naive Bayes classifier
    :returns p_word_spam, p_word_ham: probabilities of each word appearing in spam/ham
             spam_percentage: the percentage of spam in total training samples
    """
    num_samples = len(training_matrix)
    # each vector's length from training matrix is the same as vocabulary size
    vocab_size = len(training_matrix[0])
    spam_percentage = sum(training_labels) / float(num_samples)
    # here we don't use np.zeros to avoid (0 * p(..)) then the result will not become zero
    # compute p(word|ham) and p(word|spam)
    words_in_spam = np.ones(vocab_size)
    words_in_ham = np.ones(vocab_size)  # number of words appears in ham
    denom_spam = 2
    denom_ham = 2

    for i in range(num_samples):
        if training_labels[i] == 1:
            words_in_spam += training_matrix[i]
            denom_spam += sum(training_matrix[i])
        else:
            words_in_ham += training_matrix[i]
            denom_ham += sum(training_matrix[i])

    # notice: words_in_spam is a vector, thus p_spam, p_ham is a vector which
    # stores all the probabilities of each word appearing in spam/ham
    p_word_spam = np.log(words_in_spam / denom_spam)
    p_word_ham = np.log(words_in_ham / denom_ham)

    return p_word_spam, p_word_ham, spam_percentage


def classify_nb(word_vec, p_word_spam, p_word_ham, spam_percentage):
    """
    :param word_vec: a word vector for classifying e.g. [0, 1, 1, 0, 1, 0, 0, 1]
    :param p_word_spam: a vector stores all the probabilities of each word appearing in spam
    :param p_word_ham: a vector stores all the probabilities of each word appearing in ham
    :param spam_percentage: (num_spam / num_total_documents)
    :return: 1 spam
             0 ham
    """
    p_spam = np.log(spam_percentage) + sum(word_vec * p_word_spam)
    p_ham = np.log(1.0 - spam_percentage) + sum(word_vec * p_word_ham)
    if p_spam > p_ham:
        return 1
    else:
        return 0


def spam():
    doc_list = []  # emails
    label_list = []  # labels of emails
    for i in range(1, 26):
        word_list = text_parse(open('email/spam/%d.txt' % i, 'r').read())
        doc_list.append(word_list)
        label_list.append(1)  # 1 means spam

        word_list = text_parse(open('email/ham/%d.txt' % i, 'r').read())
        doc_list.append(word_list)
        label_list.append(0)  # 0 means ham

    # vocabulary is vocab_list
    vocab_list = creat_vocabulary(doc_list)
    training_set = list(range(50))  # stores all 50 indices
    test_set = []  # to store 10 test indices
    for i in range(10):
        rand_index = int(random.uniform(0, len(training_set)))
        test_set.append(training_set[rand_index])
        # remove the index from training set, thus the actual size of training set will be 40
        del (training_set[rand_index])

    training_matrix = []
    training_labels = []
    for doc_index in training_set:
        # construct training data
        training_matrix.append(set_of_word2vec(vocab_list, doc_list[doc_index]))
        training_labels.append(label_list[doc_index])

    p_word_spam, p_word_ham, spam_percentage = train_nb(np.array(training_matrix), np.array(training_labels))

    error_count = 0
    for doc_index in test_set:
        # convert current document to word vector
        test_word_vec = set_of_word2vec(vocab_list, doc_list[doc_index])
        # do classification on the document
        if classify_nb(np.array(test_word_vec), p_word_spam, p_word_ham, spam_percentage) != label_list[doc_index]:
            error_count += 1
    print('wrongly classified samples: ' + str(error_count) + ' samples')


if __name__ == '__main__':
    spam()
