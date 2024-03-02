import collections

import numpy as np

import util


def get_words(message):
    """Get the normalized list of words from a message string.

    This function should split a message into words, normalize them, and return
    the resulting list. For splitting, you should split on spaces. For normalization,
    you should convert everything to lowercase.

    Args:
        message: A string containing an SMS message

    Returns:
       The list of normalized words from the message.
    """

    # *** START CODE HERE ***
    wordListOfList = [msgList.split(' ') for msgList in message]
    wordListOfList = [[word.lower() for word in wordList] for wordList in wordListOfList]
    return(wordListOfList)
    # *** END CODE HERE ***


def create_dictionary(messages):
    """Create a dictionary mapping words to integer indices.

    This function should create a dictionary of word to indices using the provided
    training messages. Use get_words to process each message.

    Rare words are often not useful for modeling. Please only add words to the dictionary
    if they occur in at least five messages.

    Args:
        messages: A list of strings containing SMS messages

    Returns:
        A python dict mapping words to integers.
    """

    # *** START CODE HERE ***
    wordListOfList = get_words(messages)
    wordCountDict = collections.defaultdict(int)
    wordDict = {}
    wordListOfDict = [collections.Counter(wordList) for wordList in wordListOfList]
    for msgDict in wordListOfDict:
        for word in msgDict:
            wordCountDict[word] += 1
    idxWord = 0
    for word in wordCountDict:
        if wordCountDict[word] >= 5:
            wordDict[word] = idxWord
            idxWord += 1
    return(wordDict)
    # *** END CODE HERE ***


def transform_text(messages, word_dictionary):
    """Transform a list of text messages into a numpy array for further processing.

    This function should create a numpy array that contains the number of times each word
    of the vocabulary appears in each message. 
    Each row in the resulting array should correspond to each message 
    and each column should correspond to a word of the vocabulary.

    Use the provided word dictionary to map words to column indices. Ignore words that
    are not present in the dictionary. Use get_words to get the words for a message.

    Args:
        messages: A list of strings where each string is an SMS message.
        word_dictionary: A python dict mapping words to integers.

    Returns:
        A numpy array marking the words present in each message.
        Where the component (i,j) is the number of occurrences of the
        j-th vocabulary word in the i-th message.
    """
    # *** START CODE HERE ***
    wordListOfList = get_words(messages)
    vocabSize = max(word_dictionary.values()) + 1
    # print(f'Vocabulary size: {vocabSize}')
    numMsg = len(wordListOfList)

    wordArr = np.zeros((numMsg, vocabSize))
    for idxMsg, msg in enumerate(wordListOfList):
        for word in msg:
            if word in word_dictionary:
                wordArr[idxMsg, word_dictionary[word]] += 1
    return(wordArr)

    # *** END CODE HERE ***


def fit_naive_bayes_model(matrix, labels):
    """Fit a naive bayes model.

    This function should fit a Naive Bayes model given a training matrix and labels.

    The function should return the state of that model.

    Feel free to use whatever datatype you wish for the state of the model.

    Args:
        matrix: A numpy array containing word counts for the training data
        labels: The binary (0 or 1) labels for that training data

    Returns: The trained model
    """

    # *** START CODE HERE ***
    numMsg, vocabSize = matrix.shape
    phi_spam = sum(labels) / numMsg
    spamMatrix = np.array([matrix[idxMsg] for idxMsg in range(numMsg) if labels[idxMsg] == 1])
    hamMatrix = np.array([matrix[idxMsg] for idxMsg in range(numMsg) if labels[idxMsg] == 0])
    spamMatrix = np.vstack([spamMatrix, np.ones(vocabSize)])
    hamMatrix = np.vstack([hamMatrix, np.ones(vocabSize)])

    wordCountInSpam = spamMatrix.sum(0)
    wordCountInHam = hamMatrix.sum(0)
    totalCountSpam = sum(wordCountInSpam)
    totalCountHam = sum(wordCountInHam)
    phi_vocabSpam = wordCountInSpam / totalCountSpam
    phi_vocabHam = wordCountInHam / totalCountHam
    return([phi_spam, phi_vocabSpam, phi_vocabHam])

    # *** END CODE HERE ***


def predict_from_naive_bayes_model(model, matrix):
    """Use a Naive Bayes model to compute predictions for a target matrix.

    This function should be able to predict on the models that fit_naive_bayes_model
    outputs.

    Args:
        model: A trained model from fit_naive_bayes_model
        matrix: A numpy array containing word counts

    Returns: A numpy array containg the predictions from the model
    """
    # *** START CODE HERE ***
    phi_spam, phi_vocabSpam, phi_vocabHam = model
    predList = []
    for obs in matrix:
        scoreSpam, scoreHam = phi_spam, 1 - phi_spam
        for idxWord, countWord in enumerate(obs):
            scoreSpam *= phi_vocabSpam[idxWord] ** countWord
            scoreHam *= phi_vocabHam[idxWord] ** countWord
        predList.append(1 if scoreSpam >= scoreHam else 0)
    return(np.array(predList))
    # *** END CODE HERE ***


def get_top_five_naive_bayes_words(model, dictionary):
    """Compute the top five words that are most indicative of the spam (i.e positive) class.

    Ues the metric given in part-c as a measure of how indicative a word is.
    Return the words in sorted form, with the most indicative word first.

    Args:
        model: The Naive Bayes model returned from fit_naive_bayes_model
        dictionary: A mapping of word to integer ids

    Returns: A list of the top five most indicative words in sorted order with the most indicative first
    """
    # *** START CODE HERE ***
    _, phi_vocabSpam, phi_vocabHam = model
    wordList = []
    for word, idxWord in dictionary.items():
        # score = np.log(phi_vocabSpam[idxWord]) - np.log(phi_vocabHam[idxWord])
        score = np.log(phi_vocabSpam[idxWord] / phi_vocabHam[idxWord])
        wordList.append((word, score))
    wordList.sort(key = lambda x:-x[1], )
    return(wordList[:5])





    # *** END CODE HERE ***


def main():
    train_messages, train_labels = util.load_spam_dataset('spam_train.tsv')
    val_messages, val_labels = util.load_spam_dataset('spam_val.tsv')
    test_messages, test_labels = util.load_spam_dataset('spam_test.tsv')

    dictionary = create_dictionary(train_messages)

    print('Size of dictionary: ', len(dictionary))

    util.write_json('spam_dictionary', dictionary)

    train_matrix = transform_text(train_messages, dictionary)

    np.savetxt('spam_sample_train_matrix', train_matrix[:100,:])

    val_matrix = transform_text(val_messages, dictionary)
    test_matrix = transform_text(test_messages, dictionary)

    naive_bayes_model = fit_naive_bayes_model(train_matrix, train_labels)

    naive_bayes_predictions = predict_from_naive_bayes_model(naive_bayes_model, test_matrix)

    np.savetxt('spam_naive_bayes_predictions', naive_bayes_predictions)

    naive_bayes_accuracy = np.mean(naive_bayes_predictions == test_labels)

    print('Naive Bayes had an accuracy of {} on the testing set'.format(naive_bayes_accuracy))

    top_5_words = get_top_five_naive_bayes_words(naive_bayes_model, dictionary)

    print('The top 5 indicative words for Naive Bayes are: ', top_5_words)

    util.write_json('spam_top_indicative_words', top_5_words)

if __name__ == "__main__":
    main()
