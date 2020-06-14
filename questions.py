import nltk
import sys
import os
import string
import math

FILE_MATCHES = 1
SENTENCE_MATCHES = 1


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    contents = {}
    for filename in os.listdir(directory):
        with open(os.path.join(directory, filename)) as f:
            contents[filename] = f.read()
    return contents


def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    initial_word_list = nltk.word_tokenize(document)
    final_word_list = []
    for word in initial_word_list:
        word = word.lower()
        if word not in string.punctuation and word not in nltk.corpus.stopwords.words("english"):
            final_word_list.append(word)
    return final_word_list

def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    word_idf = {}
    total_docs = len(documents)
    for document in documents:
        for word in documents[document]:
            if word in word_idf:
                if not document in word_idf[word]:
                    word_idf[word].append(document)
            else:
                word_idf[word] = [document]
    for word in word_idf:
        word_idf[word] = math.log(total_docs / len(word_idf[word]))
    return word_idf
    

def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    file_rank = {}
    ranked_list = []
    counter = 0
    for document in files:
        rank = 0
        for word in query:
            rank += files[document].count(word) * idfs[word]
        file_rank[document] = rank
    file_rank = {k: v for k, v in sorted(file_rank.items(), key=lambda item: item[1], reverse=True)}
    for document in file_rank:
        if counter == n:
            break
        ranked_list.append(document)
        counter += 1
    return ranked_list

def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    sentence_rank = {}
    ranked_list = []
    counter = 0
    for sentence in sentences:
        rank = 0
        query_term_freq = 0
        for word in query:
            query_term_freq += sentences[sentence].count(word)
            if word in sentences[sentence]:
                rank += idfs[word]
        sentence_rank[sentence] = (rank, query_term_freq / len(sentences[sentence]))
    sentence_rank = {k: v for k, v in sorted(sentence_rank.items(), key=lambda item: (item[1][0], item[1][1]), reverse=True)}
    for document in sentence_rank:
        if counter == n:
            break
        ranked_list.append(document)
        counter += 1
    return ranked_list

if __name__ == "__main__":
    main()
