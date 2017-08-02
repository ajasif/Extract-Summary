import itertools
import networkx as nx
import nltk
import operator
import math
import Tkinter as tk
import tkFileDialog
import textwrap

'''
Takes in a single or multiple news or scholarly text articles
and extracts a summary. 

@author: Team 11
'''
def mymain():
    print "Enter 1 for single document summarization."
    print "Enter 2 for multiple document summarization."
    print "Enter 3 to quit."
    sum_model = input("input: ")
    if sum_model == 1:
        single_doc_summary()
    if sum_model == 2:
        multi_doc_summary()
    print "Summary created. Check file [summary.txt]."

'''
Summary based on one article.
'''
def single_doc_summary():
    testing_text = getTextFromFile()
    
    testing_sentences = getSentences(testing_text)
    
    print 'Processing...'
    
    graph = build_graph(testing_sentences)
    
    # Maps Sentence -> Value (Rank)
    sentence_ranks = nx.pagerank(graph, weight='weight') 
    
    # Sort most important to least important
    sentences = sorted(sentence_ranks, key=sentence_ranks.get,
                       reverse=True)

    summary_length = input("Enter summary length in sentences: ")
    
    if summary_length > len(sentences):
        summary_length = len(sentences)
        
    # Pulls sentences which were part of the testing set
    important_sentences = sentences[0:summary_length]
    
    # Order sentences as they appeared in testing set
    ordered_sentences = order_sentences(testing_sentences, important_sentences)
    writeToFile(ordered_sentences)

'''
Summary based on multiple articles.
'''            
def multi_doc_summary():
    num = input("Enter number of training documents: ")
    training_text = ""
    while num > 0:
        num = num - 1
        training_text = training_text + getTextFromFile()
    
    testing_text = getTextFromFile()
    
    training_sentences = getSentences(training_text)
    testing_sentences = getSentences(testing_text)
    merge_sentences = training_sentences + testing_sentences
    
    print 'Processing...'
    
    graph = build_graph(merge_sentences)
    
    # Maps Sentence -> Value (Rank)
    sentence_ranks = nx.pagerank(graph, weight='weight') 
    
    # Sort most important to least important
    sentences = sorted(sentence_ranks, key=sentence_ranks.get,
                       reverse=True)

    summary_length = input("Enter summary length in sentences: ")
    
    # Pulls sentences which were part of the testing set
    important_sentences = []
    count = 0;
    for sentence in sentences:
        if count >= summary_length:
            break
        if sentence in testing_sentences:
            important_sentences.append(sentence)
            count = count + 1
    
    # Order sentences as they appeared in testing set
    ordered_sentences = order_sentences(testing_sentences, important_sentences)
    writeToFile(ordered_sentences)


'''
Parses the text from a document into a list of sentences.
@param: String of text
'''
def getSentences(text):
    sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
    sentence_tokens = sent_detector.tokenize(text.strip())
    return sentence_tokens

'''
Prompts the user to select a text file.
@return: String of text from file
'''
def getTextFromFile():
    root = tk.Tk()
    root.withdraw()
    file_path = tkFileDialog.askopenfilename()
    with open(file_path) as f:
        return f.read().decode("utf-8")

'''
Writes summary to summary.txt file.
@param: 
'''    
def writeToFile(summary):
    f = open('summary.txt', 'w')
    f.write(textwrap.fill(' *'.join(summary)).encode('utf-8'))
    f.close()

'''
Returns a weight denoting how strongly connected two given 
sentences are. 
'''
def common_words(first, second):
    first_tokens = nltk.word_tokenize(first)
    second_tokens = nltk.word_tokenize(second)
    
    tagged_first = nltk.pos_tag(first_tokens) # (word, pos_tag)
    tagged_second = nltk.pos_tag(second_tokens)
    
    first_imporant = is_important(tagged_first)
    second_imporant = is_important(tagged_second)
    
    if len(first_imporant) <= 0 or len(second_imporant) <= 0: #no important words
        return 0
    else:
        common_words_count = 0; 
        for word in first_imporant:
            if word in second_imporant:
                common_words_count += 1
        
        # Reduces bias towards longer sentences. 
        denominator = (math.log10(len(first_imporant)) + math.log10(len(second_imporant)))
        if denominator == 0: 
            return 0
        else:
            common_words_count = float(common_words_count) / denominator
            return common_words_count

'''
Returns the stems of the important words from the given tagged tokens.
:param POS tagged tokens which make up a sentence. 
'''
def is_important(tagged_tokens):
    lmtzr = nltk.stem.wordnet.WordNetLemmatizer() # finds root form of word
    important_words = []
    for pair in tagged_tokens:
        pos = pair[1]
        if is_noun(pos):
            important_words.append(lmtzr.lemmatize(pair[0], nltk.corpus.wordnet.NOUN))
        if is_adjective(pos):
            important_words.append(lmtzr.lemmatize(pair[0], nltk.corpus.wordnet.ADJ))
    
    return important_words

'''
Returns if the given tag is a Noun. 
'''
def is_noun(tag):
    return tag in ['NN', 'NNS', 'NNP', 'NNPS']

'''
Returns if the given tag is an Adjective. 
'''
def is_adjective(tag):
    return tag in ['JJ', 'JJR', 'JJS']

'''
Returns a digraph graph where nodes are sentences and edges
between sentences represent how strongly correlated they. 
'''
def build_graph(sentences):
    gr = nx.Graph()
    gr.add_nodes_from(sentences)

    # Creates a pair for all possible pairings of nodes
    sentencePairs = list(itertools.combinations(sentences, 2))

    # add edges to the graph (weighted by number of common words)
    for sentencePair in sentencePairs:
        sentence_one = sentencePair[0]
        sentence_two = sentencePair[1]
        common_words_count = common_words(sentence_one, sentence_two)
        if common_words_count > 0: 
            gr.add_edge(sentence_one, sentence_two, weight=common_words_count)

    return gr
    
'''
Order sentences by original order in text. 
:param sentences; list of all sentences from text
:param important_sentences; list of sentences 
'''
def order_sentences(original_sentences, new_sentences):
    indexed_sentences = [];
    for n_sent in new_sentences:
        indexed_sentences.append((n_sent, original_sentences.index(n_sent)))
    indexed_sentences.sort(key=operator.itemgetter(1))
    
    result = []
    for indexed_sent in indexed_sentences:
        result.append(indexed_sent[0])
    return result

if __name__ == '__main__':
    mymain()