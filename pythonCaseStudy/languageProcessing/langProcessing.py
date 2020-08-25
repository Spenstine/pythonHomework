from collections import Counter
import string
'''
use os to navigate paths
read a couple of books in a given directory
use pandas to tabulate the books data
use matplotlib.plt to visualize the books data
ps: books should be purely text for easy reading by the program
'''

def count_words(text):
    '''
    return the frequency of words in a string.
    text: string
    return: dict(word -> count)
    '''
    wordList = []
    #word processing
    for word in text.split():
        word = word.strip(string.punctuation)
        wordList.append(word.lower())

    wordCount = Counter(wordList)
    return wordCount

def read_book(finput):
    '''
    finput: file input
    return: large string with removed special characters
    '''
    with open(finput, 'r', encoding = 'utf8') as fin:
        wordstring = fin.read()
        wordstring = wordstring.replace('\n', '').replace('\r', '').lower()
        wordstring = wordstring.replace('-', ' ').replace('_', ' ')
    return wordstring


def word_stats(wordDict):
    uniqueWords = len(wordDict)
    count = sum(wordDict.values())
    return uniqueWords, count



text = 'I love spencer\'s books.\n He is a happy man. fuck you bitch! bitch.'
wordDict = count_words(text)
uniqueWords, count = word_stats(wordDict)
print(uniqueWords, count)