import pandas as pd
import time as tm
from keybert import KeyBERT

# Coding portion for Dr. Chancellor's research interview
# Written by Leor Porat, information I used to help write my code is as follows:
# Pandas documentation: https://pandas.pydata.org/docs/
# Time documentation: https://docs.python.org/3/library/time.html

# Total number of posts
def get_num_posts(filename):
    df = pd.read_csv(filename)
    return len(df)

# Total number of unique authors
def remove_duplicate_authors(filename):
    df = pd.read_csv(filename)
    df.drop_duplicates(subset='author', keep='first', inplace=True)
    return len(df)

# Average post length
# ASSUMPTIONS: Only selftext is included in the word count (not the title), posts with just a title (empty posts) don't count 
def get_average_post_length(filename):

    # Sets up dataframe and cleans up data
    df = pd.read_csv(filename) # Reads csv into dataframe
    df.dropna(axis=0, how='any', subset='selftext', inplace=True) # Removes rows with empty selftext
    df = df.get('selftext') # Gets series containing just selftext
    total_num_words = 0
    
    # Loops over each post, getting the word count and adding it to a sum
    for post in df:
        pd_post = pd.Series([post], dtype='string')
        a = pd_post.str.split() # Separates words in Series
        total_num_words += len(a[0])

    return total_num_words/len(df) # Gets average by dividing total words by the number of posts

# Date range of dataset (finds max & min)
def print_date_range(filename):

    # Sets up dataframe and clears bad data
    df = pd.read_csv(filename)
    df.dropna(axis=0, how='any', subset='title', inplace=True) # Removes rows with empty title (get rid of bad data)
    df = df.get('created_utc')

    min_time = df[0]
    max_time = df[0]
    for time in df:
        if time > max_time: # if the current time is greater than the maximum time
            max_time = time
        if time < min_time: # if the current time is before the minimum time
            min_time = time
    print("MIN TIME:", tm.asctime(tm.gmtime(int(min_time))))
    print("MAX TIME:", tm.asctime(tm.gmtime(int(max_time))))

# Helper function for getting 20 most important words. Gets frequency of all words throughout the data and stores them in a dictionary
def get_word_freq(filename):

    # Sets up dataframe and cleans up data
    df = pd.read_csv(filename)
    df.dropna(axis=0, how='any', subset='title', inplace=True) # Removes rows with empty title
    df.dropna(axis=0, how='any', subset='selftext', inplace=True) # Removes rows with empty selftext
    df = df.get('selftext')

    word_freq = {} # Will store words & their corresponding frequency

    # Loops over posts and adds frequency of each word to overall dictionary
    for post in df:
        pd_post = pd.Series([post], dtype='string')
        post_list = pd_post.str.split() # Separates words in Series
        
        for word in post_list[0]:
            # Remove punctuation at the end of word
            if not(word[len(word)-1].isalpha()):
                word = word[0:len(word)-1]
            word = word.lower()
            # If the word is already in the dictionary
            if word in word_freq:
                word_freq[word] = int(word_freq[word])+1
            else: # If the word needs to be added to the dictionary
                word_freq[word] = 1
    return word_freq

# Gets top 20 most important words (using frequency), filters out most conjunctions & connectives
# https://stackoverflow.com/questions/613183/how-do-i-sort-a-dictionary-by-value?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
def get_top_20_words(words):
    top_20 = []

    # Sorts dictionary by values
    new_dict = dict(sorted(words.items(), key=lambda item: item[1], reverse=True))

    # Filters out common connectives/conjunctions from words
    conjunctives = ['i', 'and', 'to', 'the', 'a', 'is', 'im', 'with', 'in', 'but', 'do', 'of', 'this', 'like', 'not', 'who', 'what', 'so', 'i\'m', 'there', 'just', '', 'it', 'about', 'am', 'when', 'where', 'dont', 'for', 'nor', 'or', 'yet', 'so', 'as', 'on', 'then', 'me', 'my', 'myself', 'was', 'if', 'i\'m', 'i’ve', 'have', 'that', 'at', 'up', 'get', 'don’t', 'don\'t', 'no', 'he', 'she'
                    , 'now', 'how', 'i’m', 'all', 'you', 'be', 'are', 'one', 'can', 'had', 'her', 'they', 'been', 'know', 'want', 'out', 'go', 'even', 'them', 'will', 'i\'ve', 'from', 'it\'s', 'has', 'much', 'more', 'an', 'we', 'some', 'would', 'any', 'back', 'going', 'it’s', 'make', 'always', 'still']
    
    # Loops until 20 words are added to the list
    counter = 0
    for item in new_dict:
        if counter == 20:
            break
        if item not in conjunctives:
            top_20.append(item)
            counter += 1
    return top_20


# DOESN'T WORK | Uses external library to compute values of importance in words
def compile_keywords_keybert(filename):

    # Sets up dataframe and clears bad data
    df = pd.read_csv(filename)
    df.dropna(axis=0, how='any', subset='title', inplace=True) # Removes rows with empty title
    df.dropna(axis=0, how='any', subset='selftext', inplace=True) # Removes rows with empty selftext
    df = df.get('selftext')

    words_importance = {}

    # Loops over post, adding importance values to overall dictionary
    for post in df:
        word_list = keybert_word_extraction(post)
        for tuple in word_list:
            if tuple[0] in words_importance:
                words_importance[tuple[0]] += tuple[1]
            else:
                words_importance[tuple[0]] = tuple[1]
    return words_importance

# DOESN'T WORK | Helper function to extract keywords using external library 
def keybert_word_extraction(text):
    kb = KeyBERT()
    keywords = kb.extract_keywords(text, keyphrase_ngram_range=(1, 1))
    return keywords


def print_data(filename):
    print()
    print('**pandas dataframe length (no changes)**')
    print()
    print('number of posts:', get_num_posts(filename))
    print()
    print('**pandas dataframe info (removing duplicate authors)**')
    print()
    print('number of unique authors:', remove_duplicate_authors(filename))
    print()
    print("**average length per post**")
    print()
    print(get_average_post_length(filename))
    print()
    print('**date range**')
    print()
    print_date_range(filename)
    print()
    print('**top 20 most important words (list)**')
    print()
    words = get_word_freq(filename)
    print('20 words:', get_top_20_words(words))

if __name__ == '__main__':
    print_data('depression-sampled.csv')