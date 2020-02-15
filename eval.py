'''
This file contains code to run the Perspective comment analyzer
on a snippet of text.
'''
import requests
import json
import time
import scipy.stats as stats
import re
import pandas as pd
import numpy as np
import sklearn.metrics as metrics

def eval_text(text):
# This is the URL which Perspective API requests go to.
    PERSPECTIVE_URL = 'https://commentanalyzer.googleapis.com/v1alpha1/comments:analyze'
    key = "AIzaSyClfD1Tdk7gyO-1tonIDjeoQXp72g-jWmg";
    url = PERSPECTIVE_URL + '?key=' + key
    data_dict = {
        'comment': {'text': text},
        'languages': ['en'],
        # This dictionary specifies which attributes you care about. You are 
        # welcome to (and should) add more.
        # The full list can be found at: 
        # https://github.com/conversationai/perspectiveapi/blob/master/2-api/models.md
        'requestedAttributes': { 
            'TOXICITY': {},
            'SEVERE_TOXICITY': {},
            'IDENTITY_ATTACK': {},
            'INSULT': {},
            'PROFANITY': {},
            'THREAT': {},
            'SEXUALLY_EXPLICIT': {},
            'INFLAMMATORY': {},
            'OBSCENE': {},
            'INCOHERENT': {},
            'UNSUBSTANTIAL': {},
        },
        'doNotStore': True
    }
    response = requests.post(url, data=json.dumps(data_dict))
    response_dict = response.json()
    # Print the entire response dictionary.
    # print("\"" + text + "\"")
    # print(json.dumps(response_dict, indent=4))
    return response_dict

# Here you can add code to evaluate particular messages.




# preprocessing taken from https://github.com/t-davidson/hate-speech-and-offensive-language/blob/master/src/Automated%20Hate%20Speech%20Detection%20and%20the%20Problem%20of%20Offensive%20Language%20Python%203.6.ipynb

def preprocess(text_string):
    """
    Accepts a text string and replaces:
    1) urls with URLHERE
    2) lots of whitespace with one instance
    3) mentions with MENTIONHERE

    This allows us to get standardized counts of urls and mentions
    Without caring about specific people mentioned
    """
    space_pattern = '\s+'
    giant_url_regex = ('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|'
        '[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    mention_regex = '@[\w\-]+'
    # sometimes emojis aren't encoded properly, this takes care of that
    unicode_regex = '&.*?;'
    parsed_text = re.sub(space_pattern, ' ', text_string)
    parsed_text = re.sub(giant_url_regex, '', parsed_text)
    parsed_text = re.sub(mention_regex, '', parsed_text)
    parsed_text = re.sub(unicode_regex, '', parsed_text)
    return parsed_text

def tokenize(line):
    """Same as tokenize but without the stemming"""
    line = " ".join(re.split("[^a-zA-Z.,!?1-9]*", line.lower())).strip()
    return line

def basicScore(line):
    '''
    a basic classifier uses just toxicity score
    '''
    i = 1
    while True:
        try:
            response = eval_text(line)
            score = response[u'attributeScores'][u'TOXICITY'][u'summaryScore'][u'value']
            break
        except KeyError:
            print('resource exhausted, trying again...')
            time.sleep(i)
            i += 1
    return score

attributes = ['TOXICITY', 'SEVERE_TOXICITY', 'IDENTITY_ATTACK', 'INSULT', 'PROFANITY', \
    'THREAT', 'SEXUALLY_EXPLICIT', 'INFLAMMATORY', 'OBSCENE', 'INCOHERENT', 'UNSUBSTANTIAL']

def intermediateScore(line, attributes=attributes):
    '''
    an intermediate classifier uses an average of lots of attributes
    '''
    i = 1
    while True:
        try:
            response = eval_text(line)
            scores = 0
            for attribute in attributes:
                val = response[u'attributeScores'][attribute.decode('utf-8')][u'summaryScore'][u'value']
                if attribute in ['INCOHERENT', 'UNSUBSTANTIAL']:
                    val = 1 - val
                scores += val
            score = scores / len(attributes)
            break
        except KeyError:
            print('resource exhausted, trying again...')
            time.sleep(i)
            i += 1
    return score

def advancedScore(line, addtributes=attributes):
    '''
    would do things like cross validation of attribute selection,
    learn best weighting of attributes for classification,
    maybe a more advanced sentiment classification/ensemble of different sentiment classifcations
    '''
    return 0

def analyze(lines, nb_samples=100, scoringFunc=basicScore):
    '''
    loop through all lines in the file
    for each line, preprocess and then find toxicity
    collect stats for analysis at the end of the file
    '''
    scores = []
    for line in lines:
        line = tokenize(preprocess(line))
        score = scoringFunc(line)
        print(line)
        print(score)
        scores.append(score)
        if len(scores) >= nb_samples:
            break
    print( stats.describe(scores) )
    y_hat = np.array(scores) > 0.5
    return y_hat.astype(int)


def show_metrics(y, y_hat):
    print('accuracy: ', metrics.accuracy_score(y, y_hat))
    print('precision: ', metrics.precision_score(y, y_hat))
    print('recall: ', metrics.recall_score(y, y_hat))
    print('f1: ', metrics.f1_score(y, y_hat))
    print('')
    

if __name__ == '__main__':

    # with open('./data/gab_samples.txt') as f:
    #     analyze(f.readlines(), printToxicity=True)

    # with open('./data/twitter_samples.txt') as f:
    #     analyze(f.readlines(), printToxicity=True)

    # misclass_df = pd.read_csv('./data/misclass.csv', delimiter=';')

    # y_hat = analyze(misclass_df.text)
    # y = misclass_df.label

    # print('gab and twitter misclasses')
    # # we chose these misclasses to be wrong, so this is useless
    # show_metrics(y, y_hat)

    # scraped_df = pd.read_csv('./data/scraped_misclass.csv', delimiter=';')

    # y_hat = analyze(scraped_df.text)
    # y = scraped_df.label

    # print('scraped misclasses')
    # # we chose these misclasses to be wrong, so this is useless
    # show_metrics(y, y_hat)


    mod_df = pd.read_csv('./data/modified_misclass.csv', delimiter=';')
    y_hat = analyze(mod_df.text, scoringFunc=intermediateScore)
    y = mod_df.label
    # now ideally this is good, since we modified the text to be more easily classified
    show_metrics(y, y_hat)
