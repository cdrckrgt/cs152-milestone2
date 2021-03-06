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

attributes = ['TOXICITY', 'SEVERE_TOXICITY', 'IDENTITY_ATTACK', 'INSULT', 'PROFANITY', \
    'THREAT', 'SEXUALLY_EXPLICIT', 'INFLAMMATORY', 'OBSCENE', 'INCOHERENT', 'UNSUBSTANTIAL']

def eval_text(text, printResponse=False):
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
            attribute : {} for attribute in attributes
        },
        'doNotStore': True
    }
    response = requests.post(url, data=json.dumps(data_dict))
    response_dict = response.json()
    if printResponse:
        print("\"" + text + "\"")
        print(json.dumps(response_dict, indent=4))
    return response_dict


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
    while True:
        try:
            response = eval_text(line)
            score = response[u'attributeScores'][u'TOXICITY'][u'summaryScore'][u'value']
            break
        except KeyError:
            time.sleep(5)
    return {'TOXICITY' : score}


def intermediateScore(line, attributes=attributes):
    '''
    an intermediate classifier uses some mixture of different attributes
    '''
    score_dict = {}
    while True:
        try:
            response = eval_text(line)
            scores = 0
            for attribute in attributes:
                val = response[u'attributeScores'][attribute.decode('utf-8')][u'summaryScore'][u'value']
                if attribute in ['INCOHERENT', 'UNSUBSTANTIAL']:
                    val = 1 - val
                score_dict[attribute] = val
            break
        except KeyError:
            time.sleep(5)
    return score_dict

def advancedScore(line, addtributes=attributes):
    '''
    would do things like cross validation of attribute selection,
    learn best weighting of attributes for classification,
    maybe a more advanced sentiment classification/ensemble of different sentiment classifcations,
    all trained through supervised learning on the dataset

    alas, there is no way for us to do this here easily.
    '''
    return 0

def analyze(lines, nb_samples=50, scoringFunc=basicScore):
    '''
    loop through all lines in the file
    for each line, preprocess and then find toxicity
    collect stats for analysis at the end of the file
    '''
    # average scores records the average attribute score for each example
    #   will have length nb_samples or however many lines are in the file,
    #   whichever is less
    average_scores = []
    # aggregate_scores records the total attribute score for each attribute
    #   when divided by len(average_scores), gives the average attribute score
    #   across all examples
    aggregate_scores = {attribute : 0 for attribute in attributes}
    for line in lines:
        line = tokenize(preprocess(line))
        if line == '': continue
        score_dict = scoringFunc(line)
        avg = sum([v for k, v in score_dict.iteritems()]) / len(score_dict)
        average_scores.append(avg)
        for k, v in score_dict.iteritems():
            aggregate_scores[k] += v
        if len(average_scores) >= nb_samples:
            break
    print( stats.describe(average_scores) )
    nb_obs = len(average_scores)
    for k, v in aggregate_scores.iteritems():
        print('average {} score: {}'.format(k, v / nb_obs))
    y_hat = np.array(average_scores) > 0.5
    return y_hat.astype(int)


def show_metrics(y, y_hat):
    print('accuracy: ', metrics.accuracy_score(y, y_hat))
    print('precision: ', metrics.precision_score(y, y_hat))
    print('recall: ', metrics.recall_score(y, y_hat))
    print('f1: ', metrics.f1_score(y, y_hat))
    print('')
    

if __name__ == '__main__':

    with open('./data/gab_samples.txt') as f:
        print('gab scores')
        analyze(f.readlines(), scoringFunc=intermediateScore)

    with open('./data/twitter_samples.txt') as f:
        print('twitter scores')
        analyze(f.readlines(), scoringFunc=intermediateScore)

    misclass_df = pd.read_csv('./data/misclass.csv', delimiter=';')
    scraped_df = pd.read_csv('./data/scraped_misclass.csv', delimiter=';')
    combined_df = pd.concat([misclass_df, scraped_df])

    y_hat = analyze(scraped_df.text, scoringFunc=basicScore)
    y = scraped_df.label
    print('basicScore on scraped dataset')
    show_metrics(y, y_hat)

    y_hat = analyze(combined_df.text, scoringFunc=basicScore)
    y = combined_df.label
    print('basicScore on combined dataset')
    show_metrics(y, y_hat)

    y_hat = analyze(combined_df.text, scoringFunc=intermediateScore)
    y = combined_df.label
    print('intermediateScore on combined dataset')
    show_metrics(y, y_hat)

    y_hat = analyze(combined_df.text, scoringFunc=lambda x : intermediateScore(x, ['INFLAMMATORY']))
    y = combined_df.label
    print('intermediateScore, inflammatory on combined dataset')
    show_metrics(y, y_hat)

    mod_df = pd.read_csv('./data/modified_misclass.csv', delimiter=';')
    y_hat = analyze(mod_df.text, scoringFunc=intermediateScore)
    y = mod_df.label
    print('intermediateScore on modified combined dataset')
    show_metrics(y, y_hat)

    y_hat = analyze(mod_df.text, scoringFunc=lambda x : intermediateScore(x, ['INFLAMMATORY']))
    y = mod_df.label
    print('intermediateScore, inflammatory on modified combined dataset')
    show_metrics(y, y_hat)

    eval_text("i don't give a fuck how much it's snowing, i will brave a blizzard to go take care of you", printResponse=True)
    eval_text("that was fucking good!", printResponse=True)
