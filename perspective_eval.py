'''
This file contains code to run the Perspective comment analyzer
on a snippet of text.
'''
import requests
import json
import time
import scipy.stats as stats

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
        'requestedAttributes': { 'TOXICITY': {} },
        'doNotStore': True
    }
    response = requests.post(url, data=json.dumps(data_dict))
    response_dict = response.json()
    # Print the entire response dictionary.
    # print("\"" + text + "\"")
    # print(json.dumps(response_dict, indent=4))
    return response_dict

# Here you can add code to evaluate particular messages.

with open('./gab_samples_section.txt') as f:
    scores = []
    for line in f:
        time.sleep(1) # force sleep to limit requests to 1 per second
        try:   
            response = eval_text(line)
            toxicity_score = response[u'attributeScores'][u'TOXICITY'][u'summaryScore'][u'value']
            scores.append(toxicity_score)
        except KeyError:
            print('resource exhausted...')
        if len(scores) > 100:
            break
    print( stats.describe(scores) )
