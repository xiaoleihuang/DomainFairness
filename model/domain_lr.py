# domain adaption + adversarial training

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import utils


def domain_lr(params):

    pass


if __name__ == '__main__':
    review_dir = '../data/review/'
    hate_speech_dir = '../data/hatespeech/'
    data_list = [
        ['review_amazon_english', review_dir + 'amazon/amazon.tsv', 'english'],
        ['review_yelp-hotel_english', review_dir + 'yelp_hotel/yelp_hotel.tsv', 'english'],
        ['review_yelp-rest_english', review_dir + 'yelp_rest/yelp_rest.tsv', 'english'],
        ['review_twitter_english', review_dir + 'twitter/twitter.tsv', 'english'],
        ['review_trustpilot_english', review_dir + 'trustpilot/united_states.tsv', 'english'],
        ['review_trustpilot_french', review_dir + 'trustpilot/france.tsv', 'french'],
        ['review_trustpilot_german', review_dir + 'trustpilot/german.tsv', 'german'],
        ['review_trustpilot_danish', review_dir + 'trustpilot/denmark.tsv', 'danish'],
        ['hatespeech_twitter_english', hate_speech_dir + 'english/corpus.tsv', 'english'],
        ['hatespeech_twitter_spanish', hate_speech_dir + 'spanish/corpus.tsv', 'spanish'],
        ['hatespeech_twitter_italian', hate_speech_dir + 'italian/corpus.tsv', 'italian'],
        ['hatespeech_twitter_portuguese', hate_speech_dir + 'portuguese/corpus.tsv', 'portuguese'],
        ['hatespeech_twitter_polish', hate_speech_dir + 'polish/corpus.tsv', 'polish'],
    ]

    for data_entry in data_list:
        parameters = {
            'dname': data_entry[0],
            'dpath': data_entry[1],
            'lang': data_entry[2],
        }


    pass
