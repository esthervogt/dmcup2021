from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import pandas as pd
import numpy as np


def get_authorscores(df_authorscores, author):
    count = 0
    for i in df_authorscores['itemID']:

        # if pd.isnull(df_authorscores['author'][count]):
        #    df_authorscores.at[count, 'author_score'] = 0
        # else:
        #    df_authorscores.at[count, 'author_score'] = SequenceMatcher
        #    (None, author_scores['author'][count], author).ratio()

        if df_authorscores['author'][count] == author:
            df_authorscores.at[count, 'author_score'] = 1
        else:
            df_authorscores.at[count, 'author_score'] = 0
        count += 1

    return df_authorscores['author_score']


def get_mtopicscores(df_mtopicscores, mtopic):
    count = 0
    for i in df_mtopicscores['itemID']:

        # df_mtopicscores.at[count, 'mtopic_score'] = SequenceMatcher(None, df_mtopicscores['mt'][count]
        # .strip(' []'), mtopic).ratio()

        if df_mtopicscores['mt'][count].strip(' []') == mtopic:
            df_mtopicscores.at[count, 'mtopic_score'] = 1
        elif df_mtopicscores['mt'][count].strip(' []')[:3] == mtopic[:3]:
            df_mtopicscores.at[count, 'mtopic_score'] = 0.7
        elif df_mtopicscores['mt'][count].strip(' []')[:2] == mtopic[:2]:
            df_mtopicscores.at[count, 'mtopic_score'] = 0.3
        else:
            df_mtopicscores.at[count, 'mtopic_score'] = 0
        count += 1

    return df_mtopicscores['mtopic_score']

def get_stopicscores(df_subtopics, stopic):

    return 0

def get_publisherscores(df_subtopics, stopic):

    return 0

def get_titlescores(df_titlescores, items_df, base_book):
    # # Creating tfidf-matrix
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_items = tfidf_vectorizer.fit_transform((df_titlescores['title']))

    # # Retrieving book for recommendations
    # print(df_titlescores[items_df['itemID'] == base_book])
    test_tfidf = tfidf_items[items_df['itemID'] == base_book]

    # # Retrieving most similar books
    cos_similarity_tfidf = map(lambda x: cosine_similarity(test_tfidf, x), tfidf_items)
    output = list(cos_similarity_tfidf)
    # top = sorted(range(len(output)), key=lambda i: output[i], reverse=True)[:50]
    # scores = [output[i][0][0] for i in top]

    # result = get_recommendation(top, df_titlescores, df_titlescores['title'][items_df['itemID']
    # == base_book].to_string(), scores)
    # print(result)

    return np.array(output).ravel()
