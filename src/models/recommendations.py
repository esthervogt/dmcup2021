#!pip install langdetect
#!pip install pyarrow
#!pip install fuzzywuzzy
#!pip install python-Levenshtein

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import fuzz
import pandas as pd
import numpy as np

items_path = '20210525_items_df.csv'
path = '20210525_header_items_df.csv'
mt_path = 'items_pp.csv'

df_items = pd.read_csv('items.csv', sep='|', quoting=csv.QUOTE_NONE, error_bad_lines=False)
df_transactions = pd.read_csv('transactions.csv', sep = '|')
df_evaluation = pd.read_csv('evaluation.csv', sep = '|')

items_df = pd.read_csv(items_path, delimiter=',', encoding='utf-8')
del items_df['description']
del items_df['recommended_age']
del items_df['number_pages']

header_df = pd.read_csv(path, lineterminator='\n')
del header_df['title']
del header_df['author']
del header_df['publisher']
del header_df['item_lang_en']

mt_df = pd.read_csv(mt_path, delimiter=',', encoding='utf-8')

items_df = pd.merge(items_df, header_df,  how='left', left_on=['headerID'], right_on = ['headerID'])
items_df['mt'] = mt_df['mt']


final_df = pd.DataFrame(columns=['book_id', 'model_id', 'team_id', 'recommendation_1', 'recommendation_2',
                                 'recommendation_3', 'recommendation_4', 'recommendation_5'])

x = items_df.copy()
for index, row in df_evaluation.iterrows():
    items_df = x.copy()
    print(items_df.shape)
    base_book = row['itemID']
    print(base_book)

    author = items_df['author'][items_df['itemID'] == base_book].to_string(index=False).lstrip()
    mtopic = items_df['mt'][items_df['itemID'] == base_book].to_string(index=False).strip(' []')
    lang = items_df['language'][items_df['itemID'] == base_book].to_string(index=False).lstrip()
    headerID = int(items_df['headerID'][items_df['itemID'] == base_book])

    items_df = filter_on_lang(items_df, lang)
    items_df.reset_index(drop=True, inplace=True)

    df_authorscores = items_df[['itemID', 'title', 'author']]
    df_authorscores['author_score'] = get_authorscores_fuzzy(df_authorscores, author)
    # df_authorscores

    df_topicscore = items_df[['headerID', 'itemID', 'title', 'mt']]
    df_topicscore['mtopic_score'] = get_mtopicscores(df_topicscore, mtopic)
    # df_topicscore
    # df_authorscores

    df_titlescores = items_df[['itemID', 'title', 'author']]
    # pd.set_option('display.max_rows', 500)
    df_titlescores['title_score'] = get_titlescores(df_titlescores, items_df, base_book)
    # df_titlescores

    result_transactions = recommend_based_on_transactions(df_transactions, df_items, base_book)
    # result_transactions

    pd.options.display.max_rows = 250
    pd.options.display.min_rows = 250

    result = get_totalscore(df_titlescores, df_authorscores, df_topicscore, result_transactions)
    result = result[result['headerID'] != headerID]
    result.drop_duplicates(subset='headerID', keep="first", inplace=True)
    result = result.sort_values(by='total_score', ascending=False)

    recommendations = result.iloc[0:5, :]

    final_df = final_df.append({'book_id': row['itemID'],
                                'model_id': 'first',
                                'team_id': 'dataminerz',
                                'recommendation_1': recommendations.iloc[0, 0],
                                'recommendation_2': recommendations.iloc[1, 0],
                                'recommendation_3': recommendations.iloc[2, 0],
                                'recommendation_4': recommendations.iloc[3, 0],
                                'recommendation_5': recommendations.iloc[4, 0]}, ignore_index=True)
    print(len(final_df))

final_df.to_csv('rec_final.csv')


def filter_on_lang(df_item, language):
    if len(df_item[df_item['language'] == lang]) > 5:

        print(len(df_item[df_item['language'] == lang]))
        return df_item[df_item['language'] == lang]

    else:

        print(len(df_item))
        return df_item

def get_authorscores_fuzzy(df_authorscores, author):
    count = 0
    for i in df_authorscores['itemID']:

        # print(df_authorscores['author'][count])
        # print(author)
        if pd.isnull(df_authorscores['author'][count]):
            df_authorscores.at[count, 'author_score'] = 0
        else:

            fuzzratio = fuzz.ratio(author, df_authorscores['author'][count]) / 100

            if (fuzz.partial_ratio(author, df_authorscores['author'][count]) / 100) > 0.70 and (fuzzratio) > 0.5:
                df_authorscores.at[count, 'author_score'] = 1 + fuzzratio
            else:
                df_authorscores.at[count, 'author_score'] = 0
            # print(fuzz.partial_ratio(author, df_authorscores['author'][count]))

        count += 1

    return df_authorscores['author_score']


def get_mtopicscores(df_mtopicscores, mtopic):
    count = 0
    for i in df_mtopicscores['itemID']:

        # df_mtopicscores.at[count, 'mtopic_score'] = SequenceMatcher(None, df_mtopicscores['mt'][count].strip(' []'), mtopic).ratio()
        # print(count)

        if pd.isnull(df_mtopicscores['mt'][count]):
            df_mtopicscores.at[count, 'mtopic_score'] = 0
        else:

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


def get_mtopicscores_emb(df_mtopicscores, emb):
    emb_items = df_mtopicscores['emb_cats']

    cos_similarity_topics = map(lambda x: cosine_similarity(emb, x), emb_items)

    output = list(cos_similarity_topics)

    return np.array(output).ravel()

def get_titlescores(df_titlescores, items_df, base_book):
    # # Creating tfidf-matrix
    tfidf_vectorizer = TfidfVectorizer()
    # print(df_titlescores)
    tfidf_items = tfidf_vectorizer.fit_transform((df_titlescores['title'].apply(lambda x: np.str_(x))))

    # # Retrieving book for recommendations
    # print(df_titlescores[items_df['itemID'] == base_book])
    test_tfidf = tfidf_items[items_df['itemID'] == base_book]

    # # Retrieving most similar books
    cos_similarity_tfidf = map(lambda x: cosine_similarity(test_tfidf, x), tfidf_items)
    output = list(cos_similarity_tfidf)
    # top = sorted(range(len(output)), key=lambda i: output[i], reverse=True)[:50]
    # scores = [output[i][0][0] for i in top]

    # result = get_recommendation(top, df_titlescores, df_titlescores['title'][items_df['itemID'] == base_book].to_string(), scores)
    # print(result)

    return np.array(output).ravel()


def get_totalscore(df1, df2, df3, df_trans):
    result = pd.DataFrame(
        columns=['itemID', 'headerID', 'title', 'author', 'mt', 'title_score', 'author_score', 'mtopic_score',
                 'trans_score', 'total_score'])

    result['itemID'] = df1.iloc[:, 0]
    result['headerID'] = df3.iloc[:, 0]
    result['title'] = df3.iloc[:, 2]
    result['author'] = df1.iloc[:, 2]
    result['mt'] = df3.iloc[:, 3]
    result['title_score'] = df1.iloc[:, 3]
    result['author_score'] = df2.iloc[:, 3]
    result['mtopic_score'] = df3.iloc[:, 4]
    result['trans_score'] = 0

    count = 0
    for i in df_trans:
        result['trans_score'][result['itemID'] == df_trans[count]] = 1

        count += 1

    result['total_score'] = result['title_score'] + result['author_score'] + result['mtopic_score'].astype(float) + \
                            result['trans_score'].astype(float)

    return result
