# Author: Esther Vogt
# Creation Date: 18.04.2021
# Purpose: Get a first understanding of the task & corresponding data

# Notes on DMC task:
# - create recommendation model based on
#   (1) historical transactions
#   (2) item features
# --> MARKET BASKET ANALYSIS
# - model should return 5 best recommendations for any given product (itemID | rec_1 | rec_2 | rec_3 | rec_4 | rec_5

# Useful Links:
# - https://medium.com/swlh/a-tutorial-about-market-basket-analysis-in-python-predictive-hacks-497dc6e06b27

########################################################################################################################
# Imports & Settings
########################################################################################################################
import pandas as pd
from collections import Counter
import seaborn as sns
import numpy as np

########################################################################################################################
# Global Variables
########################################################################################################################

# source data file paths
transactions_path = 'tempData/sourceData/transactions.csv'
evaluation_path = 'tempData/sourceData/evaluation.csv'
items_path = 'tempData/sourceData/items.csv'
subject_cats_0_path = 'tempData/sourceData/subject_cats_0.csv'

########################################################################################################################
# Load Data
########################################################################################################################

# Load the dmc source data
# - clicks/baskets/order over a period of 3M
# - rows: one transaction for single item
transactions_df = pd.read_csv(transactions_path, delimiter='|', sep='.', encoding='utf-8')
# - list of product ids (subset of products from items_df) to be used for prediction
evaluation_df = pd.read_csv(evaluation_path, sep='.', encoding='utf-8')
items_df = pd.read_csv(items_path, delimiter='|', sep='.', encoding='utf-8')

# load category lookup table (manually created)
subject_cats_0 = pd.read_csv(subject_cats_0_path, delimiter=';', encoding='utf-8')

# Get shape of dfs
print(f'shape transactions_df: {transactions_df.shape}')
print(f'shape evaluation_df: {evaluation_df.shape}')
print(f'shape items_df: {items_df.shape}\n')

# Get col names + datatype
print(f'cols transactions_df: \n{transactions_df.dtypes}\n')
print(f'cols evaluation_df: \n{evaluation_df.dtypes}\n')
print(f'cols items_df: \n{items_df.dtypes}\n')

# Get description of dfs
print(f'desc transactions_df: \n{transactions_df.describe()}\n')
print(f'desc evaluation_df: \n{evaluation_df.describe()}\n')
print(f'desc items_df: \n{items_df.describe()}\n')

# Get cnt of unique sessions / items
print(f'cnt unqiue sessions: \n{transactions_df["sessionID"].nunique()}\n') #271,983
print(f'cnt unqiue items: \n{transactions_df["itemID"].nunique()}\n') #24,909

########################################################################################################################
# Preprocessing for further inspection
########################################################################################################################

# get len of mt string
items_df['mt_len'] = items_df['main topic'].str.len()
print(f'str len main topics: \n{items_df["mt_len"].describe()}\n')

# get first element (top level category) of mt string
items_df['mt_0'] = items_df['main topic'].str[0]

# add basket / order flag
transactions_df['basket_flg'] = np.where(transactions_df['basket'] > 0, 1, 0)
transactions_df['order_flg'] = np.where(transactions_df['order'] > 0, 1, 0)

########################################################################################################################
# General Overview per Table / Field
########################################################################################################################

########################################################################################################################
# Items

# AUTHOR ###############################################################################################################
# - there are 3240 books w/o author
# - on avg, there are 2.17 books per author

# count of books per author
books_per_author = pd.DataFrame.from_dict(Counter(items_df.loc[:,'author']),
                                    orient='index',
                                    columns=['book_cnt']).sort_values(by='book_cnt', ascending=False)
books_per_author['frac[%]'] = books_per_author['book_cnt'] * 100 / books_per_author['book_cnt'].sum()

# distribution of cnt af books among authors
books_per_author_cnts = pd.DataFrame.from_dict(Counter(books_per_author['book_cnt']),
                                               orient='index',
                                               columns=['author_cnt']).sort_index(ascending=False)
books_per_author_cnts['frac[%]'] = books_per_author_cnts['author_cnt'] * 100 / books_per_author_cnts['author_cnt'].sum()
books_per_author_cnts['frac.cum[%]'] = books_per_author_cnts['frac[%]'].cumsum()
print(f'books per author: \n{books_per_author.describe()}\n')

# PUBLISHER ############################################################################################################
# - almost half of publishers have at least 2 books in the items list
# - 1/3 of publishers has at least 3 books in the list

# count of books per publisher
books_per_publisher = pd.DataFrame.from_dict(Counter(items_df.loc[:,'publisher']),
                                    orient='index',
                                    columns=['book_cnt']).sort_values(by='book_cnt', ascending=False)
books_per_publisher['frac[%]'] = books_per_publisher['book_cnt'] * 100 / books_per_publisher['book_cnt'].sum()

# distribution of cnt af books among publishers
books_per_publisher_cnts = pd.DataFrame.from_dict(Counter(books_per_publisher['book_cnt']),
                                               orient='index',
                                               columns=['publisher_cnt']).sort_index(ascending=False)
books_per_publisher_cnts['frac[%]'] = books_per_publisher_cnts['publisher_cnt'] * 100 / books_per_publisher_cnts['publisher_cnt'].sum()
books_per_publisher_cnts['frac.cum[%]'] = books_per_publisher_cnts['frac[%]'].cumsum()
print(f'books per publisher: \n{books_per_publisher.describe()}\n')

# MAIN TOPICS ##########################################################################################################
# - top high level cats: Children’s, Teenage and Educational (62%), Fiction and Related items (34%)

# count of books per main topic (=mt) combo
books_per_mt = pd.DataFrame.from_dict(Counter(items_df.loc[:,'main topic']),
                                    orient='index',
                                    columns=['book_cnt']).sort_values(by='book_cnt', ascending=False)
books_per_mt['frac[%]'] = books_per_mt['book_cnt'] * 100 / books_per_mt['book_cnt'].sum()

# plot mt_0 distribution
sns.set_theme()
sns.histplot(items_df['mt_0'].astype(str).sort_values())

# count of books per first element of mt
books_per_mt_0 = pd.DataFrame.from_dict(Counter(items_df.loc[:,'mt_0']),
                                    orient='index',
                                    columns=['book_cnt']).sort_values(by='book_cnt', ascending=False).reset_index()
books_per_mt_0 = books_per_mt_0.rename(columns={'index': 'Notation'})
books_per_mt_0['frac[%]'] = books_per_mt_0['book_cnt'] * 100 / books_per_mt_0['book_cnt'].sum()

# join with category heading
books_per_mt_0 = books_per_mt_0.merge(subject_cats_0, on='Notation', how='left')
print(f'top 5 high level cats: \n{books_per_mt_0.head(5)}\n')

########################################################################################################################
# Transactions

# SESSIONID ############################################################################################################

# get cnt of items per sessionid
items_per_session = transactions_df[['sessionID', 'itemID']].groupby('sessionID')['itemID'].count().reset_index().\
    sort_values(by='itemID', ascending=False).rename(columns={'itemID': 'item_cnt'})
items_per_session['frac[%]'] = items_per_session['item_cnt'] * 100 / items_per_session['item_cnt'].sum()
print(f'items per session: \n{items_per_session.describe()}\n')

# CLICK ################################################################################################################
# get cnt of clicks per item
clicks_per_item = transactions_df[['itemID', 'click']].groupby('itemID')['click'].count().reset_index().\
    sort_values(by='click', ascending=False).rename(columns={'click': 'click_cnt'})
clicks_per_item['frac[%]'] = clicks_per_item['click_cnt'] * 100 / clicks_per_item['click_cnt'].sum()
print(f'clicks per item: \n{clicks_per_item.describe()}\n')

# BASKET ###############################################################################################################
# - difference to order: items that were added to basekt but not necessarily bought

# ORDER ################################################################################################################
# - items that where finally bought

# get cnt of orders per session
orders_per_session = transactions_df[['sessionID', 'order']].groupby('sessionID')['order'].count().reset_index().\
    sort_values(by='order', ascending=False).rename(columns={'order': 'order_cnt'})
orders_per_session['frac[%]'] = orders_per_session['order_cnt'] * 100 / orders_per_session['order_cnt'].sum()
print(f'orders per session: \n{orders_per_session.describe()}\n')

# get cnt of orders per item
orders_per_item = transactions_df[['itemID', 'order']].groupby('itemID')['order'].count().reset_index().\
    sort_values(by='order', ascending=False).rename(columns={'order': 'order_cnt'})
orders_per_item['frac[%]'] = orders_per_item['order_cnt'] * 100 / orders_per_item['order_cnt'].sum()
print(f'orders per item: \n{orders_per_item.describe()}\n')

# get frac of items that were added to basket but not bought
items_per_basket_order = transactions_df[['itemID',
                                          'basket_flg',
                                          'order_flg']].groupby(['basket_flg',
                                                                 'order_flg'])['itemID'].count().reset_index().rename(columns={'itemID': 'item_cnt'})
items_per_basket_order['frac[%]'] = items_per_basket_order['item_cnt'] * 100 / items_per_basket_order['item_cnt'].sum()
print(f'basket to order conversion: \n{items_per_basket_order}\n')

# # get items frequently ordered together
# import itertools
# item_order_agg_per_session = transactions_df.groupby(['sessionID']).agg({'itemID': lambda x: x.ravel().tolist()}).reset_index()
# combinations_list = []
# for row in item_order_agg_per_session.itemID:
#     combinations = list(itertools.combinations(row, 2))
#     combinations_list.append(combinations)
# combination_counts = pd.Series(combinations_list).explode().reset_index(drop=True)
# combination_counts.value_counts()