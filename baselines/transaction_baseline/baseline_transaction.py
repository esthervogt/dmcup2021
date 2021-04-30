import sys
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
from nltk.tokenize import word_tokenize


def recommend_based_on_transactions(df_transactions, df_items, item_id, sort_by="sum",
                                    max_number_recommendation=5, verbose=True):
    original_book = convert_id_to_name(df_items, item_id)
    original_title = original_book[["title"]].values[0][0]
    original_author = original_book[["author"]].values[0][0]
    recommendation_id = []

    if original_book.empty:
        sys.exit("ITEM ID NOT FOUND!")
    if verbose:
        print("Find recommendations based on: ")
        print("{} by {}.".format(original_title, original_author))
        print("\n")
        print("We recommend: ")
    session_ids = find_all_sessions_id(df_transactions, item_id)
    itemID_rows = find_all_item_id(df_transactions, session_ids)
    item_properties = find_number_click_basket_order(itemID_rows)

    if sort_by == "sum":
        summed_item_properties = sum_click_basket_order(item_properties)
        sorted_itemID = sorted(summed_item_properties, key=summed_item_properties.get, reverse=True)

        number_recommendation = 0
        for single_itemID in sorted_itemID:
            rank_book = convert_id_to_name(df_items, single_itemID)
            if single_itemID == item_id or compare_strings(original_title, rank_book[["title"]].values[0][0]):
                continue
            if number_recommendation < max_number_recommendation:
                recommendation_id.append(single_itemID)
                if verbose:
                    print("{}. {} by {}.".format(number_recommendation + 1,
                                                 rank_book[["title"]].values[0][0],
                                                 rank_book[["author"]].values[0][0]))
                number_recommendation += 1
            else:
                break
        if verbose:
            for i in range(max_number_recommendation):
                if i + 1 >= number_recommendation + 1:
                    print("{}. Not enough data to give recommendation".format(i + 1))

    else:
        sys.exit("INCORRECT ARGUMENT FOR sort_by!")

    return recommendation_id


def find_all_sessions_id(df, item_id):
    session_ids = df.loc[df['itemID'] == item_id]["sessionID"].to_list()
    return session_ids


def find_all_item_id(df, session_ids):
    return df.loc[df['sessionID'].isin(session_ids)]


def find_number_click_basket_order(itemID_rows):
    item_properties = {}

    for index, row in itemID_rows.iterrows():
        if row['itemID'] in item_properties.keys():
            item_properties[row['itemID']][0] += row['click']
            item_properties[row['itemID']][1] += row['basket']
            item_properties[row['itemID']][2] += row['order']
        else:
            item_properties[row['itemID']] = [row['click'], row['basket'], row['order']]
    return item_properties


def sum_click_basket_order(item_properties):
    summed_item_properties = {}
    for key, value in item_properties.items():
        summed_item_properties[key] = sum(value)

    return summed_item_properties


def convert_id_to_name(df_items, itemID):
    return df_items.loc[df_items['itemID'] == itemID]


def compare_strings(text1, text2):
    test1_wh_sw = remove_stopwords(text1)
    test2_wh_sw = remove_stopwords(text2)

    return test1_wh_sw.lower() == test2_wh_sw.lower()


def remove_stopwords(text):
    text_tokens = word_tokenize(text)
    tokens_without_sw = [word for word in text_tokens if not word in stopwords.words()]
    return " ".join(tokens_without_sw)