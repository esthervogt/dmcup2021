import pandas as pd
from textblob import TextBlob
from tqdm import tqdm

tqdm.pandas()

def preprocess_books(items_path, books_path, evaluation_books_path):
    evaluation_books = pd.read_csv(evaluation_books_path)
    books = pd.read_csv(books_path, sep="|", dtype={"subtopic_query": "str"})
    books["subtopics"][62923] = "[]"

    def to_list(x):
        try:
            return x.strip('][').split(', ')
        except AttributeError:
            return []

    books["subtopics"] = books.subtopics.progress_apply(to_list)

    books.drop(["title", "author", "publisher"], axis=1, inplace=True)

    data = pd.read_csv(items_path, encoding='utf-8')

    data_merged = pd.merge(data, books, how="left", on="itemID")

    data_merged.set_index("itemID", inplace=True)

    data_merged.drop(
        ["description", "number_pages", "recommended_age", "release_date", "publisher", "headerID", "mt_st_cl",
         "description", "number_pages", "recommended_age", "release_date"], axis=1, inplace=True)

    data_merged.replace(
        {'language':
             {
                 "Deutsch (Untertitel: Deutsch, Englisch, Dänisch, Holländisch, Finnisch, Französisch, Norwegisch, Schwedisch)": "Deutsch",
                 "Deutsch, Dänisch, Englisch, Finnisch, Französisch, Isländisch, Italienisch, Japanisch, Niederländisch, Norwegisch, Schwedisch, Spanisch (Untertitel: Deutsch, Englisch, Dänisch, Holländisch, Finnisch, Französisch, Italienisch, Japanisch, Norweg": "Deutsch",
                 "Deutsch, Dänisch, Englisch, Finnisch, Französisch, Italienisch, Niederländisch, Norwegisch, Schwedisch, Spanisch": "Deutsch",
                 "Deutsch, Englisch": "Englisch",
                 "Deutsch, Englisch (Untertitel: Deutsch)": "Englisch",
                 "Deutsch, Englisch, Französisch": "Englisch",
                 "Deutsch, Englisch, Französisch, Italienisch, Spanisch": "Deutsch",
                 "Deutsch, Englisch, Polnisch": "Englisch",
                 "Deutsch, Französisch (Untertitel: Deutsch)": "Englisch",
                 "Deutsch, Spanisch": "Deutsch",
                 "Englisch, Französisch": "Englisch",
                 "Finnisch": "Finnisch",
                 "Hindi": "Hindi",
                 "Portugiesisch": "Portugiesisch",
                 "Schwedisch": "Schwedisch",
                 "Ungarisch": "Ungarisch",
                 "Englisch, Spanisch": "Spanisch",
                 "Deutsch, Englisch (Untertitel: Deutsch, Englisch)": "Englisch",
                 "Deutsch, Japanisch (Untertitel: Deutsch)": "Englisch",
                 "Deutsch, Französisch": "Französisch",
                 "Deutsch (Untertitel: Deutsch)": "Englisch",
                 "Deutsch, Englisch (Untertitel: Englisch)": "Englisch",
                 "Deutsch, Englisch, Französisch, Italienisch, Niederländisch, Spanisch": "Deutsch",
                 "Deutsch, Italienisch": "Italienisch",
                 "Arabisch, Englisch": "Englisch",
                 "Deutsch, Japanisch": "Englisch",
                 "Deutsch, Englisch, Türkisch (Untertitel: Deutsch, Englisch, Türkisch)": "Englisch",
                 "Deutsch, Englisch, Französisch, Italienisch": "Englisch",
                 "Deutsch, Dänisch, Englisch, Französisch, Isländisch, Italienisch, Niederländisch, Norwegisch, Schwedisch, Spanisch (Untertitel: Englisch, Dänisch, Deutsch, Französisch, Holländisch, Italienisch, Norwegisch, Schwedisch, Spanisch)": "Englisch",
                 "Deutsch, Englisch, Französisch, Niederländisch (Untertitel: Deutsch, Englisch, Französisch)": "Englisch",
                 "Englisch, Italienisch": "Italienisch",
                 "Deutsch, Englisch, Spanisch": "Spanisch",
                 "Deutsch, Englisch, Französisch, Italienisch, Polnisch, Portugiesisch, Russisch, Spanisch": "Englisch",
                 "Deutsch, Englisch, Französisch, Niederländisch (Untertitel: Deutsch, Englisch, Französisch, Holländisch)": "Englisch",
                 "Arabisch, Deutsch, Dänisch, Englisch, Finnisch, Französisch, Hindi, Isländisch, Niederländisch, Norwegisch, Schwedisch, Spanisch (Untertitel: Deutsch, Englisch, Arabisch, Dänisch, Finnisch, Norwegisch, Schwedisch, Französisch, Holländisch, Hin": "Deutsch",
                 "Deutsch, Griechisch": "Deutsch",
                 "Deutsch (Untertitel: Deutsch, Englisch)": "Englisch",
                 "Chinesisch, Englisch": "Englisch",
                 "Deutsch, Schwedisch": "Schwedisch",
                 "Estnisch": "Baltisch",
                 "Litauisch": "Baltisch"
                 }
         }, inplace=True)

    query = data_merged.loc[evaluation_books.itemID]
    query.reset_index(inplace=True)

    index = query[pd.isna(query["language"])].itemID
    query.set_index("itemID", inplace=True)
    for book_id in index:
        text = query.loc[book_id]["title"]
        lang = TextBlob(text)
        query.at[book_id, 'language'] = lang.detect_language()
    query.reset_index(inplace=True)

    query.replace(
        {'language':
            {
                "de": "Deutsch",
                "en": "Englisch",
                "haw": "Englisch",
                "es": "Spanisch"
            }
        }, inplace=True)

    query.rename({"itemID": "id_query", "title": "title_query", "author": "author_query", "main topic": "topic_query",
                  "subtopics": "subtopic_query"}, axis=1, inplace=True)
    query.groupby("language").count()["id_query"].sort_values(ascending=False)

    data_merged.reset_index(inplace=True)

    data_merged.rename({"itemID": "id_document", "title": "title_document", "author": "author_document",
                        "main topic": "topic_document", "subtopics": "subtopic_document"}, axis=1, inplace=True)

    return query, data_merged


def preprocess_language(query, document, nlp):
    document["title_processed_document"] = document.title_document.progress_apply(
        lambda sentence: [token.lemma_.lower() for token in nlp(sentence) if
                          not token.is_punct and not token.is_stop])
    query["title_processed_query"] = query.title_query.progress_apply(
        lambda sentence: [token.lemma_.lower() for token in nlp(sentence) if
                          not token.is_punct and not token.is_stop])
    cross = query.merge(document, how='cross')
    cross = cross[cross.id_query != cross.id_document]

    return cross