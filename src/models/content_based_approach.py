import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Parameters
base_book = 22120
items_path = 'tempData/sourceData/items.csv'

# Some preprocessing
items_df = pd.read_csv(items_path, delimiter='|', sep='.', encoding='utf-8')
items_df = items_df.dropna(axis=0)

# Concatenating details in a combined string
title_author = pd.Series(
    items_df.apply(lambda x: f'{x["main topic"]} {" ".join(x["subtopics"].replace("[", "").replace("]", "").split(","))} {x["author"].replace(" ", "")} {x["title"]}',
                   axis=1))

# Creating tfidf-matrix
tfidfVectorizer = TfidfVectorizer()
tfidf_matrix = tfidfVectorizer.fit_transform(title_author)

# Retrieving book for recommendations
print(title_author[items_df['itemID'] == base_book])
test_tfidf = tfidf_matrix[items_df['itemID'] == base_book]

# Retrieving most similar books
cos_similarity_tfidf = list(map(lambda x: cosine_similarity(test_tfidf, x), tfidf_matrix))
top = sorted(range(len(cos_similarity_tfidf)), key=lambda i: cos_similarity_tfidf[i], reverse=True)[:50]
result = items_df.iloc[top, :]
print(result['title'].head())
