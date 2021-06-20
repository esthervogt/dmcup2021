from tqdm import tqdm

from src.features.jaccard import jaccard

tqdm.pandas()


def add_to_recommendation(cross_cut, recommendation):
    for i in tqdm(range(cross_cut.shape[0])):
        query_id = cross_cut.id_query[i]
        document_id = cross_cut.id_document[i]
        if document_id not in recommendation[query_id] and len(recommendation[query_id]) < 5:
            recommendation[query_id].append(document_id)


def search_recommendation(cross, recommendation):
    cross["jaccard"] = jaccard(cross.title_processed_query, cross.title_processed_document)
    cross = cross.sort_values("jaccard", ascending=False)

    # Filter on author
    try:
        cross_author = cross[cross.author_query == cross.author_document]
        cross_author_cutoff = cross_author[cross_author.jaccard >= 0.2].reset_index(
            drop=True)
        add_to_recommendation(cross_author_cutoff, recommendation)
    except AttributeError:
        print("No same author")

    # Filter on categories
    cross_category = cross[cross.topic_query == cross.topic_document]
    try:
        cross_filter = cross_category[cross_category["subtopic_query"].apply(lambda x: x != [""])]
        cross_subtopic = cross_filter[(cross_filter.subtopic_query == cross_filter.subtopic_document)]
        cross_subtopic_cutoff = cross_subtopic[cross_subtopic.jaccard >= 0.2].reset_index(
            drop=True)
        add_to_recommendation(cross_subtopic_cutoff, recommendation)
    except AttributeError:
        print("No same subtopic")

    try:
        cross_category_cutoff = cross_category[cross_category.jaccard >= 0.2].reset_index(
            drop=True)
        add_to_recommendation(cross_category_cutoff, recommendation)
    except AttributeError:
        print("No same category")

    cross_cutoff = cross[cross.jaccard >= 0.2].reset_index(drop=True)
    add_to_recommendation(cross_cutoff, recommendation)

    # Fill the rest
    try:
        cross_author_cut = cross_author[cross_author.jaccard < 0.2].reset_index(
            drop=True)
        add_to_recommendation(cross_author_cut, recommendation)
    except (AttributeError,UnboundLocalError):
        print("No same author")

    try:
        cross_subtopic_cut = cross_subtopic[cross_subtopic.jaccard < 0.2].reset_index(
            drop=True)

        add_to_recommendation(cross_subtopic_cut, recommendation)
    except (AttributeError,UnboundLocalError):
        print("No same subtopic")

    try:
        cross_category_cut = cross_category[cross_category.jaccard < 0.2].reset_index(
            drop=True)
        add_to_recommendation(cross_category_cut, recommendation)
    except (AttributeError,UnboundLocalError):
        print("No same category")

    cross_cut = cross[cross.jaccard < 0.2].reset_index(drop=True)

    add_to_recommendation(cross_cut, recommendation)