import pandas as pd


def jaccard(vector_source, vector_target):
    """ Calculate jaccard similarity between two list of tokens.
        Args:
            vector_source (array): Array of preprocessed token in source language.
            vector_target (array): Array of preprocessed token in target language.

        Returns:
            array: Array containing jaccard similarity.
    """
    df = pd.DataFrame()
    df['original'] = vector_source
    df['translated'] = vector_target

    def jaccard_similarity_score(original, translation):
        """ Calculate jaccard similarity between two lists.
        """
        intersect = set(original).intersection(set(translation))
        union = set(original).union(set(translation))
        try:
            return len(intersect) / len(union)
        except ZeroDivisionError:
            return 0

    jaccard_vec = df.progress_apply(lambda x: jaccard_similarity_score(x.original,
                                                                       x.translated),
                                    axis=1)
    return jaccard_vec
