"""Module to connect with the firestore database to write and retrieve documents.

  This module connects the local python instance to the NoSQL firestore database in the firebase backend.
  Additionally it contains functions to retrieve data from firestore and write data to the database.

  Typical usage example:

    >> firestore_client = connect_to_firestore(credentials="firebase_secret/dmc-book-recommendation-firebase-adminsdk-yebqi-d805561028.json",
                                               database_url="https://book-recommendation-website.firebaseio.com/")
    >> results = retrieve_documents_from_collection(firestore_client, collection="test_collection")
    >> import pandas as pd
    >> data = pd.DataFrame({test_field_write: ['test_field_value_1', 'test_field_value_2']})
    >> write_documents_to_collection(firestore_client, data, collection="test_write_collection")
"""

import firebase_admin
from firebase_admin import firestore
from tqdm import tqdm

tqdm.pandas()


def connect_to_firestore(credentials="../secrets/dmc-book-recommendation-firebase-adminsdk-yebqi-d805561028.json",
                         database_url="https://book-recommendation-website.firebaseio.com/"):
    """Connects to firestore instance.

    Connects to specified firestore instance with the admin credentials.

    Args:
        database_url: url of the database as specified with https://{project_id}.firebaseio.com/
        credentials: path to admin credential file

    Returns:
        A connection to the specified firestore database.
    """

    cred = firebase_admin.credentials.Certificate(credentials)
    firebase_admin.initialize_app(cred, {
        "databaseURL": database_url
    })
    firestore_client = firestore.client()

    return firestore_client


def retrieve_documents_from_collection(firestore_client, collection="test_collection"):
    """Retrieves all documents of a specified firestore collection.

    Args:
        collection: firestore collection of the documents that should be retrieved
        firestore_client: specified firestore connection

    Returns:
        A list of dictionaries containing the content of the documents of the specified collection.
    """

    docs_ref = firestore_client.collection(collection)
    docs = docs_ref.get()
    results = []
    for doc in docs:
        results.append(doc.to_dict())

    return results


def write_documents_to_collection(firestore_client, data, collection="test_write_collection"):
    """Writes documents from a dataframe to a specified collection.

    Args:
        collection: firestore collection where to the documents should be inserted
        data: dataframe
        firestore_client: specified firestore connection
    """

    documents = data.to_dict(orient="records")
    collection_ref = firestore_client.collection(collection)
    list(map(lambda x: collection_ref.add(x), tqdm(documents)))
