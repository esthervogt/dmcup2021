import pandas as pd
import requests
from bs4 import BeautifulSoup


def get_descriptions(books, book_source_path):
    """ Function to get all descriptions of a list of books

    Example: get_descriptions(books_evaluation_list, "../tempData/sourceData/items.csv")

    Args:
        books (list): List of books with their unique ID from the data
        book_source_path: Path to the folder in which the item file is saved in

    Returns:
        Data Frame of all books (itemIDs) and their descriptions
    """
    book_source = get_book_df(book_source_path)

    df_list =[]
    for book_id in books:
        dict_description = {"itemID":book_id,
                            "description":get_description(book_id,book_source)}
        df_list.append(dict_description)

    df_descriptions = pd.DataFrame.from_dict(df_list)
    return df_descriptions

def get_description(book_id, book_source):
    """ Function to scrape a description of a book based on its ID

        Example: get_description(203421, book_source)

        Args:
            book_id (id): unique ID of a book from the DMCUP
            book_source (dataframe): DF containing all the books considered

        Returns:
            Description from Thalia as a String
        """
    # Get the title based on the itemID
    title = book_source.loc[(book_source['itemID'] == book_id)]["title"].item()

    # Get the Thalia URL based on the title
    url_search = "https://www.thalia.de/suche?filterPATHROOT=&sq=" + title
    page = requests.get(url_search)
    soup = BeautifulSoup(page.content, 'html.parser')
    href = soup.find_all('a', caption="suchergebnis-klick")[0]['href']
    url_book = "https://www.thalia.de" + href

    # Scrape the Cover from
    page = requests.get(url_book)
    soup = BeautifulSoup(page.content, "html.parser")
    description = soup.find_all('div', class_='text-infos')[0]
    des = str(description)
    des = des.replace('<div class="text-infos">', '').replace('<p>', '').replace('</p>', '').replace('</b><br/>',' ').replace('</b>', '').replace('<br/>', '').replace('<b>', '')
    des = des.split('<')[0]

    return des

def get_book_df(path):
    """ Function to load a certain dataframe with ids and book titles from the DMCUP

            Example: get_book_df("../tempData/sourceData/items.csv")

            Args:
                path (string): Path of the folder in which the data is stored

            Returns:
                Data Frame which contains the IDs and the titles
            """
    book_df = pd.read_csv(path, delimiter='|', sep='.', encoding='utf-8')
    book_df = book_df.drop('main topic', axis=1).drop('subtopics', axis=1)
    return book_df

Test = get_descriptions([15606], "../tempData/sourceData/items.csv")
print(Test)