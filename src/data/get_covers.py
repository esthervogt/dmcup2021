import pandas as pd
import requests
from bs4 import BeautifulSoup
import os


def get_covers(books, target_folder_name, book_source_path):
    """ Function to get all covers of a list of books

    Example: get_covers(books_evaluation_list, "Covers", "../tempData/sourceData/items.csv")

    Args:
        books (list): List of books with their unique ID from the data
        target_folder_name: Name of the folder we want our images to be saved in
        book_source_path: Path to the folder in which the item file is saved in

    Returns:
        Saves scraped cover images in a certain folder
    """
    book_source = get_book_df(book_source_path)
    try:
        os.mkdir(os.path.join(os.getcwd(), target_folder_name))
    except:
        pass
    os.chdir(os.path.join(os.getcwd(), target_folder_name))
    for book_id in books:
        get_cover(book_id, book_source)



def get_cover(book_id, book_source):
    """ Function to scrape a cover of a book based on its ID and save it in a certain folder

        Example: get_cover(203421, book_source)

        Args:
            book_id (id): unique ID of a book from the DMCUP
            book_source (dataframe): DF containing all the books considered

        Returns:
            Saves a scraped cover image
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
    cover = soup.find_all('img', class_='largeImg')[0]['src']
    name = str(book_id)

    # Saves the Cover as a jpg in the Cover folder
    with open(name + '.jpg', 'wb') as f:
        im = requests.get(cover)
        f.write(im.content)

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

get_covers([15606],"Covers", "../tempData/sourceData/items.csv")