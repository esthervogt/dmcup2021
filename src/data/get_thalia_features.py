import pandas as pd
import requests
from bs4 import BeautifulSoup
from datetime import datetime


def get_thalia_features(books, book_source_path):
    """ Function to get all features scraped from Thalia for a list of books

    Example: get_thalia_features(books_evaluation_list, "../tempData/sourceData/items.csv")

    Args:
        books (list): List of books with their unique ID from the data
        book_source_path: Path to the folder in which the item file is saved in

    Returns:
        Data Frame of all books (itemIDs) and their descriptions, rating, number of pages, age recommendation, release date, language, sales rank
    """
    book_source = get_book_df(book_source_path)

    df_list =[]
    for book_id in books:
        # Get the Thalia URL based on the title    
        url_book = get_url(book_id, book_source)
        if url_book != '':
            page = requests.get(url_book)
            soup = BeautifulSoup(page.content, "html.parser")    
            dict_description = {"itemID": book_id,
                                "description": get_description(book_id,book_source, soup),
                                "rating": get_rating(book_id, book_source, soup),
                                "number_pages": get_number_pages(book_id, book_source, soup),
                                "recommended_age": get_recommended_age(book_id, book_source, soup),
                                "release_date": get_release_date(book_id, book_source, soup),
                                "language": get_language(book_id, book_source, soup),
                                "thalia_ranking": get_thalia_ranking(book_id, book_source, soup),
                                "cover_url": get_cover(book_id, book_source, soup)}
            df_list.append(dict_description)
        else:
            dict_description = {"itemID": book_id,
                                "description": '',
                                "rating": '',
                                "number_pages": '',
                                "recommended_age": '',
                                "release_date": '',
                                "language": '',
                                "thalia_ranking": '',
                                "cover_url": ''}
            df_list.append(dict_description)
    df_descriptions = pd.DataFrame.from_dict(df_list)
    
    return df_descriptions


def get_url(book_id, book_source):
    """ Function to get the correct URL to scrape a book from Thalia

        Example: get_url(203421, book_source)

        Args:
            book_id (id): unique ID of a book from the DMCUP
            book_source (dataframe): DF containing all the books considered

        Returns:
            url to the Thalia webpage of the book as a String
        """
    try:
        # Get the title based on the itemID
        title = book_source.loc[(book_source['itemID'] == book_id)]["title"].item()
    
        # Get the Thalia URL based on the title
        url_search = "https://www.thalia.de/suche?filterPATHROOT=&sq=" + title
        page = requests.get(url_search)
        soup = BeautifulSoup(page.content, 'html.parser')
        href = soup.find_all('a', caption="suchergebnis-klick")[0]['href']
        url_book = "https://www.thalia.de" + href
    except:
        url_book = ''
    
    return url_book


def get_description(book_id, book_source, soup):
    """ Function to scrape a description of a book based on its ID

        Example: get_description(203421, book_source)

        Args:
            book_id (id): unique ID of a book from the DMCUP
            book_source (dataframe): DF containing all the books considered
            soup (bs4.BeautifulSoup): Content of Thalia webpage of the book as BeautifulSoup object

        Returns:
            Description from Thalia as a String
        """
    try:
        description = soup.find_all('div', class_='text-infos')[0]
        des = str(description)
        des = des.replace('<div class="text-infos">', '').replace('<p>', '').replace('</p>', '').replace('</b><br/>',' ').replace('</b>', '').replace('<br/>', '').replace('<b>', '')
        des = des.split('<')[0]
    except:
        des = ''

    return des


def get_rating(book_id, book_source, soup):
    """ Function to scrape a rating of a book based on its ID

        Example: get_rating(203421, book_source)

        Args:
            book_id (id): unique ID of a book from the DMCUP
            book_source (dataframe): DF containing all the books considered
            soup (bs4.BeautifulSoup): Content of Thalia webpage of the book as BeautifulSoup object

        Returns:
            Rating from Thalia as INT with 5 being the best and 1 being the worst rating. 0 means that there is no rating.
        """
    try:
        rating = len(soup.find_all('div', class_="oRating")[0].find_all('span', class_="active"))
    except:
        rating = ''

    return rating


def get_number_pages(book_id, book_source, soup):
    """ Function to scrape the number of pages of a book based on its ID

        Example: get_number_pages(203421, book_source)

        Args:
            book_id (id): unique ID of a book from the DMCUP
            book_source (dataframe): DF containing all the books considered
            soup (bs4.BeautifulSoup): Content of Thalia webpage of the book as BeautifulSoup object

        Returns:
            Number of pages from Thalia as INT
        """
    try:
        number_pages = int(soup.find_all('tr', {'data-test':"produktdetails_seitenzahl"})[0].find_all('td')[0].text.strip())
    except:
        number_pages = ''

    return number_pages


def get_recommended_age(book_id, book_source, soup):
    """ Function to scrape the recommended age of a book based on its ID

        Example: get_recommended_age(203421, book_source)

        Args:
            book_id (id): unique ID of a book from the DMCUP
            book_source (dataframe): DF containing all the books considered
            soup (bs4.BeautifulSoup): Content of Thalia webpage of the book as BeautifulSoup object

        Returns:
            Recommended age from Thalia as string
        """
    try:
        recommended_age = soup.find_all('tr', {'data-test':"produktdetails_altersempfehlung"})[0].find_all('td')[0].text.strip()
    except:
        recommended_age = ''

    return recommended_age


def get_release_date(book_id, book_source, soup):
    """ Function to scrape the release_date of a book based on its ID

        Example: get_release_date(203421, book_source)

        Args:
            book_id (id): unique ID of a book from the DMCUP
            book_source (dataframe): DF containing all the books considered
            soup (bs4.BeautifulSoup): Content of Thalia webpage of the book as BeautifulSoup object

        Returns:
            Release date from Thalia as datetime
        """
    try:
        release_date = datetime.strptime((soup.find_all('tr', {'data-test':"produktdetails_erscheinungsdatum"})[0].find_all('td')[0].text.strip()), '%d.%m.%Y')
    except:
        release_date = ''
        
    return release_date


def get_language(book_id, book_source, soup):
    """ Function to scrape the language of a book based on its ID

        Example: get_language(203421, book_source)

        Args:
            book_id (id): unique ID of a book from the DMCUP
            book_source (dataframe): DF containing all the books considered
            soup (bs4.BeautifulSoup): Content of Thalia webpage of the book as BeautifulSoup object

        Returns:
            Language from Thalia as string
        """
    try:
        language = soup.find_all('tr', {'data-test':"produktdetails_sprache"})[0].find_all('td')[0].text.strip()
    except:
        language = ''

    return language


def get_thalia_ranking(book_id, book_source, soup):
    """ Function to scrape the thalia_ranking of a book based on its ID

        Example: get_thalia_ranking(203421, book_source)

        Args:
            book_id (id): unique ID of a book from the DMCUP
            book_source (dataframe): DF containing all the books considered
            soup (bs4.BeautifulSoup): Content of Thalia webpage of the book as BeautifulSoup object

        Returns:
            thalia_ranking from Thalia as INT
        """
    try:
        thalia_ranking = int(soup.find_all('tr', {'data-test':"produktdetails_verkaufsrang"})[0].find_all('td')[0].text.strip())
    except:
        thalia_ranking = ''

    return thalia_ranking


def get_cover(book_id, book_source, soup):
    """ Function to scrape a cover of a book based on its ID and save it in a certain folder

        Example: get_cover(203421, book_source)

        Args:
            book_id (id): unique ID of a book from the DMCUP
            book_source (dataframe): DF containing all the books considered

        Returns:
            Saves a scraped cover image
        """
    try:
        cover = soup.find_all('img', class_='largeImg')[0]['src']
    except:
        cover = ''
    
    return cover


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

#Test = get_thalia_features([15606], "../tempData/sourceData/items.csv")
#print(Test)