import requests
from bs4 import BeautifulSoup
import concurrent.futures

def is_valid_author_id(author_id):
    url = f"https://www.fool.com/author/{author_id}/"
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        error_message = soup.find('div', string='404 Error')
        if error_message:
            return False

        no_articles = soup.find('p', string='No articles present.')
        return no_articles is None
    except Exception as e:
        print(f"Error checking author ID {author_id}: {e}")
        return False

def save_author_id(author_id, filename='valid_author_ids.txt'):
    with open(filename, 'a') as file:
        file.write(f"{author_id}\n")

def process_author_id(author_id):
    print(f"Checking author ID: {author_id}")
    if is_valid_author_id(author_id):
        print(f"Found valid author ID: {author_id}")
        save_author_id(author_id)

def find_valid_author_ids(start_id, end_id):
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        author_ids = range(start_id, end_id + 1)
        executor.map(process_author_id, author_ids)

# Example usage
find_valid_author_ids(10208 + 1, 25000)
