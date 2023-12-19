import requests
from bs4 import BeautifulSoup

def is_valid_author_id(author_id):
    url = f"https://www.fool.com/author/{author_id}/"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Check for 404 Error
    error_message = soup.find('div', string='404 Error')
    if error_message:
        return False

    # Check for "No articles present."
    no_articles = soup.find('p', string='No articles present.')
    return no_articles is None

def save_author_id(author_id, filename='valid_author_ids.txt'):
    with open(filename, 'a') as file:
        file.write(f"{author_id}\n")

def find_valid_author_ids(start_id, end_id):
    for author_id in range(start_id, end_id + 1):
        print(f"Checking author ID: {author_id}")
        if is_valid_author_id(author_id):
            print(f"Found valid author ID: {author_id}")
            save_author_id(author_id)

# Example usage
find_valid_author_ids(6000, 25000)
