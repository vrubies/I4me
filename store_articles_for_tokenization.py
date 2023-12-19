import os
import requests
import json
from bs4 import BeautifulSoup

def clean_text(text):
    # Replace common HTML entities or Unicode characters
    replacements = {
        '\u00a0': ' ',  # Non-breaking space
        # Add more replacements as needed
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    # Optionally, add a step to remove multiple consecutive spaces
    text = ' '.join(text.split())

    return text

def extract_text_from_html(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    
    article_body = soup.find('div', class_='tailwind-article-body')
    if not article_body:
        return "Article content not found"

    article_text = ' '.join(p.get_text() for p in article_body.find_all('p'))
    
    # Clean the extracted text
    clean_article_text = clean_text(article_text)
    return clean_article_text

def process_author_folder(author_folder):
    with open(os.path.join(author_folder, 'articles.txt'), 'r') as file:
        urls = file.read().splitlines()
    
    articles_data = []
    total_urls = len(urls)
    for index, url in enumerate(urls, start=1):
        print(f"Processing URL {index} of {total_urls} in folder {author_folder}")
        try:
            response = requests.get(url)
            article_text = extract_text_from_html(response.content)
            word_count = len(article_text.split())

            if word_count >= 50:
                articles_data.append({url: article_text})
        except Exception as e:
            print(f"Error processing URL {url}: {e}")

    with open(os.path.join(author_folder, 'articles_text.json'), 'w') as file:
        json.dump(articles_data, file, indent=4)

def main():
    authors_dir = './authors'  # Replace with your directory path
    author_folders = [f for f in os.listdir(authors_dir) if os.path.isdir(os.path.join(authors_dir, f))]
    total_folders = len(author_folders)

    for index, author_id in enumerate(author_folders, start=1):
        print(f"Processing folder {index} of {total_folders}: {author_id}")
        author_folder = os.path.join(authors_dir, author_id)
        process_author_folder(author_folder)
        print(f"Finished processing folder {index} of {total_folders}: {author_id}")

if __name__ == '__main__':
    main()
