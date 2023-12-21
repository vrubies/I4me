import os
import requests
import json
from bs4 import BeautifulSoup
import concurrent.futures
import gzip
import shutil


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

    # List of common class names used for article bodies
    common_classes = ['tailwind-article-body', 'article-body', 'post-body', 'content', 'main-content', 'story-content']

    article_body = None
    for class_name in common_classes:
        article_body = soup.find('div', class_=class_name)
        if article_body:
            break

    if not article_body:
        return "Article content not found"

    article_text = ' '.join(p.get_text() for p in article_body.find_all('p'))

    # Clean the extracted text
    clean_article_text = clean_text(article_text)
    return clean_article_text

def compress_file(file_path):
    with open(file_path, 'rb') as f_in:
        with gzip.open(file_path + '.gz', 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    os.remove(file_path)

def process_author_folder(author_folder):
    compressed_file_path = os.path.join(author_folder, 'articles_text.json.gz')
    if os.path.exists(compressed_file_path):
        print(f"Compressed file already exists for {author_folder}, skipping.")
        return
    
    with open(os.path.join(author_folder, 'articles.txt'), 'r') as file:
        urls = file.read().splitlines()
    
    articles_data = []
    total_urls = len(urls)
    for index, url in enumerate(urls, start=1):
        print(f"Processing URL {index} of {total_urls} in folder {author_folder}")
        try:
            response = requests.get(url, timeout=10)  # Setting a timeout of 10 seconds
            # print(f"URL: {url}, Status Code: {response.status_code}")
            article_text = extract_text_from_html(response.content)
            # print(f"Extracted text length for {url}: {len(article_text)}")
            word_count = len(article_text.split())

            if word_count >= 50:
                articles_data.append({url: article_text})
        except requests.exceptions.Timeout:
            print(f"Timeout occurred while processing URL {url}")
        except Exception as e:
            print(f"Error processing URL {url}: {e}")

    if articles_data:
        json_path = os.path.join(author_folder, 'articles_text.json')
        with open(json_path, 'w') as file:
            json.dump(articles_data, file, indent=4)

        compress_file(json_path)
    else:
        print("No valid articles to compress.")

def main():
    authors_dir = './authors'  # Replace with your directory path
    author_folders = [os.path.join(authors_dir, f) for f in os.listdir(authors_dir) if os.path.isdir(os.path.join(authors_dir, f))]

    total_authors = len(author_folders)
    authors_processed = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        futures = {executor.submit(process_author_folder, author_folder): author_folder for author_folder in author_folders}
        for future in concurrent.futures.as_completed(futures):
            folder = futures[future]
            authors_processed += 1
            remaining_authors = total_authors - authors_processed
            print(f"Finished processing folder: {folder}. Remaining authors: {remaining_authors}")
            try:
                future.result()
            except Exception as e:
                print(f"Error processing folder {folder}: {e}")

if __name__ == '__main__':
    main()