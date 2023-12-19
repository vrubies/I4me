import re
import multiprocessing
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.common.exceptions import WebDriverException, NoSuchElementException
from bs4 import BeautifulSoup
import os
import time

def scrape_author_articles_process(queue, author_id, url, page_load_timeout):
    url_set = set()
    url_pattern = re.compile(r'.*/\d{4}/\d{2}/\d{2}/.*')
    # url_pattern = re.compile(r'/\d{4}/\d{2}/\d{2}/$')

    options = webdriver.ChromeOptions()
    options.add_argument('--headless')
    driver = webdriver.Chrome(options=options)
    driver.set_page_load_timeout(page_load_timeout)

    start_time = time.time()

    try:
        driver.get(url)
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        for link in soup.find_all('a', href=True):
            if url_pattern.search(link['href']):
                url_set.add(link['href'])

        while True:
            if (time.time() - start_time > page_load_timeout - 1):
                break

            try:
                load_more_button = driver.find_element(By.CLASS_NAME, "load-more-button")
                if not (load_more_button.is_displayed() and load_more_button.is_enabled()):
                    break
                driver.execute_script("arguments[0].click();", load_more_button)
                time.sleep(0.5)  # Adjust sleep time if necessary
                # Re-parse the page
                soup = BeautifulSoup(driver.page_source, 'html.parser')
                for link in soup.find_all('a', href=True):
                    if url_pattern.search(link['href']):
                        url_set.add(link['href'])
            except NoSuchElementException:
                break
    except WebDriverException as e:
        print(f"Error occurred while loading the page for author ID {author_id}: {e}")
    finally:
        driver.quit()

    queue.put((author_id, list(url_set)))

def scrape_author_articles(author_id, url, page_load_timeout=5):
    queue = multiprocessing.Queue()
    p = multiprocessing.Process(target=scrape_author_articles_process, args=(queue, author_id, url, page_load_timeout))
    p.start()
    p.join(timeout=120)  # Set the timeout for the process

    if p.is_alive():
        p.terminate()  # Terminate the process if it's still running
        p.join()
        print(f"Timeout occurred while scraping articles for author ID {author_id}")
        return []

    author_id, articles = queue.get()  # Unpack the tuple
    return articles  # Return only the list of articles

def read_author_ids(file_path):
    with open(file_path, 'r') as file:
        return [line.strip() for line in file]

def save_articles_to_file(author_id, articles):
    author_folder = f"authors/{author_id}"
    os.makedirs(author_folder, exist_ok=True)
    with open(f"{author_folder}/articles.txt", 'w') as file:
        for article in articles:
            full_url = f"https://www.fool.com{article}\n"
            file.write(full_url)

def main():
    author_ids = read_author_ids("valid_author_ids.txt")
    for author_id in author_ids:
        url = f"https://www.fool.com/author/{author_id}/"
        articles = scrape_author_articles(author_id, url)
        save_articles_to_file(author_id, articles)

if __name__ == '__main__':
    main()
