import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import os
import re


def save_text_page(content, url):
    filename = re.sub(r"[^\w\-]", "_", url) + ".html"
    filepath = os.path.join("scraped_text", filename)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"Saved text from {url} as {filename}")


def download_file(url, folder):
    try:
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            parsed_url = urlparse(url)
            filename = os.path.basename(parsed_url.path)

            if not filename:
                filename = "downloaded_file"

            if parsed_url.query:
                filename += "_" + re.sub(r"[^\w\-]", "_", parsed_url.query)

            if not filename or filename == "/":
                filename = "downloaded_file"

            filepath = os.path.join(folder, filename)
            os.makedirs(folder, exist_ok=True)

            with open(filepath, "wb") as f:
                f.write(response.content)

            print(f"Downloaded {filename} from {url}")
    except Exception as e:
        print(f"Failed to download {url}: {e}")


def extract_and_download_images(soup: BeautifulSoup, base_url):
    for img_tag in soup.find_all("img", src=True):
        img_url = urljoin(base_url, img_tag["src"])
        if re.match(r"^.*\.(jpg|jpeg|png|gif)$", img_url):
            download_file(img_url, "scraped_images")


def scrape_page():
    with open("links.txt", "r") as file:
        links = file.read()
    links = links.split("\n")
    try:
        for url in links:
            response = requests.get(url)

            if response.status_code == 200:
                print(f"Scraping {url}...")
                soup = BeautifulSoup(response.text, "html.parser")
                save_text_page(response.text, url)
                extract_and_download_images(soup, url)
    except Exception as e:
        print(f"Failed to scrape {url}: {e}")


if __name__ == "__main__":
    scrape_page()
