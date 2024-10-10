from bs4 import BeautifulSoup
from urllib.parse import urlparse

def parse_sitemap():
    with open('sitemap-0.xml', 'r') as file:
        xml_content = file.read()

    soup = BeautifulSoup(xml_content, 'xml')

    batch_links = []
    for loc in soup.find_all('loc'):
        url = urlparse(loc.text)
        splitted = url.path.split('/')
        if ('the-batch' in splitted) and not ('tag' in splitted or 'page' in splitted):
            batch_links.append(loc.text)

    batch_links = sorted(batch_links)

    with open("links.txt", "w") as file:
        for link in batch_links:
            file.write(link + '\n')