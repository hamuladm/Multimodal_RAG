from langchain_text_splitters.html import HTMLHeaderTextSplitter

def split_html(path: str):
    headers = [
        ("p", ""),
    ]

    with open(path, "r") as html:
        content = html.read()
    splitter = HTMLHeaderTextSplitter(headers_to_split_on=headers, return_each_element=True)
    splitted = splitter.split_text(content)
    text = [p.page_content for p in splitted if len(p.page_content) >= 50]
    return ' '.join(text)
