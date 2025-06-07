import bs4

def download_tic(file_name, class_):
    with open(file_name) as f:
        html = f.read()
    soup = bs4.BeautifulSoup(html)
    return [tic.text.strip() for tic in soup.find_all('span', class_=class_)]