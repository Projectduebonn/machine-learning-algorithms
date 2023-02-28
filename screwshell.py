import requests
from bs4 import BeautifulSoup

url = 'https://www.boc.cn/sourcedb/whpj/'
response = requests.get(url)

soup = BeautifulSoup(response.text, 'html.parser')
table = soup.find('table', attrs={'class': 'BOC_main publish'})

rows = table.find_all('tr')
for row in rows:
    cols = row.find_all('td')
    cols = [col.text.strip() for col in cols]
    if cols and cols[0] == '欧元':
        print(cols[4])  # 输出欧元现钞卖出价数据
