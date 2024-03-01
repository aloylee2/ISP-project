# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 11:52:47 2021

# @author: aloyl
"""
import requests
import pandas as pd
import re
import time

data = pd.read_json("News_Category_Dataset_v2.json", lines=True)
test = data[1:2001].to_dict(orient="records")

def get_html(row):
    time.sleep(0.15)
    content = requests.get(row['link']).content
    res = re.findall('<p>(.*?)</p>', str(content))
    text = "\n".join(res)
    # print(text)
    # print("---------------------------------------------------------------------") 
    clean = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
    cleantext = re.sub(clean, '',text)
    print(cleantext)    
      
    return {"category": row["category"], "html": cleantext}





htmls = [get_html(row) for row in test]
pd.DataFrame({"category": [dt['category'] for dt in htmls], "html": [dt['html'] for dt in htmls]}).to_csv("newsgroup2.csv", index=False)


# import requests
# import pandas as pd
# import re
# import time
# import multiprocess as mp
# import numpy as np
# from bs4 import BeautifulSoup as bs4
# import itertools
# import sys

# html_file = "html_file.csv"
# error_log = "error_log.csv"

# with open(html_file, "w") as html:
#     html.write("category,html\n")
# with open(error_log, "w") as error:
#     error.write("headline,error\n")

# cpu_count = int(mp.cpu_count() * 0.95)

# data = pd.read_json("News_Category_Dataset_v2.json", lines=True)[1:20001]
# data = [df.to_dict(orient="records") for df in np.split(data, range(int(len(data.index)/cpu_count), len(data.index), int(len(data.index)/cpu_count)))]

# print(len(data))
# [print(f"{len(df)}\n") for df in data]

# def get_html(row):
#     try:
#         print(f"in get_html doing for {row['headline']}")
#         time.sleep(0.15)
#         content = requests.get(row['link']).content
#         soup = bs4(content, 'html.parser')
#         res = [p.get_text() for p in soup.find_all("p")]
#         text = " ".join(res).replace(",", ".")

#         with open(html_file, "a") as html:
#             html.write(f"{row['category']},{str(text.encode('utf-8'))}\n")
#         return(1)
#     except Exception as e:
#         with open(error_log, "a") as error:
#             error.write(f"{row['headline']},{e.message}\n")
# if __name__ == "__main__":
#     pool = mp.Pool(cpu_count)
#     res = [pool.map_async(get_html, dt).get() for dt in data]

#     print(f"{sum(list(itertools.chain.from_iterable(res)))} got")

#     pool.close()
