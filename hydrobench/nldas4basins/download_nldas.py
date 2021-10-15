"""
Download NLDAS forcing data from NASA Earth Data service

Before downloading, you have to register an account for NASA Earth Data service.
In addition, it requires us some manual operation:
- chose data in https://disc.sci.gsfc.nasa.gov/datasets/NLDAS_FORA0125_H_002/summary?keywords=NLDAS
- download the data-downloading-list and put it in current directory
Then, use the following code to download the data

Notice: if you are in China or somewhere the GFW exists, you need a scientific internet!!!!!!!
"""
import os

from pydap.client import open_url
from pydap.cas.urs import setup_session

earth_data_account_file = os.path.join(os.path.split(os.path.realpath(__file__))[0], "earth_data")


def download_nldas_with_url_lst(url_lst_file, save_dir):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    with open(earth_data_account_file, 'r') as f:
        count = 0
        for line in f:
            if count == 0:
                username = str(line[:-1])
            else:
                if line[-1] == "\n":
                    password = str(line[:-1])
                else:
                    password = str(line)
            count = count + 1
        f.close()
    url_lst = []
    with open(url_lst_file, 'r') as f:
        count = 0
        for line in f:
            if count > 0:
                url_lst.append(str(line))
            count = count + 1
        f.close()
    for dataset_url in url_lst:
        session = setup_session(username, password, check_url=dataset_url)
        dataset = open_url(dataset_url, session=session)
        print("Downloading data")
