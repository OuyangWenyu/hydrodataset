"""
Download NLDAS forcing data from NASA Earth Data service

Before downloading, you have to register an account for NASA Earth Data service.
In addition, it requires us some manual operation:
- chose data in https://disc.gsfc.nasa.gov/datasets/NLDAS_FORA0125_H_2.0/summary
- set the time period range
- download the data-downloading-list and put it in current directory
Then, use the following code to download the data

Notice: if you are in China or somewhere the GFW exists, you need scientific internet access!!!!!!!
Please download it year by year, if you chose too much file once a time, the downloading will fail.
"""
import os
import time
from netrc import netrc

import requests
from tqdm import tqdm

from hydrobench.utils.hydro_utils import hydro_logger


# overriding requests.Session.rebuild_auth to mantain headers when redirected
class SessionWithHeaderRedirection(requests.Session):
    AUTH_HOST = 'urs.earthdata.nasa.gov'

    def __init__(self, username, password):
        super().__init__()
        self.auth = (username, password)

    # Overrides from the library to keep headers when redirected to or from
    # the NASA auth host.
    def rebuild_auth(self, prepared_request, response):
        headers = prepared_request.headers
        url = prepared_request.url
        if 'Authorization' in headers:
            original_parsed = requests.utils.urlparse(response.request.url)
            redirect_parsed = requests.utils.urlparse(url)
            if (original_parsed.hostname != redirect_parsed.hostname) \
                    and redirect_parsed.hostname != self.AUTH_HOST \
                    and original_parsed.hostname != self.AUTH_HOST:
                del headers['Authorization']
        return


def download_nldas_with_url_lst(url_lst_file, save_dir):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    if not os.path.isfile(url_lst_file):
        raise FileNotFoundError("Please download a Data File with Links for NLDAS data from:"
                                "https://disc.gsfc.nasa.gov/datasets/NLDAS_FORA0125_H_2.0/summary")
    netrc_dir = os.path.expanduser(os.path.join("~", ".netrc"))
    if not os.path.isfile(netrc_dir):
        raise FileNotFoundError("Please create a .netrc file to save username and password of your earthdata account:\n"
                                "urs.earthdata.nasa.gov")
    urs = 'urs.earthdata.nasa.gov'
    # The EarthData username string
    username = netrc(netrc_dir).authenticators(urs)[0]
    # The EarthData password string
    password = netrc(netrc_dir).authenticators(urs)[2]
    url_lst = []
    with open(url_lst_file, 'r') as f:
        count = 0
        for line in f:
            if count > 0:
                if line[-1] == "\n":
                    url_lst.append(str(line[:-1]))
                else:
                    url_lst.append(str(line))
            count = count + 1
        f.close()
    session = SessionWithHeaderRedirection(username, password)
    for url in tqdm(url_lst):
        # extract the filename from the url to be used when saving the file
        filename = os.path.join(save_dir, url.split("/")[-1])
        if os.path.isfile(filename):
            hydro_logger.info("Downloaded: ", filename)
            continue
        # if we need to download a data file, be patient, sleep for a little while before downloading
        time.sleep(0.1)
        try:
            # submit the request using the session
            response = session.get(url, stream=True)
            hydro_logger.info("Downloading: ", filename)
            # raise an exception in case of http errors
            response.raise_for_status()
            # save the file
            with open(filename, 'wb') as fd:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    fd.write(chunk)
        except requests.exceptions.HTTPError as e:
            # handle any errors here
            print(e)
