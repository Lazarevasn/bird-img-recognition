"Module used for loading the file from an URL"
import requests

def download_file_from_url(url: str, destination_path: str) -> None:
    '''
    Downloads a file from a URL into a local file with an assigned path.
      Parameters:
        url(str): URL address from which contains the file that needs to be downloaded
        destination_path(str): local path (including the filename) where the file wll be downloaded
      Returns:
        None
    '''
    response = requests.get(url, stream = True, timeout = 20)
    with open(destination_path, 'wb') as file:
        for chunk in response.iter_content(chunk_size = 128):
            file.write(chunk)
