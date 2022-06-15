import wget
import pandas as pd

URL_VAL = "https://desycloud.desy.de/index.php/s/llbX3zpLhazgPJ6/download?path=%2F&files=val.h5"
URL_TEST = "https://desycloud.desy.de/index.php/s/llbX3zpLhazgPJ6/download?path=%2F&files=test.h5"
URL_TRAIN = "https://desycloud.desy.de/index.php/s/llbX3zpLhazgPJ6/download?path=%2F&files=train.h5"

# download!
print('Download training dataset...')
wget.download(URL_TRAIN, out="train.h5")
print('\nDownload validation dataset...')
wget.download(URL_VAL, out="val.h5")
print('\nDownload test dataset...')
wget.download(URL_TEST, out="test.h5") 