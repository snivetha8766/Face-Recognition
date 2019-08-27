# fileid="1V0ELMWF5qSmCF4rOuKChnizgp-w49I5a"
# filename='fer2013.csv'

# from google_drive_downloader import GoogleDriveDownloader as gdd
# gdd.download_file_from_google_drive(file_id=fileid,
#                                     dest_path='./'+filename,
#                                     unzip=False)


import pandas
data = pandas.read_csv("./fer2013.csv")
data.head()
print('Samples distribution across Usage:')
print(data.Usage.value_counts())
print('Samples per emotion:')
print(data.emotion.value_counts())

print('Number of pixels for a sample:')
print(len(data.pixels[0].split(' ')))
train_set = data[(data.Usage == 'Training')]
validation_set = data[(data.Usage == 'PublicTest')]
test_set = data[(data.Usage == 'PrivateTest')]
emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
num_classes = len(emotion_labels)

from math import sqrt
depth = 1
height = int(sqrt(len(data.pixels[0].split())))
width = height

#@title Choose a sample to visualize
sample_number = 656 #@param {type:"slider", min:0, max:1000, step:1}

import numpy as np
import scipy.misc
from IPython.display import display

array = np.mat(data.pixels[sample_number]).reshape(48,48)
image = scipy.misc.toimage(array)
display(image)
print(emotion_labels[data.emotion[sample_number]])

# X_train = np.array(map(str.split, train_set.pixels), np.float32)
# X_validation = np.array(map(str.split, validation_set.pixels), np.float32)
# X_test = np.array(map(str.split, test_set.pixels), np.float32)

# num_train = X_train.shape[0]
# num_validation = X_validation.shape[0]
# num_test = X_test.shape[0]

# X_train = X_train.reshape(num_train, width, height, depth)
# X_validation = X_validation.reshape(num_test, width, height, depth)
# X_test = X_test.reshape(num_test, width, height, depth)

# print('Training: ',X_train.shape)
# print('Validation: ',X_validation.shape)
# print('Test: ',X_test.shape)
