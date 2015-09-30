from skimage import io
import numpy as np
import cPickle
#from svmutil import *
channels = 3
row_pixels = 256
col_pixels = 256
from os import listdir

img_path = '/home/deepak/Documents/secondSem/machineLearning/asgnmnt/PA-1/Datasets/DS4/data_students/coast/Test/coast_natu128.jpg'
data_location = '/home/deepak/Documents/secondSem/machineLearning/asgnmnt/PA-1/Datasets/DS4/data_students/'
img_types = ['coast', 'forest', 'insidecity', 'mountain']
folders = ['Test', 'Train']

def gen_fea_vec(data_path, im_type, folder, filename):
    img_path = data_path + im_type + '/' + folder + '/' + filename
    img = io.imread(img_path)
    feat_vec = np.zeros(96)
    for i in range(3):
        for j in range(row_pixels):
            for k in range(col_pixels):
                bin = img[j,k,i]/8
                bin += i*32
                feat_vec[bin] += 1
    return feat_vec


for img_type in img_types:
    for fol in folders:
        feat_vectors = []
        for img_file in listdir(data_location+img_type+'/'+fol):
            feat_vectors.append(gen_fea_vec(data_location, img_type, fol, img_file))
        pickl_file_name = img_type + '_' + fol + '.save'
        f = open('./4_que_data/'+pickl_file_name, 'wb')
        cPickle.dump(feat_vectors, f, protocol = cPickle.HIGHEST_PROTOCOL)
        f.close()

np.set_printoptions(suppress=True)
X = gen_fea_vec(data_location, 'coast', 'Test', 'coast_natu128.jpg')
#print listdir('/home/deepak/Documents/secondSem/machineLearning/asgnmnt/PA-1/Datasets/DS4/data_students/coast/Test/')
print feat_vectors[0]