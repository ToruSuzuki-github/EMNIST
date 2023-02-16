from sys import stdout
import emnist
import numpy as np
import pandas as pd
import os
from copy import copy

#binary_threshold未満or以上でpixel値を二値化する関数
def binarizationDataset(dataset, binary_threshold):
    return np.array([np.where(data<binary_threshold, 0, 1) for data in copy(dataset)])

#pixel値を正規化
def normalizationDataset(dataset):
    return np.array([data/255 for data in copy(dataset)])

#データを一次元化
def unidimensionalizationDataset(dataset):
    return np.array([data.ravel() for data in copy(dataset)])

#特徴抽出を行う関数
def featureExtraction(unfinished_flag,unfinished_data_num,binary_threshold,binarization_flag):

    #------------EMNIST to ndarray------------
    (training_dataset, training_labelset) = emnist.extract_training_samples('digits')
    (test_dataset, test_labelset) = emnist.extract_test_samples('digits')

    if unfinished_flag:
        training_dataset=copy(training_dataset)[:unfinished_data_num]
        test_dataset=copy(test_dataset)[:unfinished_data_num]
        training_labelset=copy(training_labelset)[:unfinished_data_num]
        test_labelset=copy(test_labelset)[:unfinished_data_num]
    
    #試験用
    '''
    np.set_printoptions(threshold=np.inf)
    for data in training_dataset:
        stdout.write('\n\noriginal:\n'+str(data))
        stdout.flush()
    np.set_printoptions(threshold=1000)
    '''

    #------------二値化------------
    if binarization_flag:
        stdout.write('\nbinarization now')
        stdout.flush()
        training_dataset=binarizationDataset(training_dataset, binary_threshold)
        test_dataset=binarizationDataset(test_dataset, binary_threshold)
        stdout.write('\nbinarization completed')
        stdout.flush()
    #------------正規化------------
    else:
        stdout.write('\nnormalization now')
        stdout.flush()
        training_dataset=normalizationDataset(training_dataset)
        test_dataset=normalizationDataset(test_dataset)
        stdout.write('\nnormalization completed')
        stdout.flush()
    
    #試験用
    '''
    np.set_printoptions(threshold=np.inf)
    for data in training_dataset:
        stdout.write('\n\nbinarization or normalization\n'+str(data))
        stdout.flush()
    np.set_printoptions(threshold=1000)
    '''

    #------------特徴抽出------------
    #特徴量データセット
    feature_training_dataset=pd.DataFrame(training_labelset.T,columns=['labels'])
    feature_test_dataset=pd.DataFrame(test_labelset.T,columns=['labels'])
    #一次元化
    stdout.write('\nunidimensionalization now')
    stdout.flush()
    feature_names=['pixel_value_'+str(pixcel_num) for pixcel_num in range(len(training_dataset[0])*len(training_dataset[0][0]))]
    feature_training_dataset=pd.concat([feature_training_dataset,pd.DataFrame(unidimensionalizationDataset(training_dataset),columns=feature_names)],axis=1)
    feature_test_dataset=pd.concat([feature_test_dataset,pd.DataFrame(unidimensionalizationDataset(test_dataset),columns=feature_names)],axis=1)
    stdout.write('\nunidimensionalization completed')
    stdout.flush()

    #------------ファイル出力------------
    stdout.write('\noutputing now')
    stdout.flush()
    """
    stdout.write('\n\nfeature training:\n'+str(feature_training_dataset))
    stdout.write('\n\nfeature test:\n'+str(feature_test_dataset))
    """
    output_data=pd.concat([feature_training_dataset,feature_test_dataset],ignore_index=True)
    """
    stdout.write('\n\noutput data:\n'+str(output_data))
    stdout.flush()
    return True
    """
    output_file_path='./EMNIST/feature/pixel_value'
    if binarization_flag:
        output_file_path+='/binarization'
    else:
        output_file_path+='/nomal'
    output_file_path+='/'+str(len(feature_training_dataset))+'_pixel_value.csv'
    os.makedirs(os.path.dirname(output_file_path),exist_ok=True)
    output_data.to_csv(output_file_path)
    stdout.write('\noutputing completed')
    stdout.write('\noutput file path: '+str(output_file_path))
    stdout.flush()

    return True