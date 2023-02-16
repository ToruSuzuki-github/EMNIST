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

#全て0の行若しくは列を削除
def removeSurplus(dataset):
    removed_dataset=[data for data in copy(dataset)]
    for data_num,data in enumerate(dataset):
        remove_row=[row_num for row_num,row in enumerate(data) if sum(row)==0]
        remove_col=[col_num for col_num,col in enumerate(data.T) if sum(col)==0]
        if remove_row:
            removed_dataset[data_num]=np.delete(removed_dataset[data_num],remove_row,0)
        if remove_col:
            removed_dataset[data_num]=np.delete(removed_dataset[data_num],remove_col,1)
        stdout.flush()
    return removed_dataset

#データをnx × nyのメッシュに分割する関数
def meshingDataset(dataset, nx, ny):
    mesh_dataset=[]
    for data in copy(dataset):
        mesh_data=[]
        for split_x_dataset in np.array_split(data,nx,1):
            mesh_data+=np.array_split(split_x_dataset,ny,0)
        mesh_dataset.append(mesh_data)
    return mesh_dataset

#メッシュ毎の平均合計要素値数を計算
def averageTotalElementValue(mesh_dataset):
        return np.array([[np.mean(mesh.flatten())for mesh in mesh_data] for mesh_data in copy(mesh_dataset)])

#特徴抽出を行う関数
def featureExtraction(mesh_x_num,mesh_y_num,unfinished_flag,unfinished_data_num,binary_threshold,binarization_flag,remove_surplus_flag):

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

    #------------全てが0の行と列を削除------------
    if remove_surplus_flag:
        stdout.write('\nremove surplus now')
        stdout.flush()
        training_dataset=removeSurplus(training_dataset)
        test_dataset=removeSurplus(test_dataset)
        stdout.write('\nremove surplus completed')
        stdout.flush()
        
        #試験用
    '''
        np.set_printoptions(threshold=np.inf)
        for data in training_dataset:
            stdout.write('\n\nremove surplus\n'+str(data))
            stdout.flush()
        np.set_printoptions(threshold=1000)
    '''

    #------------特徴抽出------------
    #特徴量データセット
    feature_training_dataset=pd.DataFrame(training_labelset.T,columns=['labels'])
    feature_test_dataset=pd.DataFrame(test_labelset.T,columns=['labels'])

    #Nx×Nyメッシュの作成
    stdout.write('\nmeshing now')
    stdout.flush()
    mesh_training_dataset=meshingDataset(training_dataset,mesh_x_num,mesh_y_num)
    mesh_test_dataset=meshingDataset(test_dataset,mesh_x_num,mesh_y_num)
    stdout.write('\nmeshing completed')
    stdout.flush()

    #メッシュ毎の平均合計要素値数を計算
    stdout.write('\ncalculation mesh average now')
    stdout.flush()
    feature_names=['average_mesh_'+str(mesh_x_num)+'x'+str(mesh_y_num)+'_'+str(mesh_num) for mesh_num in range(mesh_x_num*mesh_y_num)]
    feature_training_dataset=pd.concat([feature_training_dataset,pd.DataFrame(averageTotalElementValue(mesh_training_dataset),columns=feature_names)],axis=1)
    feature_test_dataset=pd.concat([feature_test_dataset,pd.DataFrame(averageTotalElementValue(mesh_test_dataset),columns=feature_names)],axis=1)
    stdout.write('\ncalculation mesh average completed')
    stdout.flush()
    
    #試験用
    '''
    np.set_printoptions(threshold=np.inf)
    stdout.write('\n\nfeature:\n'+str(feature_training_dataset))
    stdout.flush()
    np.set_printoptions(threshold=1000)
    '''

    #------------ファイル出力------------
    stdout.write('\noutputing now')
    stdout.flush()
    output_data=pd.concat([feature_training_dataset,feature_test_dataset],ignore_index=True)
    """
    stdout.write('\n\nfeature training:\n'+str(feature_training_dataset))
    stdout.write('\n\nfeature test:\n'+str(feature_test_dataset))
    stdout.write('\n\noutput data:\n'+str(output_data))
    stdout.flush()
    return True
    """

    output_file_path='./EMNIST/feature/mesh'
    if remove_surplus_flag and binarization_flag:
        output_file_path+='/remove_binarization'
    elif remove_surplus_flag:
        output_file_path+='/only_remove'
    elif binarization_flag:
        output_file_path+='/onry_binarization'
    else:
        output_file_path+='/nomal'
    output_file_path+='/'+str(len(feature_training_dataset))+'_mesh_'+str(mesh_x_num)+'x'+str(mesh_y_num)+'.csv'
    os.makedirs(os.path.dirname(output_file_path),exist_ok=True)
    output_data.to_csv(output_file_path)
    stdout.write('\noutputing completed')
    stdout.write('\noutput file path: '+str(output_file_path))
    stdout.flush()

    return True