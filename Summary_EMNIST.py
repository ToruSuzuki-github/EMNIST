from sys import stdout,argv
import numpy as np
import emnist
from copy import copy

if argv[1]=='help':
    stdout.write('\nThis program file does not require a second command line argument or later.')

#------------EMNIST to ndarray------------
(training_data, training_label) = emnist.extract_training_samples('digits')
(test_data, test_label) = emnist.extract_test_samples('digits')

#------------データセットの概要調査------------
'''
#stdout.write('\ntraining_data:\n'+str(training_data))
stdout.write('\n\ntraining_data type:\n'+str(type(training_data)))
stdout.write('\ntraining_data length:\n'+str(len(training_data)))
#stdout.write('\ntraining_label:\n'+str(training_label))
stdout.write('\ntraining_label type:\n'+str(type(training_label)))
stdout.write('\ntraining_label length:\n'+str(len(training_label)))

#stdout.write('\ntest_data:\n'+str(test_data))
stdout.write('\n\ntest_data type:\n'+str(type(test_data)))
stdout.write('\ntest_data length:\n'+str(len(test_data)))
#stdout.write('\ntest_label:\n'+str(test_label))
stdout.write('\ntest_label type:\n'+str(type(test_label)))
stdout.write('\ntest_label length:\n'+str(len(test_label)))

stdout.write('\n\ntotal length:\n'+str(len(training_label)+len(test_label)))
'''

#------------例として1つのデータを描画------------
np.set_printoptions(threshold=np.inf)
for num in range(10):
    stdout.write('\n\n'+str(training_data[num]))
    stdout.flush()
np.set_printoptions(threshold=1000)

#------------二値化------------
binary_threshold=128
binary_training_data=training_data.copy()
for num in range(10):
    for row in range(len(training_data[num])):
        for pixel in range(len(training_data[num][row])):
            binary_training_data[num][row][pixel]=training_data[num][row][pixel]//binary_threshold
    stdout.write('\n\n'+str(binary_training_data[num]))
    stdout.flush()

stdout.write('\n')
stdout.flush()