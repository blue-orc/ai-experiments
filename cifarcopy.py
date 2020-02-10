
import os 
import shutil 
    
# This script takes the original 5 files in the cifar dataset
# and copies them to get a larger dataset of 1,000 files
  
# Adjust the path as necessary 
dataSourcePath = '/home/opc/cifar-10-batches-py/'
destinationPath = '/mnt/cifar-data/'

for y in range(0,200):
    j = y*5
    print(str(y))
    for x in range(1,6):
        
        source = dataSourcePath + 'data_batch_'+ str(x)
        destination = destinationPath + 'data_batch_'+ str(j+x)
 
        dest = shutil.copyfile(source, destination) 

