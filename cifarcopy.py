
# Python program to explain shutil.copyfile() method  
    
# importing os module  
import os 
  
# importing shutil module  
import shutil 
  
# path 
path = '/home/opc/cifar-10-batches-py/'

for y in range(0,200):
    j = y*5
    for x in range(1,6):
        # Source path 
        source = path + 'data_batch_'+ str(x)
        
        # Destination path 
        destination = '/mnt/cifar-data/' + 'data_batch_'+ str(j+x)
        
        # Copy the content of 
        # source to destination 
        dest = shutil.copyfile(source, destination) 

