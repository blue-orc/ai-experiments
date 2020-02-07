
# Python program to explain shutil.copyfile() method  
    
# importing os module  
import os 
  
# importing shutil module  
import shutil 
  
# path 
path = '/home/opc/cifar-10-batches-py/'

for y in range(101,300):
    j = y*5
    for x in range(1,6):
        # Source path 
        source = path + 'data_batch_'+ str(x)
        
        # Destination path 
        destination = path + 'data_batch_'+ str(j+x)
        
        # Copy the content of 
        # source to destination 
        dest = shutil.copyfile(source, destination) 

