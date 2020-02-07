import torch.utils.data
import pickle
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, file):
        ds = unpickle(file)
        self.data = ds[b'data']
        self.target = ds[b'labels']
        
    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]
        
        return x, y
    
    def __len__(self):
        return len(self.target)

for x in range(1,11):
    base = "/Users/jmblau/Documents/cifar-10-batches-py/data_batch_"
    t = MyDataset(base+ str(x))
    print(x)
# b'data' and b'labels'