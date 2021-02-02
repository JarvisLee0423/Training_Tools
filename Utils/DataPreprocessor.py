'''
    @Copyright:     JarvisLee
    @Date:          2021/1/31
'''

# Import the necessary library.
import os
import numpy as np
import pickle
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

# Set the class to encapsulate all the functions.
class Preprocessor():
    '''
        This class is used to encapsulate all the functions which is used to preprocess the datasets.\n
        This class contains four parts:\n
            - 'Decomposer' is used to decompose the archive.
            - 'ImageNet' is used to preprocess the image data.
            - 'MNIST' is used to preprocess the MNIST data.
            - 'CIFAR10' is used to preprocess the CIFAR10 data.
    '''
    # Set the function to decompose the archive.
    def Decomposer(filename):
        '''
            This function is used to decompose the archive.\n
            Params:\n
                - filename: The name of the archive.
        '''
        # Open the file.
        file = open(filename, 'rb')
        # Get the data in the file.
        dict = pickle.load(file, encoding = 'bytes')
        # Return the data.
        return dict
    
    # Set the function to preprocess the image data.
    def ImageNet(batchSize, mean, std, cropSize, trainRoot = './train', valRoot = './val'):
        '''
            This function is used to preprocess the image data.\n
            Params:\n
                - batchSize: The size of the each batch.
                - mean: The mean of the dataset.
                - std: The standard deviation of the dataset.
                - cropSize: The size of the data in the training and validation datasets.
                - trainRoot: The root of the training dataset.
                - valRoot: The root of the validation dataset. 
        '''
        # Indicate whether the parameters are valid.
        assert len(mean) == 3, 'The mean must be the list with length equal 3.'
        assert len(std) == 3, 'The std must be the list with length equal 3.'
        # Set the normalization method.
        normalize = transforms.Normalize(
            mean = mean,
            std = std
        )
        # Initialize the data.
        trainData = 0
        valData = 0
        # Get the training dataset.
        if os.path.exists(trainRoot):
            trainData = DataLoader(
                datasets.ImageFolder(
                    root = trainRoot,
                    transform = transforms.Compose([
                        transforms.Resize((cropSize, cropSize)),
                        transforms.ToTensor(),
                        normalize
                    ])
                ),
                batch_size = batchSize,
                shuffle = True,
                drop_last = True
            )
        # Get the validation dataset.
        if os.path.exists(valRoot):
            valData = DataLoader(
                datasets.ImageFolder(
                    root = valRoot,
                    transform = transforms.Compose([
                        transforms.Resize((cropSize, cropSize)),
                        transforms.ToTensor(),
                        normalize
                    ])
                ),
                batch_size = batchSize,
                shuffle = False,
                drop_last = False
            )
        # Return the datasets.
        return trainData, valData
    
    # Set the function to preprocess the MNIST dataset.
    def MNIST(root, batchSize):
        '''
            This function is used to preprocess the MNIST dataset.\n
            Params:\n
                - root: The root of the dataset.
                - batchSize: The size of the each batch.
        '''
        # Check whether the dataset exists or not.
        if os.path.exists(root + '/MNIST/'):
            download = False
        else:
            download = True
        # Get the training dataset.
        trainData = DataLoader(
            datasets.MNIST(
                root = root,
                train = True,
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (1.0,))
                ]),
                download = download
            ),
            batch_size = batchSize,
            shuffle = True,
            drop_last = True
        )
        # Get the validation dataset.
        valData = DataLoader(
            datasets.MNIST(
                root = root,
                train = False,
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (1.0,))
                ]),
                download = download
            ),
            batch_size = batchSize,
            shuffle = False
        )
        # Return the datasets.
        return trainData, valData
    
    # Set the function to preprocess the CIFAR10 dataset.
    def CIFAR10(root, batchSize):
        '''
            This function is used to preprocess the CIFAR10 dataset.\n
            Params:\n
                - root: The root of the dataset.
                - batchSize: The size of the each batch.
        '''
        # Decompose the dataset.
        data1 = Preprocessor.Decomposer(f'{root}/CIFAR10/data_batch_1')
        data2 = Preprocessor.Decomposer(f'{root}/CIFAR10/data_batch_2')
        data3 = Preprocessor.Decomposer(f'{root}/CIFAR10/data_batch_3')
        data4 = Preprocessor.Decomposer(f'{root}/CIFAR10/data_batch_4')
        data5 = Preprocessor.Decomposer(f'{root}/CIFAR10/data_batch_5')
        test = Preprocessor.Decomposer(f'{root}/CIFAR10/test_batch')
        # Get the data.
        ds = []
        dlabels = []
        test_ds = []
        test_dlabels = []
        for i in range(10000):
            im = np.reshape(data1[b'data'][i],(3, 32, 32))
            ds.append(im)
            dlabels.append(data1[b'labels'][i])
        for i in range(10000):
            im = np.reshape(data2[b'data'][i],(3, 32, 32))
            ds.append(im)
            dlabels.append(data2[b'labels'][i])
        for i in range(10000):
            im = np.reshape(data3[b'data'][i],(3, 32, 32))
            ds.append(im)
            dlabels.append(data3[b'labels'][i])
        for i in range(10000):
            im = np.reshape(data4[b'data'][i],(3, 32, 32))
            ds.append(im)
            dlabels.append(data4[b'labels'][i])
        for i in range(10000):
            im = np.reshape(data5[b'data'][i],(3, 32, 32))
            ds.append(im)
            dlabels.append(data5[b'labels'][i])
        for i in range(10000):
            im = np.reshape(test[b'data'][i],(3, 32, 32))
            test_ds.append(im)
            test_dlabels.append(test[b'labels'][i])
        # Get the training dataset and validation dataset.
        train = torch.utils.data.TensorDataset(torch.Tensor(ds), torch.LongTensor(dlabels))
        test = torch.utils.data.TensorDataset(torch.Tensor(test_ds), torch.LongTensor(test_dlabels))
        trainData = torch.utils.data.DataLoader(train, batch_size = batchSize, shuffle = True, drop_last = True)
        valData = torch.utils.data.DataLoader(test, batch_size = batchSize)
        # Return the data.
        return trainData, valData

# Test the codes.
if __name__ == "__main__":
    trainSet, devSet = Preprocessor.MNIST('.\Datasets', 32)
    print(type(trainSet))
    print(type(devSet))
    trainSet, devSet = Preprocessor.CIFAR10('.\Datasets', 32)
    print(type(trainSet))
    print(type(devSet))
    trainSet, devSet = Preprocessor.ImageNet(32, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225], 224, './train', './val')
    print(type(trainSet))
    print(type(devSet))
    trainSet, devSet = Preprocessor.ImageNet(32, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5], 64, './Datasets')
    print(type(trainSet))
    print(type(devSet))