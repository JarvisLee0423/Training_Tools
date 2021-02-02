# Training_Tools
 Simple Training Tools for Pytorch

 This training tools contains three parts:

    - Data Preprocessor is used to preprocess the training data. (It contains MNIST, CIFAR10 and ImageFolder data preprocessor)

    - Info Logger is used to log the training information with visdom and logging.

    - Params Handler is used to handle the hyperparameters. (It contains easydict and argparse)

For training:

    - First, please write the book code in the Params.txt file with format 'ParamsName:value' like:

            learningRate:0.1
        
        And one line for one hyperparameter.
    
    - Second, please write the model class with pytorch method 'nn.Module' in Model folder.

    - Third, please modify the corresponding parts in Trainer.py file for training.

For execution:

    There are two ways to train the model.

    One is setting the hyperparameters in Params.txt file by using book code like above. And execute the Trainer.py file directly.

    The other way is that after setting the default value in Params.txt file, directly use the command below to train the model:

        python Trainer.py -Params1 value -Params2 value ...
