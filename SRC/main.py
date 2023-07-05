# Performance and visualization
import os
import wandb
import multiprocessing

# Data manipulation packages
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
import random

# Own packages
from train import *
from test import *
from utils.utils import *
from models.models import *

import warnings
warnings.filterwarnings("ignore") # Ignoring bleu score innecessary warnings.

# Global variables
global device

# Environment variables (different for each computer)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128" # Setting CUDA ALLOC split size to 128 to avoid running out of memory
os.environ["WANDB_DISABLE_SYMLINKS"] = "true" # Stopping wandb from creating symlinks

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def model_pipeline(cfg: dict):
    """
    Main loop containing the train-test tasks. All parameter information is included on the config dictionary.
    
    Parameters
    -----------
    cfg: dictionary.
    	Contains all the specific parameters to execute the pipeline in string format.
        # Paths
        root_dir: Directory where the images are stored.
        captions_file: Directory where the captions are stored.
        DATA_LOCATION: Father directory of the images and captions. Where all the data will be storage.
        save: Boolean, if True then execution logs will be storaged.

        # Training data
        epochs: Number of epoch.
        batch_size: Batch size to split the data.
        train_size: Train size (in percentage).
        
        # Model data
        optimizer: Name of the optimizer. 'Adam', 'Adagrad', 'SGD'.
        criterion: Name of the criterion used. 'CrossEntropy', 'MSE'
        learning_rate: Learning rate used for the optimizer.
        momentum: Momentum used for the optimizer if required.
        device: Device used. 'CPU' or 'cuda:0'.
        encoder: Pre-trained net used for encoding the data. 'ResNet50', 'ResNet152', 'googleNet', 'VGG'.
        transforms: Transforms applied to the images. 'transforms_2' is a bit faster.
        embed_size: Embedding size. Recomended: 300
        attention_dim: Dimension for the attention. Recomended: 256.
        encoder_dim: Dimension for the encoder. Depends on the encoder chosen. Recomended: 2048 (ResNet). 512 (VGG). 1024 (GoogleNet)
        decoder_dim: Dimension for the decoder. Recomended: 512.
        
    Returns
    ---------
    Trained Model.
    """
    
    with wandb.init(project="pytorch-demo", config=cfg): # Starting wandb
        config = wandb.config # access all HPs through wandb.config, so logging matches execution!

        # Execute only once to create the dataset
        # generate_and_dump_dataset(config.root_dir, config.captions_file, config.transforms, cfg.DATA_LOCATION)

        # Generate Dataset
        dataset = make_dataset(config)

        # Get the data loaders
        train_loader, test_loader = make_dataloaders(config, dataset, 1)

        # Extract vocab
        vocab = dataset.vocab
        config.vocab_size = len(vocab)

        # Get the model
        my_model = make_model(config, device)

        # Define the loss and optimizer
        criterion = get_criterion(config.criterion, vocab.stoi["<PAD>"])
        criterion.ignore_index=vocab.stoi["<PAD>"]
        
        optimizer = get_optimizer(config.optimizer, my_model.parameters(), config.learning_rate, config.momentum)
        
        # Arrays to log data
        train_loss_arr_epoch, test_loss_arr_epoch, acc_arr_epoch  = [], [], [] # Epoch-wise
        train_loss_arr_batch, test_loss_arr_batch, acc_arr_batch = [], [], [] # Batch-wise
        train_execution_times, test_execution_times = [], [] # Execution times

        # Main loop
        for epoch in tqdm(range(1, config.epochs + 1)):
            # Training
            my_model.train()
            train_loss_arr_aux, train_time = train(my_model, train_loader, criterion, optimizer, config, epoch)
            my_model.eval()

            # Testing
            acc_arr_aux, test_loss_arr_aux, test_time = test(my_model, test_loader, criterion, vocab, config, device)

            # Check how model performs
            test_model_performance(my_model, test_loader, device, vocab, epoch, config)
            
            # Logging data for vizz
            train_loss_arr_epoch.append(np.mean(train_loss_arr_aux)); test_loss_arr_epoch.append(np.mean(test_loss_arr_aux))
            train_loss_arr_batch += train_loss_arr_aux; test_loss_arr_batch += test_loss_arr_aux
            acc_arr_epoch.append(np.mean(acc_arr_aux)); acc_arr_batch += acc_arr_aux
            train_execution_times.append(train_time); test_execution_times.append(test_time)

        # Saving the logs
        if config.save:
            export_data(train_loss_arr_epoch, test_loss_arr_epoch, acc_arr_epoch, train_execution_times, test_execution_times,
                   train_loss_arr_batch, acc_arr_batch, test_loss_arr_batch, config)
            
            save_model(my_model, config, config.DATA_LOCATION+'/logs'+'/EncoderDecorder_model.pth')

    return my_model


if __name__ == "__main__":
    wandb.login()

    print("Using: ", device)

    # Resized and then crops (maintains perspective)
    transforms = T.Compose([
        T.Resize(226),
        T.RandomCrop(224),
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    # Directly resized (doesn't maintain perspective)
    transforms_2 = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    DATA_LOCATION = '../data'

    config = dict(
        # Paths
        root_dir=DATA_LOCATION+"/Images",
        captions_file=DATA_LOCATION+"/captions.txt",
        DATA_LOCATION=DATA_LOCATION,
        save=True,

        # Training data
        epochs=10,
        batch_size=50,
        train_size=0.8,
        
        # Model data
        optimizer='Adam',
        criterion='CrossEntropy',
        learning_rate=3e-4,
        device=device,
        encoder='ResNet152',
        transforms=transforms_2,
        embed_size=300,
        attention_dim=256,
        encoder_dim=2048,
        decoder_dim=512,
        momentum=0.8
    )

    model = model_pipeline(config)
