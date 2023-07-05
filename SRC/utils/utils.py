import pandas as pd
from collections import Counter
import spacy
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import joblib
from copy import deepcopy
from sklearn.model_selection import train_test_split
from models.models import *
import time


def get_criterion(name, ignoring_index):
    """
    Sets up the criterion used for training the model.
    
    Params:
    --------
    name: Name of the criterion. 'CrossEntropy', 'MSE'.
    ignoring_index: Indexes to be ignored when calculating the criterion.
    
    Returns:
    --------
    Criterion.
    """
    if name == 'CrossEntropy':
        return nn.CrossEntropyLoss(ignore_index=ignoring_index)
    elif name == 'MSE':
        return torch.nn.MSELoss()
    else:
        print("WRONG CRITERION")
        return nn.CrossEntropyLoss(ignore_index=ignoring_index)
    
    
def get_optimizer(name, params, lr, momentum=None):
    """
    Sets up the optimizer used for training the model.
    
    Params:
    --------
    name: Name of the optimizer. 'Adam', 'Adagrad', 'SGD'.
    lr: Learning rate of the optimizer.
    momentum: Momentum of the optimizer if required.
    
    Returns:
    --------
    Optimizer.
    """
    if name == 'Adam':
        return torch.optim.Adam(params=params, lr=lr)
    elif name == 'Adagrad':
        return torch.optim.Adagrad(params=params, lr=lr)
    elif name == 'SGD':
        return torch.optim.SGD(params=params, lr=lr, momentum=momentum)
    else:
        print("WRONG Optimizer")
        return torch.optim.Adam(params=params, lr=lr)

    
def export_data(train_loss_arr_epoch, test_loss_arr_epoch, acc_arr_epoch, train_execution_times, test_execution_times,
                   train_loss_arr_batch, acc_arr_batch, test_loss_arr_batch, config):
    """
    Exports the data logs into a folder named "logs" inside the data location indicated. It separates the logs into three files:
    - epoch_df: Train loss, test loss, test accuracy, train execution times and test execution times for each epoch.
    - loss_batch_df: Train loss for each batch.
    - acc_batch_df: Accuracy and loss from each test batch.
    
    Params:
    --------
    train_loss_arr_epoch: float Array.
    	Array with the train losses for each epoch.
        
    test_loss_arr_epoch: float Array.
    	Array with the test losses for each epoch.
        
    acc_arr_epoch: float Array.
    	Array with the test accuracies for each epoch.
        
    train_execution_times: float Array.
    	Array with the execution times for the train each epoch.
        
    test_execution_times: float Array.
    	Array with the execution times for the test each epoch.
        
    train_loss_arr_batch: float Array.
    	Array with the train losses for the train each batch.
        
    test_loss_arr_batch: float Array.
    	Array with the train losses for the test each batch.
        
    acc_arr_batch: float Array.
    	Array with the test accuracies for each batch.
        
	config: Dictionary.
    	Contains hyperparameter information. Needs to contain a "DATA_LOCATION" key with the data location.
    """
    
    epoch_df = pd.DataFrame([train_loss_arr_epoch, test_loss_arr_epoch, acc_arr_epoch, train_execution_times,
                                 test_execution_times],
                            columns=['epoch_' + str(i) for i in range(len(train_loss_arr_epoch))],
                            index=['train_loss', 'test_loss' ,'test_acc', 'train_times','test_times'])
    
    loss_batch_df = pd.DataFrame([train_loss_arr_batch],
                                 columns=['batch_' + str(i) for i in range(len(train_loss_arr_batch))],
                                 index=['train_loss'])
    
    acc_batch_df = pd.DataFrame([acc_arr_batch, test_loss_arr_batch],
                                columns=['batch_' + str(i) for i in range(len(acc_arr_batch))],
                                index=['test_acc', 'test_loss'])
    
    epoch_df.to_csv(config.DATA_LOCATION+'/logs'+'/epoch_df.csv')
    loss_batch_df.to_csv(config.DATA_LOCATION+'/logs'+'/loss_batch_df.csv')
    acc_batch_df.to_csv(config.DATA_LOCATION+'/logs'+'/acc_batch_df.csv')


def flickr_train_test_split(dataset, train_size):
    """
    Performs a train-test split on the the flickr dataset. Assumes that test_size = 1-train_size.
    Makes sure to avoid biassing the model splitting the dataset in such a way that there is no image in the train and
    test data at the same time.
    
    Parameters:
    ------------
    dataset: FlickrDataset instance.
    	Dataset to be splitted.
        
    train_size: float.
    	Percentage of the data used to train the model.
        
    Returns:
    train_dataset: FlickrDataset instance.
    	Dataset containing the train samples.
        
    test_dataset: FlickrDataset instance.
    	Dataset containing the test samples.
    """
    # Finding idx of the data to train making sure no img is in train and test at the same time
    to_train = int(dataset.df.shape[0] * train_size)

    while to_train % 5 != 0:  # While some image still splits into train and test, take more images
        to_train += 1

    # Splitting dataset
    dataset.df = dataset.df.sort_values(by='image').reset_index(drop=True) # Grouping each img in 5 rows
    train_X, test_X = dataset.df.iloc[:to_train], dataset.df.iloc[to_train:]

    # Creating the datasets
    train_dataset, test_dataset = deepcopy(dataset), deepcopy(dataset)
    
    train_dataset.df = train_X
    train_dataset.df.reset_index(drop=True, inplace=True)
    train_dataset.imgs, train_dataset.captions = train_X['image'], train_X['caption']

    test_dataset.df = test_X
    test_dataset.df.reset_index(drop=True, inplace=True)
    test_dataset.imgs, test_dataset.captions = test_X['image'], test_X['caption']

    return train_dataset, test_dataset


def make_dataset(config):
    """
    Loads the FlickrDataset previously processed.
    
    Parameters:
    ------------
    	config: dictionary
        	Will have the data location from where the file will be loaded.
    
    Returns:
    -----------
    	dataset: FlickrDataset instance
        	Loaded dataset.
	"""
    dataset = joblib.load(config.DATA_LOCATION + "/processed_dataset.joblib")
    dataset.spacy_eng = spacy.load("en_core_web_sm")
    return dataset


def make_dataset_notebook(config):
    """
    Loads the FlickrDataset previously processed. Adapted for use in notebooks.

    Parameters:
    ------------
        config: dictionary
            Will have the data location from where the file will be loaded.

    Returns:
    -----------
        dataset: FlickrDataset instance
            Loaded dataset.
    """
    dataset = joblib.load(config['DATA_LOCATION'] + "/processed_dataset.joblib")
    dataset.spacy_eng = spacy.load("en_core_web_sm")
    return dataset


class ImgGroupingDataset:
    """
    Auxiliar class made to simplify the process of loading the processed data from the DataLoaders.
    The data schema is: [[img1,[cap1, cap2, cap3, cap4, cap5]], [...], ...]]
    
    Helps avoiding redundance when storaging the images in the RAM.
    """
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)*5 # self.data has exactly num_imgs*captions_per_image = num_imgs*5

    def __getitem__(self, index): # Recives the index out of 40.000
        img = self.data[index//5][0]
        captions = self.data[index//5][1][index%5]
        return img, captions   # No preprocessing here

    
def preprocess_dataset(dataset):
    """
    Preprocesses the dataset before giving it to the data loader to avoid innecessary executions.
    Saves the data in such a way that each image is only stored once.

    Parameters:
    ------------
		dataset: FlickrDataset instance
        	The dataset to be processed
    
    Returns:
    ----------
    	data_list: final processed data following the schema:  [[img1,[cap1, cap2, cap3, cap4, cap5]], [...], ...]]
    """
    
    data_list = []
    current_img_captions = []
    for idx, (img_name, caption) in dataset.df.reset_index(drop=True).iterrows():
        caption_vec = []
        caption_vec += [dataset.vocab.stoi["<SOS>"]]
        caption_vec += dataset.vocab.numericalize(caption, dataset.spacy_eng)
        caption_vec += [dataset.vocab.stoi["<EOS>"]]

        current_img_captions.append(torch.tensor(caption_vec))
        if not ((idx + 1) % 5):  # One time each 5 iterations the image will be processed
            img_location = os.path.join(dataset.root_dir, img_name)
            img = Image.open(img_location)
            img = img.convert("RGB")

            img = dataset.transform(img)

            data_list.append([img.to(torch.float16), current_img_captions])
            current_img_captions = []

    return data_list


def make_dataloaders(config, dataset, num_workers):
    """
    Main function to get the data loaders from the dataset. Splits de dataset, preprocesses it and creates the data loaders.
    
    Parameters:
    ------------
    	config: dictionary.
        	Contains the train_size split and the batch size information
        dataset: FlickrDataset instance.
        	Dataset from which the data loaders will be created.
        num_workers: Number of workers to be used in the data loaders (recomended 1).
        
    Returns:
    -----------
    	train_loader: DataLoader instance.
        	Contains the train data.
        test_loader: DataLoader instance.
        	Contains the test data.
    """
    train_dataset, test_dataset = flickr_train_test_split(dataset, config.train_size)

    processed_train = ImgGroupingDataset(preprocess_dataset(train_dataset))
    processed_test = ImgGroupingDataset(preprocess_dataset(test_dataset))

    train_loader = get_data_loader(processed_train, dataset, batch_size=config.batch_size,
                                   num_workers=num_workers, shuffle=True)
    test_loader = get_data_loader(processed_test, dataset, batch_size = 5,
                                  num_workers=num_workers, shuffle=False)

    return train_loader, test_loader


def make_dataloaders_notebook(config, dataset, num_workers):
    """
    Main function to get the data loaders from the dataset. Splits de dataset and creates the data loaders.
    Adapted for use in notebook.

    Parameters:
    ------------
        config: dictionary.
            Contains the train_size split and the batch size information
        dataset: FlickrDataset instance.
            Dataset from which the data loaders will be created.
        num_workers: Number of workers to be used in the data loaders (recomended 1).

    Returns:
    -----------
        train_loader: DataLoader instance.
            Contains the train data.
        test_loader: DataLoader instance.
            Contains the test data.
    """
    train_dataset, test_dataset = flickr_train_test_split(dataset, config['train_size'])

    train_loader = get_data_loader(train_dataset, dataset, batch_size=config['batch_size'], num_workers=num_workers)
    test_loader = get_data_loader(test_dataset, dataset, batch_size=5, num_workers=num_workers)

    return train_loader, test_loader


def get_data_loader(dataset, dataset_aux, batch_size, shuffle=False, num_workers=1):
    """
    Returns torch dataloader for the flicker8k dataset

    Parameters
    -----------
    dataset: FlickrDataset or List of lists
        custom torchdataset named FlickrDataset
    batch_size: int
        number of data to load in a particular batch
    shuffle: boolean,optional;
        should shuffle the dataset (default is False)
    num_workers: int,optional
        numbers of workers to run (default is 1)
    """

    pad_idx = dataset_aux.vocab.stoi["<PAD>"]
    collate_fn = CapsCollate(pad_idx=pad_idx, batch_first=True)

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn
    )

    return data_loader


class Vocabulary:
    """
    Class used to tokenize the captions in both ways. Storages the vocabulary that will be used to tokenize the captions.
    """
    def __init__(self, freq_threshold):
        # int to string tokens
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}

        # string to int tokens
        self.stoi = {v: k for k, v in self.itos.items()}

        self.freq_threshold = freq_threshold
        self.spacy_eng = spacy.load("en_core_web_sm")
        
    def __len__(self):
        return len(self.itos)

    def get_caption(self, numerized_caption):
        """
        Returns the captions from a tokenized list.
        
        Parameters:
        ------------
        	numerized_captions: list.
            	Contains the tokenized captions.
        
        Returns:
        ------------
        	caption: list.
            	contains the captions in str format.
        """
        caption = [self.itos[x] for x in numerized_caption if x not in [0, 1]] # Not using SOS and PAD
        return caption
    
    @staticmethod
    def tokenize(text, spacy_eng):
        """
        Returns the tokenized captions.
        
        Parameters:
        ------------
        	text: list.
            	Contains the captions in str format.
            spacy_eng: 
            	Contains the tokenizer from spacy.
        
        Returns:
        ------------
        	tokenized_caption: list.
            	contains the captions tokenized.
        """
        tokenized_caption =  [token.text.lower() for token in spacy_eng.tokenizer(text)]
        return tokenized_caption

    
    def build_vocab(self, sentence_list):
        """
        Builds the vocabulary for the dataset based on the frequencies of the words and saves it into the object.
        
        Parameters:
        ------------
        	sentence_list: list of lists of str.
            	Contains a list of the sentences. to be added to the vocab.
        """
        frequencies = Counter()
        idx = 4

        i = 1
        length = len(sentence_list)
        for sentence in sentence_list:
            i += 1
            if not (i % 100):
                print(i, "Iterations done out of", length)
            for word in self.tokenize(sentence, self.spacy_eng):
                frequencies[word] += 1

                # add the word to the vocab if it reaches minum frequecy threshold
                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

                    
    def numericalize(self, text, spacy_eng):
        """ For each word in the text corresponding index token for that word form the vocab built as list """
        tokenized_text = self.tokenize(text, spacy_eng)
        return [self.stoi[token] if token in self.stoi else self.stoi["<UNK>"] for token in tokenized_text]


class FlickrDataset(Dataset):
    """
    Dataset class that storages the information needed to process the data.
    """

    def __init__(self, root_dir, captions_file, transform=None, freq_threshold=5):
        self.root_dir = root_dir
        self.df = pd.read_csv(captions_file)
        self.transform = transform

        # Get image and caption column from the dataframe
        self.imgs = self.df["image"]
        self.captions = self.df["caption"]

        # Initialize vocabulary and build vocab
        self.vocab = Vocabulary(freq_threshold)
        self.vocab.build_vocab(self.captions.tolist())
        self.spacy_eng = spacy.load("en_core_web_sm")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        caption = self.captions[idx]
        img_name = self.imgs[idx]
        img_location = os.path.join(self.root_dir, img_name)
        img = Image.open(img_location)
        img = img.convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        # numericalize the caption text
        caption_vec = []
        caption_vec += [self.vocab.stoi["<SOS>"]]
        caption_vec += self.vocab.numericalize(caption, self.spacy_eng)
        caption_vec += [self.vocab.stoi["<EOS>"]]

        return img.clone(), torch.tensor(caption_vec)


class CapsCollate:
    """
    Collate to apply the padding to the captions with dataloader
    """

    def __init__(self, pad_idx, batch_first=False):
        self.pad_idx = pad_idx
        self.batch_first = batch_first

    def __call__(self, batch):
        imgs = [item[0].unsqueeze(0) for item in batch]
        imgs = torch.cat(imgs, dim=0)

        targets = [item[1] for item in batch]
        targets = pad_sequence(targets, batch_first=self.batch_first, padding_value=self.pad_idx)
        return imgs, targets




# helper function to save the model
def save_model(model, config, model_path):
    """
    Saves the trained model into the path indicated.
    """
    
    model_state = {
        'embed_size': config.embed_size,
        'vocab_size': config.vocab_size,
        'attention_dim': config.attention_dim,
        'encoder_dim': config.encoder_dim,
        'decoder_dim': config.decoder_dim,
        'state_dict': model.state_dict()
    }

    torch.save(model_state, model_path)


def get_caps_from(model, features_tensors, vocab, device='cuda'):
    """
    Generates the caption using the model and the encoded-decoded image.
    
    Parameters:
    -----------
    	model: EncoderDecoder instance.
        	Model that will be used to predict the captions.
            
        feature_tensors: Tensor
        	encoded-decoded image that will be fed to the model.
         
        vocab: Vocabulary instance.
        	Will be used to generate the caption.
            
    Return:
    ----------
    	caps: list.
        	Captions for the image.
        alphas: list.
        	Alphas of the attention given to each word from the caption.
    """
    model.eval()
    with torch.no_grad():
        features = model.encoder(features_tensors.to(device))
        caps, alphas = model.decoder.generate_caption(features, vocab=vocab)

    return caps, alphas


def generate_and_dump_dataset(root_dir, captions_file, transforms, data_location):
    """
    Creates the FlickrDataset and saves it into the indicated path.

    Parameters:
    ------------
    	root_dir: Str.
        	Images directory.
        captions_file: Str.
        	Captions path.
        transforms: List
        	Transformations to be applied to the images.
        data_location: Str.
        	Path where the processed dataset will be dumped.
    """
    dataset = FlickrDataset(
        root_dir=root_dir,
        captions_file=captions_file,
        transform=transforms
    )

    joblib.dump(dataset, data_location+"/processed_dataset.joblib")


def test_model_performance(model, test_loader, device, vocab, epoch, config):
    """
    Tests the model with only the first image of the test DataLoader and saves the outputs such as the image, predicted caption,
    real caption and the atention's alphas.
    
    Used for vizualization purposes.
    """
    with torch.no_grad():
        dataiter = iter(deepcopy(test_loader))
        img, real_captions = next(dataiter)
        img = img.to(torch.float32)
        features = model.encoder(img[0:1].to(device))
        caps, alphas = model.decoder.generate_caption(features, vocab=vocab)
        caption = ' '.join(caps)

        # Saving
        joblib.dump(img[0], config.DATA_LOCATION+'/logs'+'/img_epoch_' + str(epoch) + '.joblib')
        joblib.dump(caption, config.DATA_LOCATION+'/logs'+'/caption_epoch_' + str(epoch) + '.joblib')
        joblib.dump(caps, config.DATA_LOCATION + '/logs' + '/caps_epoch_' + str(epoch) + '.joblib')
        joblib.dump(alphas, config.DATA_LOCATION+ '/logs' + '/aphas_epoch_' + str(epoch) + '.joblib')
