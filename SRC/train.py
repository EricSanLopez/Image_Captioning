from tqdm.auto import tqdm
import wandb
from utils.utils import *
from test import *


def train(model, data_loader, criterion, optimizer, config, epoch=-1, verbatim = True):
    """
    Main function to train the model through all the data stored in the DataLoader.
    
    Parameters:
    ------------
    	model: EncoderDecoder instance.
        	Model to be trained.
        
        data_loader: DataLoader.
        	Contains the data to be trained in batches.
            
        criterion: torch.nn.criterion.
        	Establishes the criterion to be used to get the model's loss.
            
        optimizer: torch.nn.optimizer.
        	Establishes the optimizer to be used to minimize the loss function on the model.
            
        config: Dictionary.
        	Contains hyperparameters for the model.
            
        epoch: int
        	Epoch in which the model is at. Used only for logging purposes.
            
        verbatim: Boolean.
        	Defaul: True. If True will print each 25 batches the loss.
            
    Returns:
    ----------
        loss_arr_batch: List of int.
        	Logs containing loss for each batch. Loss calculated using the criterion.
           
        time: Float.
        	Execution time of all the train process.
    
    """
    t0 = time.time()
    
    # Tell wandb to watch what the model gets up to: gradients, weights, and more!
    wandb.watch(model, criterion, log="all", log_freq=10)

    # Run training and track with wandb
    example_ct = 0  # number of examples seen

    loss_arr_batch = []  # Losses of the batches

    for idx, (image, captions) in enumerate(iter(data_loader)):
        # Training the model for one batch
        loss = train_batch(image.to(torch.float32), captions, model, config.vocab_size, optimizer, criterion, device=config.device)
        loss_arr_batch.append(loss.tolist()) 
        
        # Report metrics every 25th batch
        if ((idx + 1) % 25) == 0 and verbatim:
            example_ct += len(image)
            train_log(loss, example_ct, epoch, config.batch_size)

    return loss_arr_batch, time.time()-t0


def train_batch(image, captions, model, vocab_size, optimizer, criterion, device='cuda:0'):
    """
    Trains the model for on one single batch and returns the loss.
    """
    image, captions = image.to(device), captions.to(device)

    # Zero the gradients.
    optimizer.zero_grad()

    # Feed forward
    outputs, attentions = model(image, captions)

    # Calculate the batch loss.
    targets = captions[:, 1:]
    loss = criterion(outputs.view(-1, vocab_size), targets.reshape(-1))

    # Backward pass.
    loss.backward()

    # Update the parameters in the optimizer.
    optimizer.step()

    return loss


def train_log(loss, example_ct, epoch, batch_size):
    """
    Logs on wandb and console.
    """
    wandb.log({"epoch": epoch, "loss": loss}, step=example_ct)
    print(f"Loss after {str(example_ct*batch_size).zfill(5)} examples: {loss:.3f}")
