import wandb
import torch
from utils.utils import *
from nltk.translate.bleu_score import sentence_bleu
import numpy as np


def test(model, test_loader, criterion, vocab, config, device="cuda", verbatim=True):
    """
    Run the model on some test examples and stores some metrics. 
    Assumes that batch_size = 5 to calculate the accuracy propperly.
    
    Parameters:
    ------------
    	model: EncoderDecoder instance.
        	Model to be evaluated.
        
        test_loader: DataLoader.
        	Contains the data to be tested in batches.
            
        criterion: torch.nn.criterion.
        	Establishes the criterion to be used to test the model's loss.
            
        vocab: Vocabulary instance.
        	Has the vocabulary needed to create the captions.
            
        config: Dictionary.
        	Contains hyperparameters for the model.
            
        verbatim: Boolean.
        	Defaul: True. If True will print each 25 batches the loss and accuracy.
            
    Returns:
    ----------
    	acc_arr_batch: List of int.
        	Logs containing accuracy for each batch.
            
        loss_arr_batch: List of int.
        	Logs containing loss for each batch. Loss calculated using the criterion.
           
        time: Float.
        	Execution time of all the test process.
    
    """
    t0 = time.time()
    acc_arr_batch, loss_arr_batch = [], []
    total, total_time = 0, 0

    with torch.no_grad():
        for images, captions in test_loader:
            # Sending data to GPU
            images, captions = images.to(device).to(torch.float32), captions.to(device)

            # Calculating loss
            outputs, attentions = model(images, captions)
            targets = captions[:, 1:]
            loss = criterion(outputs.view(-1, config.vocab_size), targets.reshape(-1))

            # Calculating accuracy
            images = images[0].detach().clone()
            predicted, _ = get_caps_from(model, images.unsqueeze(0), vocab=vocab, device=device)
            caps = [vocab.get_caption(cap.tolist()) for cap in captions]
            acc_score = sentence_bleu(caps, predicted)

            # Appending metrics
            acc_arr_batch.append(acc_score) ; loss_arr_batch.append(loss.tolist())
            total += 1

            # Report metrics every 25th batch
            if ((total + 1) % 25) == 0 and verbatim:
                print("Batch:", total, "\nAcc_score = ", acc_score); print("Loss:", loss.tolist())

        print(f"Mean BLEU score of the model on the {total} " +
              f"test images: {np.mean(acc_arr_batch)}%")
        
        wandb.log({"test_mean_bleu": sum(acc_arr_batch)/len(acc_arr_batch)})
        
    return acc_arr_batch, loss_arr_batch, time.time()-t0
