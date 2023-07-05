import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


# Image Encoder
class EncoderCNN(nn.Module):
    """
    Encodes the image using the pre-trained CNN sent as argument.
    Possible encoders: ResNet50, ResNet152, googleNet, VGG.

    It extracts the relevant features from the image as a Tensor,
    and then reshapes it, so it matches the format of the following
    phases of the model.
    """
    def __init__(self, encoder='ResNet50'):
        super(EncoderCNN, self).__init__()

        # Chosing the encoder
        if encoder == 'ResNet50':
            my_encoder = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        elif encoder == 'ResNet152':
            my_encoder = models.resnet152(weights=models.ResNet152_Weights.DEFAULT)
        elif encoder == 'googleNet':
            my_encoder = models.googlenet(pretrained=True)
        elif encoder == 'VGG':
            my_encoder = models.vgg19(weights=models.VGG19_Weights.DEFAULT)

        # We don't want to re-train the encoder. Using the pre-trained one.
        for param in my_encoder.parameters():
            param.requires_grad_(False)

        modules = list(my_encoder.children())[:-2]
        self.my_encoder = nn.Sequential(*modules)

    def forward(self, images):
        features = self.my_encoder(images)  # (batch_size,2048,7,7)
        features = features.permute(0, 2, 3, 1)  # (batch_size,7,7,2048)
        features = features.view(features.size(0), -1, features.size(-1))  # (batch_size,49,2048)
        return features


class Attention(nn.Module):
    """
    Attention mechanism based on the Bahdanau attention (also known as additive
    attention). Used to weigh the relevance of input features when generating
    the next word of the caption.
    """
    
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super(Attention, self).__init__()

        self.attention_dim = attention_dim

        self.W = nn.Linear(decoder_dim, attention_dim)
        self.U = nn.Linear(encoder_dim, attention_dim)

        self.A = nn.Linear(attention_dim, 1)

    def forward(self, features, hidden_state):
        u_hs = self.U(features)  # (batch_size,num_layers,attention_dim)
        w_ah = self.W(hidden_state)  # (batch_size,attention_dim)

        combined_states = torch.tanh(u_hs + w_ah.unsqueeze(1))  # (batch_size,num_layers,attemtion_dim)

        attention_scores = self.A(combined_states)  # (batch_size,num_layers,1)
        attention_scores = attention_scores.squeeze(2)  # (batch_size,num_layers)

        alpha = F.softmax(attention_scores, dim=1)  # (batch_size,num_layers)

        attention_weights = features * alpha.unsqueeze(2)  # (batch_size,num_layers,features_dim)
        attention_weights = attention_weights.sum(dim=1)  # (batch_size,num_layers)

        return alpha, attention_weights


# Attention Decoder
class DecoderRNN(nn.Module):
    """
    Performs the embedding of the words of the caption as they are generated.
    To generate the words it uses an LSTM (RNN) combined with an attention mechanism.
    The words are generated in a for loop, where the features of the image coming from
    the Encoder are combined with the attention scores to produce a context vector, which
    is then used to determine what information is relevant to produce the next word.
    """
    def __init__(self, embed_size, vocab_size, attention_dim, encoder_dim, decoder_dim, drop_prob=0.3, device='cuda'):
        super().__init__()

        # save the model param
        self.vocab_size = vocab_size
        self.attention_dim = attention_dim
        self.decoder_dim = decoder_dim
        self.device = device

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)

        self.init_h = nn.Linear(encoder_dim, decoder_dim)
        self.init_c = nn.Linear(encoder_dim, decoder_dim)
        self.lstm_cell = nn.LSTMCell(embed_size + encoder_dim, decoder_dim, bias=True)
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)

        self.fcn = nn.Linear(decoder_dim, vocab_size)
        self.drop = nn.Dropout(drop_prob)

    def forward(self, features, captions, device='cuda'):

        # vectorize the caption
        embeds = self.embedding(captions)

        # Initialize LSTM state
        h, c = self.init_hidden_state(features)  # (batch_size, decoder_dim)

        # get the seq length to iterate
        seq_length = len(captions[0]) - 1  # Exclude the last one
        batch_size = captions.size(0)
        num_features = features.size(1)

        preds = torch.zeros(batch_size, seq_length, self.vocab_size).to(self.device)
        alphas = torch.zeros(batch_size, seq_length, num_features).to(self.device)

        for s in range(seq_length):
            alpha, context = self.attention(features, h)
            lstm_input = torch.cat((embeds[:, s], context), dim=1)
            h, c = self.lstm_cell(lstm_input, (h, c))

            output = self.fcn(self.drop(h))

            preds[:, s] = output
            alphas[:, s] = alpha

        return preds, alphas

    def generate_caption(self, features, max_len=20, vocab=None, device='cuda'):
        # Inference part
        # Given the image features generate the captions

        batch_size = features.size(0)
        h, c = self.init_hidden_state(features)  # (batch_size, decoder_dim)

        alphas = []

        # starting input
        word = torch.tensor(vocab.stoi['<SOS>']).view(1, -1).to(self.device)
        embeds = self.embedding(word)

        captions = []

        for i in range(max_len):
            alpha, context = self.attention(features, h)

            # store the alpha score
            alphas.append(alpha.cpu().detach().numpy())

            lstm_input = torch.cat((embeds[:, 0], context), dim=1)
            h, c = self.lstm_cell(lstm_input, (h, c))
            output = self.fcn(self.drop(h))
            output = output.view(batch_size, -1)

            # select the word with most val
            predicted_word_idx = output.argmax(dim=1)

            # save the generated word
            captions.append(predicted_word_idx.item())

            # end if <EOS detected>
            if vocab.itos[predicted_word_idx.item()] == "<EOS>":
                break

            # send generated word as the next caption
            embeds = self.embedding(predicted_word_idx.unsqueeze(0))

        # covert the vocab idx to words and return sentence
        return [vocab.itos[idx] for idx in captions], alphas

    def init_hidden_state(self, encoder_out):
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)  # (batch_size, decoder_dim)
        c = self.init_c(mean_encoder_out)
        return h, c
    

# Full model
class EncoderDecoder(nn.Module):
    """
    Wrapper that contains the full model encoder-decoder to predict captions
    from a given image. The images are sent to the encoder, which returns the features,
    then the features are sent to the decoder, which returns the captions.
    """
    def __init__(self, embed_size, vocab_size, attention_dim, encoder_dim, decoder_dim, drop_prob=0.3, device='cuda', encoder='ResNet50'):
        super().__init__()
        self.encoder = EncoderCNN(
            encoder=encoder
        )
        self.decoder = DecoderRNN(
            embed_size=embed_size,
            vocab_size=vocab_size,
            attention_dim=attention_dim,
            encoder_dim=encoder_dim,
            decoder_dim=decoder_dim,
            device=device
        )

    def forward(self, images, captions):
        features = self.encoder(images)
        outputs = self.decoder(features, captions)
        return outputs


def make_model(config, device='cuda'):
    """
    Generates the model. If another model wants to be used, add it here with the propper conditionals.
    
    Parameters:
    ------------
    config: Dictionary.
    	Must have the parameters explained in the "model_pipeline" function.
        
    Returns:
    -----------
    model: Generated model.
    """
    model = EncoderDecoder(config.embed_size, config.vocab_size, config.attention_dim, config.encoder_dim,
                           config.decoder_dim, device=device, encoder=config.encoder).to(device)

    return model

def load_ED_model(model_path, device, encoder='ResNet50'):
    """
    Loads a saved model given the model_path. 
    
    Call example: model = load_ED_model('attention_model_state.pth')
    
    Parameters:
    ------------
    model_path: str.
    	Path of the saved model.
        
    Returns:
    -----------
    model: Loaded model
    """
    checkpoint = torch.load(model_path, map_location=torch.device(device))

    model = EncoderDecoder(
        embed_size=checkpoint['embed_size'],
        vocab_size=checkpoint['vocab_size'],
        attention_dim=checkpoint['attention_dim'],
        encoder_dim=checkpoint['encoder_dim'],
        decoder_dim=checkpoint['decoder_dim'],
        encoder=encoder
    )
    model.load_state_dict(checkpoint['state_dict'])

    return model
