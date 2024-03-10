import torch
import torch.nn as nn
import numpy as np

class PositionalEncoder(nn.Module):

    """
    This class represents a positional encoder for a transformer model.
    It is used to give the model information about the position of words in a sentence.
    Inherits from the PyTorch nn.Module.

    Args:
        dm (int): The dimensionality of the model.
        maxlen (int): The maximum sequence length that the model can handle.
        dropout (float, optional): The dropout rate for the dropout layer. Default is 0.1.
        scale (bool, optional): Whether to scale the embeddings by dm. Default is True.

    Attributes:
        dm (int): The models dimensionality.
        drop (torch.nn.Dropout): The dropout layer.
        scale (bool): Indicator for scaling the embeddings.
        pos_encodings (torch.Tensor): The positional encodings, with shape (1, maxlen, dm).
    """


    def __init__(self, dm, maxlen, dropout=0.1, scale=True):
        super().__init__()
        self.dm = dm
        self.drop = nn.Dropout(dropout)
        self.scale = scale

        # shape: pos - (maxlen, 1) dim - (dm, )
        pos = torch.arange(maxlen).float().unsqueeze(1)
        dim = torch.arange(dm).float()

        # apply pos / (10000^2*i / dm) -> use sin for even indices & cosine for odd indices
        values = pos / torch.pow(1e4, 2 * torch.div(dim, 2, rounding_mode="floor") / dm)
        encodings = torch.where(dim.long() % 2 == 0, torch.sin(values), torch.cos(values))

        # reshape: encodings - (1, maxlen, dm)
        encodings = encodings.unsqueeze(0)
        
        # register encodings w/o grad
        self.register_buffer("pos_encodings", encodings)

    def forward(self, embeddings):

        """
        This method applies the positional encodings to the input embeddings.
        
        Args:
            embeddings (torch.Tensor): The input embeddings, with shape (batch_size, seq_len, dm).

        Returns:
            torch.Tensor: Positionally encoded embeddings, with shape (batch_size, seq_len, dm).
        """

        # inshape: embeddings - (batch_size, seq_len, dm)

        # enlarge embeddings (if applicable)
        if self.scale:
            embeddings = embeddings * np.sqrt(self.dm)
        # sum embeddings w/ respective positonal encodings | shape: embeddings - (batch_size, seq_len, dm)
        seq_len = embeddings.size(1)
        embeddings = embeddings + self.pos_encodings[:, :seq_len]
        # drop neurons | out - (batch_size, seq_len, dm)
        out = self.drop(embeddings)
        return out