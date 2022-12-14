import torch
import torch.nn as nn

from models.spoter_model import _get_clones, SPOTERTransformerDecoderLayer


class SPOTER_EMBEDDINGS(nn.Module):
    """
    Implementation of the SPOTER (Sign POse-based TransformER) architecture for sign language recognition from sequence
    of skeletal data.
    """

    def __init__(self, features, hidden_dim=108, nhead=9, num_encoder_layers=6, num_decoder_layers=6,
                 norm_emb=False, dropout=0.1):
        super().__init__()

        self.pos_encoding = nn.Parameter(torch.rand(1, 1, hidden_dim))  # init positional encoding
        self.class_query = nn.Parameter(torch.rand(1, 1, hidden_dim))
        self.transformer = nn.Transformer(hidden_dim, nhead, num_encoder_layers, num_decoder_layers, dropout=dropout)
        self.linear_embed = nn.Linear(hidden_dim, features)

        # Deactivate the initial attention decoder mechanism
        custom_decoder_layer = SPOTERTransformerDecoderLayer(self.transformer.d_model, self.transformer.nhead, 2048,
                                                             dropout, "relu")
        self.transformer.decoder.layers = _get_clones(custom_decoder_layer, self.transformer.decoder.num_layers)
        self.norm_emb = norm_emb

    def forward(self, inputs, src_masks=None):

        h = torch.transpose(inputs.flatten(start_dim=2), 1, 0).float()
        h = self.transformer(
            self.pos_encoding.repeat(1, h.shape[1], 1) + h,
            self.class_query.repeat(1, h.shape[1], 1),
            src_key_padding_mask=src_masks
        ).transpose(0, 1)
        embedding = self.linear_embed(h)

        if self.norm_emb:
            embedding = nn.functional.normalize(embedding, dim=2)

        return embedding
