"""SAINT: Separated Attentive INternet-based Tracing (encoder-decoder transformer)."""

import math
import torch
import torch.nn as nn


class SAINT(nn.Module):
    def __init__(self, n_questions: int, n_kcs: int, d_model: int = 256,
                 num_heads: int = 8, enc_layers: int = 4, dec_layers: int = 4,
                 dropout: float = 0.1, max_seq_len: int = 200):
        super().__init__()
        self.n_questions = n_questions
        self.d_model = d_model

        # Encoder: question (exercise) embeddings
        self.q_embed = nn.Embedding(n_questions, d_model)
        self.kc_embed = nn.Embedding(n_kcs, d_model)
        self.enc_pos = nn.Embedding(max_seq_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=enc_layers)

        # Decoder: response embeddings
        self.r_embed = nn.Embedding(2, d_model)
        self.dec_pos = nn.Embedding(max_seq_len, d_model)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=num_heads, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, activation="gelu")
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=dec_layers)

        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(d_model, 1)

    def forward(self, q_ids, kc_ids, corrects, mask=None):
        """
        Args:
            q_ids: [B, T]
            kc_ids: [B, T]
            corrects: [B, T]
            mask: [B, T]
        Returns:
            logits: [B, T]
        """
        B, T = q_ids.shape
        device = q_ids.device
        pos = torch.arange(T, device=device).unsqueeze(0)

        # Encoder input: question embeddings
        enc_input = self.q_embed(q_ids) + self.enc_pos(pos)
        key_padding = ~mask.bool() if mask is not None else None
        enc_out = self.encoder(enc_input, src_key_padding_mask=key_padding)

        # Decoder input: response embeddings (shifted right)
        # Shift: at position t, decoder sees responses 0..t-1
        r_shifted = torch.zeros_like(corrects)
        r_shifted[:, 1:] = corrects[:, :-1]
        dec_input = self.r_embed(r_shifted.long()) + self.dec_pos(pos)

        # Causal mask for decoder
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            T, device=device)

        dec_out = self.decoder(
            dec_input, enc_out,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=key_padding,
            memory_key_padding_mask=key_padding,
        )

        dec_out = self.layer_norm(dec_out)
        logits = self.fc_out(self.dropout(dec_out)).squeeze(-1)  # [B, T]
        return logits

    def predict_next(self, q_ids, kc_ids, corrects, mask=None):
        logits = self.forward(q_ids, kc_ids, corrects, mask)
        return torch.sigmoid(logits)
