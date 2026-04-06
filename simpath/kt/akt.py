"""AKT: Attentive Knowledge Tracing (attention-based)."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class AKT(nn.Module):
    def __init__(self, n_questions: int, n_kcs: int, d_model: int = 256,
                 num_heads: int = 8, num_blocks: int = 4, dropout: float = 0.1,
                 max_seq_len: int = 200):
        super().__init__()
        self.n_questions = n_questions
        self.n_kcs = n_kcs
        self.d_model = d_model

        self.q_embed = nn.Embedding(n_questions, d_model)
        self.kc_embed = nn.Embedding(n_kcs, d_model)
        self.r_embed = nn.Embedding(2, d_model)  # correct/incorrect
        self.pos_embed = nn.Embedding(max_seq_len, d_model)

        # Interaction embedding: question + response
        self.interaction_proj = nn.Linear(d_model * 2, d_model)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, dropout)
            for _ in range(num_blocks)
        ])

        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(d_model, 1)

    def forward(self, q_ids, kc_ids, corrects, mask=None):
        """
        Args:
            q_ids: [B, T] question indices
            kc_ids: [B, T] KC indices
            corrects: [B, T] correctness
            mask: [B, T] padding mask
        Returns:
            logits: [B, T] prediction logits
        """
        B, T = q_ids.shape
        device = q_ids.device

        # Embeddings
        q_emb = self.q_embed(q_ids)  # [B, T, D]
        pos = torch.arange(T, device=device).unsqueeze(0)  # [1, T]
        pos_emb = self.pos_embed(pos)  # [1, T, D]

        # Shift responses right: at position t, only see r[0..t-1]
        # This prevents data leakage (can't see current answer to predict it)
        r_shifted = torch.zeros_like(corrects)
        r_shifted[:, 1:] = corrects[:, :-1]
        r_emb = self.r_embed(r_shifted.long())  # [B, T, D]

        # Interaction embedding: (q[t-1], r[t-1]) at position t
        interaction = self.interaction_proj(
            torch.cat([q_emb, r_emb], dim=-1))  # [B, T, D]
        interaction = interaction + pos_emb

        # Query: current question embedding
        query = q_emb + pos_emb

        # Causal mask
        causal_mask = torch.triu(torch.ones(T, T, device=device), diagonal=1).bool()

        # Transformer blocks
        h = interaction
        for block in self.blocks:
            h = block(query, h, causal_mask, mask)

        h = self.layer_norm(h)
        logits = self.fc_out(self.dropout(h)).squeeze(-1)  # [B, T]
        return logits

    def predict_next(self, q_ids, kc_ids, corrects, mask=None):
        logits = self.forward(q_ids, kc_ids, corrects, mask)
        return torch.sigmoid(logits)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout,
                                          batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, kv, causal_mask=None, padding_mask=None):
        # Self-attention with causal mask
        normed_kv = self.norm1(kv)
        normed_q = self.norm1(query)

        key_padding = ~padding_mask.bool() if padding_mask is not None else None
        attn_out, _ = self.attn(normed_q, normed_kv, normed_kv,
                                attn_mask=causal_mask,
                                key_padding_mask=key_padding)
        h = query + self.dropout(attn_out)

        # Feed-forward
        h = h + self.ff(self.norm2(h))
        return h
