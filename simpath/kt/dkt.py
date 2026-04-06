"""DKT: Deep Knowledge Tracing (LSTM-based)."""

import torch
import torch.nn as nn


class DKT(nn.Module):
    def __init__(self, n_questions: int, n_kcs: int, hidden_dim: int = 256,
                 num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.n_questions = n_questions
        self.n_kcs = n_kcs
        # Input: one-hot of (question, correct) pairs → 2 * n_questions
        self.input_dim = 2 * n_questions
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(self.input_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, n_questions)

    def forward(self, q_ids, corrects, mask=None):
        """
        Args:
            q_ids: [B, T] question indices
            corrects: [B, T] correctness (0 or 1)
            mask: [B, T] padding mask
        Returns:
            logits: [B, T, n_questions] prediction logits for next step
        """
        # Encode input: question_id + correct * n_questions
        x = q_ids + corrects.long() * self.n_questions  # [B, T]
        x = self.embedding(x)  # [B, T, hidden]
        h, _ = self.lstm(x)  # [B, T, hidden]
        h = self.dropout(h)
        logits = self.fc(h)  # [B, T, n_questions]
        return logits

    def predict_next(self, q_ids, corrects):
        """Predict P(correct) for next question given history."""
        logits = self.forward(q_ids, corrects)
        return torch.sigmoid(logits)
