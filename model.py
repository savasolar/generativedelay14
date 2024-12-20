import torch
import torch.nn as nn
import torch.nn.functional as F

class MelodicStudent(nn.Module):
    def __init__(self, vocab_size, embedding_dim=32, hidden_size=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = nn.Embedding(32, embedding_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=1,
            dropout=0.1,
            batch_first=True
        )
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            bidirectional=True,
            batch_first=True
        )
        self.output = nn.Linear(hidden_size * 2, vocab_size)

    def forward(self, x):
        # Ensure positions don't exceed embedding size
        max_positions = min(x.shape[1], 32)  # 32 is our position embedding size
        positions = torch.arange(max_positions, device=x.device)
        
        token_emb = self.embedding(x)
        pos_emb = self.position_embedding(positions).unsqueeze(0)
        
        # Make sure pos_emb matches the sequence length
        if x.shape[1] > 32:
            pos_emb = pos_emb.repeat(1, x.shape[1] // 32 + 1, 1)[:, :x.shape[1], :]
            
        x = token_emb + pos_emb
        attention_mask = self._create_local_attention_mask(x.size(1), window_size=8)
        x, _ = self.attention(x, x, x, attn_mask=attention_mask)
        x, _ = self.lstm(x)
        logits = self.output(x)
        return logits

    def _create_local_attention_mask(self, size, window_size):
        mask = torch.ones(size, size, dtype=torch.bool, device=self.embedding.weight.device)
        for i in range(size):
            start = max(0, i - window_size)
            end = min(size, i + 1)
            mask[i, start:end] = False
        return mask