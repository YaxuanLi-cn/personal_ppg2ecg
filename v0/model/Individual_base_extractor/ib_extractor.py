import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def rotate_half(x):
    """Splits the last dimension in half and swaps the halves with negation."""
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)

class RoPEEmbedder(nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        self.dim = dim
        self.base = base
        self.inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    
    def forward(self, x):
        """Applies RoPE to the input tensor x."""
        seq_len = x.shape[1]
        device = x.device
        pos = torch.arange(seq_len, device=device, dtype=torch.float).unsqueeze(1)
        freqs = torch.einsum("nd,d->nd", pos, self.inv_freq.to(device))
        emb = torch.cat([freqs, freqs], dim=-1)
        
        cos_emb = emb.cos().unsqueeze(0)  # Shape (1, seq_len, dim)
        sin_emb = emb.sin().unsqueeze(0)
        
        return x * cos_emb + rotate_half(x) * sin_emb

# Feed Forward Network
class FeedForward(nn.Module):
    def __init__(self, embed_dim, ff_hidden_size):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(embed_dim, ff_hidden_size)
        self.fc2 = nn.Linear(ff_hidden_size, embed_dim)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


class IBEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_hidden_size, dropout):
        super(IBEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.feed_forward = FeedForward(embed_dim, ff_hidden_size)
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Self-attention sublayer with residual connection
        x1 = self.layer_norm1(x)
        attn_output = self.self_attn(x1, x1, x1, need_weights=False)[0]
        x = x + self.dropout(attn_output)
        
        # Feed-forward sublayer with residual connection
        x2 = self.layer_norm2(x)
        ff_output = self.feed_forward(x2)
        x = x + self.dropout(ff_output)
        
        return x

class IBExtractor(nn.Module):
    def __init__(self, embed_dim=256, in_channel=32, num_heads=8, ff_hidden_size=1024, num_layers=3, dropout=0, text_embed_dim=768, patient_info_size=3):
        super(IBExtractor, self).__init__()
        self.embed_dim = embed_dim
        self.signal_embedding = nn.Linear(in_channel, embed_dim) 
        self.RoPE = RoPEEmbedder(dim=embed_dim)

        self.down_sample = nn.Sequential(
            nn.Conv1d(2, 64, kernel_size=15, stride=5, padding=7),  # (B, 2, 1250) -> (B, 64, 250)
            nn.ReLU(),
            nn.Conv1d(64, 32, kernel_size=15, stride=5, padding=7), # (B, 64, 250) -> (B, 32, 50)
            nn.ReLU(),
        )

        self.layers = nn.ModuleList([
            IBEncoderLayer(embed_dim, num_heads, ff_hidden_size, dropout)
            for _ in range(num_layers)
        ])
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, ppg, ecg, reduce=True):
        # Add positional encoding to input embeddings
        # (B, L, C) -> (B, L, embed_dim)
        x = torch.cat([ppg, ecg], dim=-1)
        x = self.down_sample(x.transpose(1, 2)).transpose(1, 2)
        x = self.signal_embedding(x)
        x = self.RoPE(x)

        # Pass input through each transformer encoder layer
        for layer in self.layers:
            x = layer(x)
        
        if reduce:
            # (B, L, dim) -> (B, dim)
            return x.mean(dim=1)
        else:
            # (B, L, dim)
            return x


if __name__ == "__main__":
    device = 'cpu'

    model = IBExtractor(num_layers=3).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters in the model: {total_params:,}")

    # 随机输入 (N, 4, 50)
    N = 8
    ppg1 = torch.randn(N, 1250, 1).to(device)
    ecg1 = torch.randn(N, 1250, 1).to(device)
    ppg2 = torch.randn(N, 1250, 1).to(device)
    ecg2 = torch.randn(N, 1250, 1).to(device)

    rep_1 = model(ppg1, ecg1)
    print(rep_1.shape)