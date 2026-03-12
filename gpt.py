import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Token and Positional Embedding

class TokenPositionEmbeddings(nn.Module):
    def __init__(self, vocab_size, block_size, n_embd, dropout):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, idx):
        B, T = idx.shape
        
        pos = torch.arrange(T, device=idx.device).unsqueeze(0)
        
        tok = self.token_emb(idx)
        pos = self.pos_emb(pos)
        
        x = tok + pos
        x = self.dropout(x)
        return x
    
#  Masked Self Attention

class MaskedSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        
        assert n_embd % n_head == 0
        
        self.n_head = n_head
        self.head_dim = n_embd // n_head

        self.q_proj = nn.Linear(n_embd, n_embd)
        self.k_proj = nn.Linear(n_embd, n_embd)
        self.v_proj = nn.Linear(n_embd, n_embd)

        self.out_proj = nn.Linear(n_embd, n_embd)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        mask = torch.tril(torch.ones(block_size, block_size))
        self.register_buffer("causal_mask", mask.view(1, 1, block_size, block_size))
        
    def forward(self, x):
        B, T, C = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Attention scores
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Masks
        att = att.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float("-inf"))

        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        y = self.out_proj(y)
        y = self.resid_dropout(y)

        return y
    
# Feed forward Neural Network
class FeedForwardNN(nn.Module):
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.ff1 = nn.Linear(n_embd, 4 * n_embd)
        self.act = nn.GELU()
        self.ff2 = nn.Linear(4 * n_embd, n_embd)
        self.dropout = dropout
        
    def forward(self, x):
        x = self.ff1(x)
        x = self.act(x)
        x = self.ff2(x)
        x = self.dropout(x)
        return x
    
# Transformer block
class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = MaskedSelfAttention(n_embd, n_head, block_size, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ff = FeedForwardNN(n_embd, dropout)
        
    def forward(self, x):
        x = x + self.attn(self.ln1)
        x = x + self.ff(self.ln2)
        return x

# GPT 
class RouteGPT(nn.Module):
    def __init__(self, vocab_size, block_size, pad_id, n_embd, n_head, n_layer, dropout):
        super().__init__()
        self.block_size = block_size
        self.pad_id = pad_id

        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.dropout = nn.Dropout(dropout)

        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)]
        )

        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size)

        self.apply(self._init_weights)


    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        if T > self.block_size:
            raise ValueError(f"Sequence length {T} exceeds block size {self.block_size}")

        pos = torch.arange(T, device=idx.device).unsqueeze(0)

        tok = self.token_emb(idx)
        pos = self.pos_emb(pos)

        x = self.dropout(tok + pos)
        x = self.blocks(x)
        x = self.ln_f(x)

        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=self.pad_id
            )

        return logits, loss