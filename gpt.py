import math
import pandas as pd
import ast
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split



# # Token and Positional Embedding 
# class TokenPositionEmbeddings(nn.Module):
#     def __init__(self, vocab_size, block_size, n_embd, dropout):
#         super().__init__()
#         self.token_emb = nn.Embedding(vocab_size, n_embd)
#         self.pos_emb = nn.Embedding(block_size, n_embd)
#         self.dropout = nn.Dropout(dropout)
    
#     def forward(self, idx):
#         B, T = idx.shape
        
#         pos = torch.arange(T, device=idx.device).unsqueeze(0)
        
#         tok = self.token_emb(idx)
#         pos = self.pos_emb(pos)
        
#         x = tok + pos
#         x = self.dropout(x)
#         return x
    
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
        self.dropout = nn.Dropout(dropout)
        
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
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
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
    
    @torch.no_grad()
    def generate(self, angle, grade, stoi, itos, max_new_tokens=40, temperature=1.0, top_k=None):
        self.eval()

        start_tokens = [
            f"<ANG_{angle}>",
            f"<GRADE_{grade}>",
            "<ROUTE_START>"
        ]

        idx = torch.tensor(
            [[stoi[token] for token in start_tokens]],
            dtype=torch.long,
            device=self.head.weight.device
        )

        end_id = stoi["<ROUTE_END>"]

        banned_tokens = set()
        for token in stoi:
            if token.startswith("<ANG_"):
                banned_tokens.add(stoi[token])
            elif token.startswith("<GRADE_"):
                banned_tokens.add(stoi[token])

        banned_tokens.add(stoi["<ROUTE_START>"])
        banned_tokens.add(stoi["<PAD>"])

        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)

            logits = logits[:, -1, :] / temperature

            for bad_id in banned_tokens:
                logits[0, bad_id] = float("-inf")

            if top_k is not None:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < values[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            idx = torch.cat((idx, next_token), dim=1)

            if next_token.item() == end_id:
                break

        token_ids = idx[0].tolist()
        tokens = [itos[i] for i in token_ids]

        return tokens
    
# Data Preprocessing

def hold_to_token(label, coord):
    x, y = coord
    return f"<{label.upper()}_X{x}_Y{y}>"

def row_to_token(row):
    angle = int(row["angle"])
    grade = int(row["v_grade"])
    
    start_holds = ast.literal_eval(row["start"])
    middle_holds = ast.literal_eval(row["middle"])
    finish_holds = ast.literal_eval(row["finish"])
    
    tokens = [f"<ANG_{angle}>", f"<GRADE_{grade}>", "<ROUTE_START>"]
    
    for label, coord in start_holds:
        tokens.append(hold_to_token(label, coord))

    for label, coord in middle_holds:
        tokens.append(hold_to_token(label, coord))

    for label, coord in finish_holds:
        tokens.append(hold_to_token(label, coord))

    tokens.append("<ROUTE_END>")
    return tokens

def get_tokens():
    df = pd.read_csv("DATA.csv")
    routes = []

    for _, row in df.iterrows():
        tokens = row_to_token(row)
        routes.append(tokens)
    return routes

def build_vocab(routes):
    vocab = set()
    
    for route in routes:
        for token in route:
            vocab.add(token)
    
    vocab.add("<PAD>")
    
    # Lookup dictionary
    stoi = {token: i for i, token in enumerate(vocab)} # String to Int
    itos = {i: token for token, i in stoi.items()} # Int to String

    return stoi, itos

def encode_routes(routes, stoi):
    encoded_routes = []

    for route in routes:
        encoded = [stoi[token] for token in route]
        encoded_routes.append(encoded)

    return encoded_routes

class RouteDataset(Dataset):
    def __init__(self, encoded_routes):
        self.encoded_routes = encoded_routes

    def __len__(self):
        return len(self.encoded_routes)

    def __getitem__(self, idx):
        route = self.encoded_routes[idx]
        x = torch.tensor(route[:-1], dtype=torch.long)
        y = torch.tensor(route[1:], dtype=torch.long)
        return x, y
    
# Padding function
def collate_fn(batch, pad_id):
    xs, ys = zip(*batch)

    max_len = max(len(x) for x in xs)

    x_padded = torch.full((len(xs), max_len), pad_id, dtype=torch.long)
    y_padded = torch.full((len(xs), max_len), pad_id, dtype=torch.long)

    for i, (x, y) in enumerate(zip(xs, ys)):
        x_padded[i, :len(x)] = x
        y_padded[i, :len(y)] = y

    return x_padded, y_padded

def data_splits(encoded_routes, stoi):
    dataset = RouteDataset(encoded_routes)
    
    n = len(dataset)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)
    n_test = n - n_train - n_val

    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(67)
    )
    
    return train_dataset, val_dataset, test_dataset



def train():
    df = pd.read_csv("DATA.csv")
    routes = get_tokens()
    stoi, itos = build_vocab(routes)
    encoded_routes = encode_routes(routes, stoi)
    
    train_dataset, val_dataset, test_dataset = data_splits(encoded_routes, stoi)
    pad_id = stoi["<PAD>"]

    train_loader = DataLoader(train_dataset, batch_size= 512, shuffle=True, collate_fn=lambda batch: collate_fn(batch, pad_id)) 
    val_loader = DataLoader(val_dataset, batch_size= 512, shuffle=False, collate_fn=lambda batch: collate_fn(batch, pad_id))
    test_loader = DataLoader(test_dataset, batch_size= 512, shuffle=False, collate_fn=lambda batch: collate_fn(batch, pad_id))
    
    device = "cuda"
    
    model = RouteGPT(vocab_size=len(stoi), block_size=128, pad_id=pad_id, n_embd=256, n_head=16, n_layer=8, dropout=0.1).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    best_val_loss = float("inf")
    model_path = "best_route_gpt.pt"

    for epoch in range(50):
        model.train()
        train_loss = 0.0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logits, loss = model(x, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)

                logits, loss = model(x, y)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        print(f"epoch {epoch+1}: train_loss={train_loss:.4f} val_loss={val_loss:.4f}") 
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "stoi": stoi,
                "itos": itos,
                "pad_id": pad_id,
                "model_config": {
                    "vocab_size": len(stoi),
                    "block_size": 128,
                    "n_embd": 256,
                    "n_head": 16,
                    "n_layer": 8,
                    "dropout": 0.1
                }
            }, model_path)
            print(f"saved best model at epoch {epoch+1} with val_loss={val_loss:.4f}")
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    test_loss = 0.0
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)

            logits, loss = model(x, y)
            test_loss += loss.item()

    test_loss /= len(test_loader)
    print(f"best model test_loss={test_loss:.4f}")

if __name__ == "__main__":
    # train()
    
    device = "cuda"
    checkpoint = torch.load("best_route_gpt.pt", map_location=device)
    stoi = checkpoint["stoi"]
    itos = checkpoint["itos"]
    pad_id = checkpoint["pad_id"]
    config = checkpoint["model_config"]
    
    model = RouteGPT(
        vocab_size=config["vocab_size"],
        block_size=config["block_size"],
        pad_id=pad_id,
        n_embd=config["n_embd"],
        n_head=config["n_head"],
        n_layer=config["n_layer"],
        dropout=config["dropout"]
    ).to(device)
    
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    generated = model.generate(
        angle=40,
        grade=3,
        stoi=stoi,
        itos=itos,
        max_new_tokens=30,
        temperature=0.8,
        top_k=10
    )

    print(generated)