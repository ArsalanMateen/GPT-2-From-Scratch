import tiktoken
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

"""
Embeddings Pipeline
"""
class GPTDataset (Dataset):
    def __init__(self, token_ids, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # data sampling with a sliding window
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i+1:i + max_length+1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

def dataloader(token_ids, batch_size, 
               max_length, stride, 
               shuffle=True, drop_last=True,
               num_workers = 0):
    
    dataset = GPTDataset(token_ids, max_length, stride)

    # Create dataloader
    dataloader = DataLoader(dataset, 
                            batch_size=batch_size, 
                            shuffle=shuffle, drop_last=drop_last,
                            num_workers=num_workers)

    return dataloader

"""
Masked Multihead Attention
"""
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert (d_out % num_heads == 0), \
            "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out) # Linear layer to combine head outputs
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length),
                       diagonal=1))

    def forward(self, input):
        b, num_tokens, d_in = input.shape # batch size, number of tokens, input dimension

        keys = self.W_key(input)
        queries = self.W_query(input)
        values = self.W_value(input)

        # We implicitly split the matrix by adding a num_heads dimension
        # (batch_size, num_tokens, d_out) -> (batch_size, num_tokens, num_heads, head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim) 
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # Transpose: (batch_size, num_tokens, num_heads, head_dim) -> (batch_size, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        attn_scores = queries @ keys.transpose(2, 3)  # Dot product for each head

        # Original mask truncated to the number of tokens and converted to boolean
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)
        
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # (batch_size, num_tokens, num_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2) 
        
        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec) # optional projection

        return context_vec

"""
Transformer Block
"""
class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, input):
        mean = input.mean(dim=-1, keepdim=True)
        var = input.var(dim=-1, keepdim=True, unbiased=False)
        input_norm = (input - mean) / torch.sqrt(var + self.eps)
        return self.scale * input_norm + self.shift

class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return 0.5 * input * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0/torch.pi)) *
            (input + 0.044715 * torch.pow(input,3))
        ))

class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.Sequential(
        nn.Linear(config["emb_dim"], 4 * config["emb_dim"]),
        GELU(),
        nn.Linear(4 * config["emb_dim"], config["emb_dim"]))
        
    def forward(self, input):
        return self.layers(input)

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = MultiHeadAttention(
            d_in = config["emb_dim"],
            d_out = config["emb_dim"],
            context_length = config["context_length"],
            num_heads = config["n_heads"],
            dropout = config["drop_rate"],
            qkv_bias = config["qkv_bias"])
        self.feedforward = FeedForward(config)
        self.attnnorm = LayerNorm(config["emb_dim"])
        self.ffnorm = LayerNorm(config["emb_dim"])
        self.drop_shortcut = nn.Dropout(config["drop_rate"])
    
    def forward(self, input):

        shortcut = input
        input = self.attnnorm(input)
        input = self.attn(input)
        input = self.drop_shortcut(input)
        input = input + shortcut

        shortcut = input
        input = self.ffnorm(input)
        input = self.feedforward(input)
        input = self.drop_shortcut(input)
        input = input + shortcut
        return input

"""
GPT Architecture
"""
class GPTModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.tok_emb = nn.Embedding(config["vocab_size"], config["emb_dim"])
        self.pos_emb = nn.Embedding(config["context_length"], config["emb_dim"])
        self.drop_emb = nn.Dropout(config["drop_rate"])
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(config) for _ in range(config["n_layers"])])
        self.final_norm = LayerNorm(config["emb_dim"])
        self.out_head = nn.Linear(
            config["emb_dim"], config["vocab_size"], bias=False
        )

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(
          torch.arange(seq_len, device=in_idx.device)
        )
        input = tok_embeds + pos_embeds
        input = self.drop_emb(input)
        input = self.trf_blocks(input)
        input = self.final_norm(input)
        logits = self.out_head(input)
        return logits

"""
Generating text from model
(logits to text, greedy decoding)
"""
def generate_text(model, idx, max_new_tokens, context_size): 
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]   
        """ 
        In Pytorch, by default, it is always building computation graph 
        of the model in the background that is used for the 
        backpropagation algorithm, and if we don't train the model, 
        this very inefficent
        """
        # suppressing the generation of the gradient computation, the computation graph
        with torch.no_grad():  
            logits = model(idx_cond)
            
        logits = logits[:, -1, :]
        probas = torch.softmax(logits, dim=-1)          
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)   
        idx = torch.cat((idx, idx_next), dim=1)
        
    return idx

def text_to_token_ids(raw_text, tokenizer):
    encoded = tokenizer.encode(raw_text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)   
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)               
    return tokenizer.decode(flat.tolist())

"""
Model Evaluation
"""
def calc_batch_loss(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)        
    target_batch = target_batch.to(device)      
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), target_batch.flatten()
    )
    return loss

def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)    
    else:
        num_batches = min(num_batches, len(data_loader))  
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_batch_loss(
                input_batch, target_batch, model, device
            )
            total_loss += loss.item()   
        else:
            break
    return total_loss / num_batches

