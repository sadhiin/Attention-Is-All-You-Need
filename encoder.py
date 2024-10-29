import torch
import torch.nn as nn
import torch.nn.functional as F

def scaled_dot_product_attention(Q, K, V, mask=None):
    # q, k, v = bs=32 x head=8, seq_len=200 x 64

    d_k = Q.shape[-1]   # 64
    # k_transpose = K.transpose(-1, -2)  # 32 x 8 x 64 x 200
    scaled = torch.matmul(Q, K.transpose(-1, -2)) / torch.sqrt(d_k)     # 32 x 8 x 200 x 200
    if mask is not None:
        # scaled = scaled.masked_fill(mask == 0, -1e9)
        # scaled.add_(mask)
        scaled += mask
    attention = F.softmax(scaled, dim=-1)    # 32 x 8 x 200 x 200
    values = torch.matmul(attention, V)  # 32 x 8 x 200 x 64
    return values, attention

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads) -> None:
        super(MultiHeadAttention, self).__init__()

        self.d_model = d_model      # embedding dimension = 512
        self.num_heads = num_heads  # number of heads = 8
        self.head_dim = d_model // num_heads    # 512 / 8 = 64
        self.qkv_layer = nn.Linear(d_model, d_model * 3)    # 512 x 512*3=1536
        self.fc = nn.Linear(d_model, d_model)   # 512 x 512


    def forward(self, x, mask=None):
        batch_size, sequence_length, d_model = x.size()     # bs=32 x seq_len=200 x d_model=512
        qkv = self.qkv_layer(x)    # 32 x 200 x 1536
        qkv = qkv.reshape(batch_size, sequence_length, self.num_heads, 3 * self.head_dim)   # 32 x 200 x 8 x 192
        qkv = qkv.permute(0, 2, 1, 3)       # 32 x 8 x 200 x 192
        Q, K, V = qkv.chunk(3, dim=-1)      # 32 x 8 x 200 x 64 each of (Q, K, V)

        # attention = # 32 x 8 x 200 x 64
        # values = # 32 x 8 x 200 x 64
        values, attention = scaled_dot_product_attention(Q, K, V, mask)
        values = values.permute(0, 2, 1, 3)..reshape(batch_size, sequence_length, self.d_model * self.num_heads)     # 32 x 200 x 512

        out = self.fc(values)   # 32 x 200 x 512
        return out


class LayerNormalization(nn.Module):
    def __init__(self, param_shape, eps = 1e-9) -> None:
        super(LayerNormalization, self).__init__()
        self.param_shape = param_shape      # [512]
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(param_shape))  # [512]
        self.beta = nn.Parameter(torch.zeros(param_shape))  # [512]

    def forward(self, x):       # 32 x 200 x 512
        dims = [-(i+1) for i in range(len(self.param_shape))]   # [-1]
        mean = x.mean(dims, keepdim=True)       # 30 x 200 x 1
        var = (x - mean).pow(2).mean(dims, keepdim=True)    # 32 x 200 x 1
        std = (var + self.eps).sqrt()    # 32 x 200 x 1
        y = (x - mean) / std   # 32 x 200 x 512

        return self.gamma * y + self.beta   # 32 x 200 x 512



class PositionwiseFeedForward(nn.Module):
    def __init(self, d_model, hidden_size, dropout=0.1) -> None:
        super(PositionwiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, hidden_size)      # 512 x 2048
        self.fc2 = nn.Linear(hidden_size, d_model)      # 2048 x 512
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):   # 32 x 200 x 512
        x = self.fc1(x)    # 32 x 200 x 2048
        x = self.relu(x)    # 32 x 200 x 2048
        x = self.dropout(x) # 32 x 200 x 2048

        x = self.fc2(x)     # 32 x 200 x 512
        return x


class EncoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, num_heads, dropout) -> None:
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model,num_heads)
        self.norm1 = LayerNormalization([d_model])
        self.dropout = nn.Dropout(p=dropout)
        self.ffn = PositionwiseFeedForward(d_model, ffn_hidden, dropout)
        self.norm2 = LayerNormalization([])


    def forward(self, x):
        residual_x = x
        x = self.attention(x, mask=None)        # 32 x 200 x 512
        x = self.dropout(x)             # 32 x 200 x 512
        x = self.norm1(x + residual_x)  # 32 x 200 x 512

        residual_x = x      # 32 x 200 x 512

        x = self.ffn(x)    # 32 x 200 x 512
        x = self.dropout(x)     # 32 x 200 x 512
        x = self.norm2(x + residual_x)  # 32 x 200 x 512

        return x


class Encoder(nn.Module):
    def __init__(self, d_model, ffn_hidden, num_heads, dropout, num_layers) -> None:
        super(Encoder, self).__init__()

        self.layers = nn.Sequential(
            *[EncoderLayer(d_model, ffn_hidden, num_heads, dropout) for _ in range(num_layers)]
        )

    def forward(self, x):
        return self.layers(x)

if __name__=="__main__":
    batch_size = 32
    d_model = 512
    num_heads = 8

    drop_prob = 0.1

    max_sequence_length = 200

    ffn_hidden = 2048
    num_layers = 5
    # num_layers = 12

    encoder = Encoder(d_model, ffn_hidden, num_heads, drop_prob, num_layers)

    x = torch.randn( (batch_size, max_sequence_length, d_model) )
    out = encoder(x)