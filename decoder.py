import torch
import torch.nn as nn

def scaled_dot_product(Q, K, V, mask=None):
    # Q: 32 x 8 x 200 x 64
    # K: 32 x 8 x 200 x 64
    # V: 32 x 8 x 200 x 64
    # mask: 200 x 200

    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / d_k ** 0.5
    if mask is not None:
        print('Addding mask of shape : ', mask.size())
        scores = scores.masked_fill(mask == 0, -1e9)

    attention = torch.softmax(scores, dim=-1)   # 32 x 8 x 200 x 200
    # values are actual context aware final tensors. [for every batch, for every head, for every word/token, 64 dimension of context for that word]
    values = torch.matmul(attention, V)   # 32 x 8 x 200 x 64
    return values, attention

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads) -> None:
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.qkv_layer = nn.Linear(d_model, 3* d_model)    # 512 x 1536
        self.fc = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        batch_size, sequence_length, d_model = x.size()     # 32 x 200 x 512
        print(f'x.size() : {x.size()}')
        qkv = self.qkv_layer(x)     # 32 x 200 x 1536
        print(f'qkv.size() : {qkv.size()}')

        qkv = qkv.reshape(batch_size, sequence_length, self.num_heads, 3 * self.head_dim)   # 32 x 200 x 8 x 192
        print('qkv after reshape : ', qkv.size())

        qkv = qkv.permute(0, 2, 1, 3)   # 32 x 8 x 200 x 192
        print('qkv after permute : ', qkv.size())

        Q, K, V = qkv.chunk(3, dim=-1)  # 32 x 8 x 200 x 64 : each of (Q, K, V)
        print(f"Q: {Q.size()}, K: {K.size()}, V: {V.size()}")

        values, attention = scaled_dot_product(Q, K, V, mask)   # 32 x 8 x 200 x 64
        print(f"values: {values.size()}, attention: {attention.size()}")

        values = values.reshape(batch_size, sequence_length, self.head_dim * self.num_heads)   # 32 x 200 x 512
        print('values after reshape : ', values.size())

        out = self.fc(values)   # 32 x 200 x 512
        print('out after passing through fc layers : ', out.size())

        return out  # 32 x 200 x 512

class LayerNormalization(nn.Module):
    def __init__(self, param_shape, eps = 1e-9) -> None:
        super(LayerNormalization, self).__init__()
        self.param_shape = param_shape      # [512]
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(param_shape))  # [512]
        self.beta = nn.Parameter(torch.zeros(param_shape))  # [512]

    def forward(self, x):       # 32 x 200 x 512
        dims = [-(i+1) for i in range(len(self.param_shape))]   # [-1]
        print(f"dims: {dims}")
        mean = x.mean(dims, keepdim=True)       # 30 x 200 x 1
        print(f"mean: {mean.size()}")
        var = (x - mean).pow(2).mean(dims, keepdim=True)    # 32 x 200 x 1
        std = (var + self.eps).sqrt()    # 32 x 200 x 1
        y = (x - mean) / std   # 32 x 200 x 512

        return self.gamma * y + self.beta   # 32 x 200 x 512

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, hidden, drop_prob=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, hidden)
        self.linear2 = nn.Linear(hidden, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):
        #  x: 30 x 200 x 512
        x = self.linear1(x) #30 x 200 x 2048
        print(f"x after first linear layer: {x.size()}")
        x = self.relu(x) #30 x 200 x 2048
        print(f"x after relu layer: {x.size()}")
        x = self.dropout(x) #30 x 200 x 2048
        print(f"x after dropout layer: {x.size()}")
        x = self.linear2(x) #30 x 200 x 512
        print(f"x after 2nd linear layer: {x.size()}")
        return x #30 x 200 x 512


class MultiHeadCrossAttention(nn.Module):

    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.kv_layer = nn.Linear(d_model , 2 * d_model) # 1024
        self.q_layer = nn.Linear(d_model , d_model)
        self.linear_layer = nn.Linear(d_model, d_model)

    def forward(self, x, y, mask=None):
        batch_size, sequence_length, d_model = x.size() # 30 x 200 x 512
        print(f"x.size(): {x.size()}")
        kv = self.kv_layer(x) # 30 x 200 x 1024
        print(f"kv.size(): {kv.size()}")
        q = self.q_layer(y) # 30 x 200 x 512
        print(f"q.size(): {q.size()}")
        kv = kv.reshape(batch_size, sequence_length, self.num_heads, 2 * self.head_dim)  # 30 x 200 x 8 x 128
        q = q.reshape(batch_size, sequence_length, self.num_heads, self.head_dim)  # 30 x 200 x 8 x 64
        kv = kv.permute(0, 2, 1, 3) # 30 x 8 x 200 x 128
        q = q.permute(0, 2, 1, 3) # 30 x 8 x 200 x 64
        k, v = kv.chunk(2, dim=-1) # K: 30 x 8 x 200 x 64, v: 30 x 8 x 200 x 64
        values, attention = scaled_dot_product(q, k, v, mask) #  30 x 8 x 200 x 64
        print(f"values: {values.size()}, attention:{attention.size()}")
        values = values.reshape(batch_size, sequence_length, d_model) #  30 x 200 x 512
        out = self.linear_layer(values)  #  30 x 200 x 512
        print(f"out after passing through linear layer: {out.size()}")
        return out  #  30 x 200 x 512

class DecoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, num_heads, dropout) -> None:
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = LayerNormalization([d_model])
        self.dropout1 = nn.Dropout(p=dropout)
        self.encoder_decoder_attention = MultiHeadCrossAttention(d_model, num_heads)
        self.norm2 = LayerNormalization([d_model])
        self.dropout2 = nn.Dropout(p=dropout)
        self.ffn = PositionwiseFeedForward(d_model, ffn_hidden, dropout)
        self.norm3 = LayerNormalization([d_model])
        self.dropout3 = nn.Dropout(p=dropout)

    def forward(self, x, y, decoder_mask):
        residual_y = y
        print('--'*20,' Maksed self attention ','--'*20)
        y = self.self_attention(y, mask=decoder_mask)   # 32 x 200 x 512

        print('--'*20,' Dropout-1 ','--'*20)
        y = self.dropout1(y)
        print('--'*20,' Add + Layer-Normalization-1 ','--'*20)
        y = self.norm1(y + residual_y)

        residual_y = y

        print('--'*20,' Maksed encoder-decoder / Cross Attention ','--'*20)
        y = self.encoder_decoder_attention(x, y, mask=None)   # 32 x 200 x 512

        print('--'*20,' Dropout-2 ','--'*20)
        y = self.dropout2(y)
        print("--"*20, "Add + Layer-Normalization-2", "--"*20)
        y = self.norm2(y + residual_y)


        residual_y = y
        print('--'*20,' Feed Forward Network ','--'*20)
        y = self.ffn(y)   # 32 x 200 x 512

        print('--'*20,' Dropout-3 ','--'*20)
        y = self.dropout3(y)
        print('--'*20,' Add + Layer-Normalization-3 ' , '--'*20)
        y = self.norm3(y + residual_y)
        return y


class SequentialDecoder(nn.Sequential):
    def forward(self,*inputs):
        x, y, mask = inputs
        for module in self._modules.values():
            y = module(x, y, mask)  # 32 x 200 x 512
        return y

class Decoder(nn.Module):
    def __init__(self, d_model, ffn_hidden, num_heads, dropout, num_layers=1):
        super(Decoder, self).__init__()
        self.layers = SequentialDecoder(
            *[DecoderLayer(d_model, ffn_hidden, num_heads, dropout) for _ in range(num_layers)]
        )

    def forward(self, x, y, mask):
        # x: 32 x 200 x 512
        # y: 32 x 200 x 512
        # mask : 200 x 200
        y = self.layers(x, y, mask) # 32 x 200 x 512
        return y




if __name__=="__main__":
    d_model = 512
    num_heads = 8
    dropout = 0.1
    batch_size = 32
    max_seq_len = 200
    ffn_hidden = 2048
    num_layers = 5

    x = torch.randn( (batch_size, max_seq_len, d_model))
    y = torch.randn( (batch_size, max_seq_len, d_model))

    mask = torch.full( [max_seq_len, max_seq_len], -float('Inf'))
    mask = torch.triu(mask, diagonal=1)

    decoder = Decoder(d_model, ffn_hidden, num_heads, dropout, num_layers)
    out = decoder(x, y, mask)
    print(out.shape) # torch.Size([32, 200, 512])