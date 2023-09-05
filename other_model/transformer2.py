import torch
import torch.nn as nn
import torch.nn.functional as F


NUM_TOLENS = 4
NUMBER_CLASSES = 2 
EMBEDDING_DIM = 96
TRANSFORMER_DEPTH = 24
HEADS = 24
DROPOUT = 0.5



class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, heads, dropout):
        super(TransformerBlock, self).__init__()

        self.multi_head_attention = nn.MultiheadAttention(embed_dim, heads)
        self.dropout1 = nn.Dropout(dropout)
        self.layer_norm1 = nn.LayerNorm(embed_dim)

        self.feed_forward = nn.Sequential(nn.Linear(embed_dim, 4 * embed_dim),
                                           nn.ReLU(),
                                           nn.Dropout(dropout),
                                           nn.Linear(4 * embed_dim, embed_dim))
        self.dropout2 = nn.Dropout(dropout)
        self.layer_norm2 = nn.LayerNorm(embed_dim)

    def forward(self, src):
        out = self.multi_head_attention(src, src, src)[0]
        out = self.dropout1(out)
        out = self.layer_norm1(out + src)

        src2 = out
        out = self.feed_forward(out)
        out = self.dropout2(out)
        out = self.layer_norm2(out + src2)

        return out


class TransformerClassifier(nn.Module):
    def __init__(self, num_tokens = NUM_TOLENS, 
                num_classes = NUMBER_CLASSES, 
                embedding_dim = EMBEDDING_DIM, 
                transformer_depth = TRANSFORMER_DEPTH,
                heads = HEADS, 
                dropout=DROPOUT):
        super(TransformerClassifier, self).__init__()

        self.num_tokens = num_tokens
        self.token_embedding = nn.Embedding(num_tokens, embedding_dim)
        self.transformer_blocks = nn.Sequential(
            *[TransformerBlock(embedding_dim, heads, dropout) for _ in range(transformer_depth)]
        )
        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        #batch_size, num_channels, seq_len = x.shape
        #x = x.view(batch_size * num_channels, seq_len)
        x = self.token_embedding(x)
        #x = x.float()
        x = x.transpose(0, 1)
        x = self.transformer_blocks(x)
        x = x.transpose(0, 1)
        x = self.pooling(x.transpose(1, 2))
        x = x.squeeze(-1)
        x = self.fc(x)
 
        #return torch.sigmoid(x)
        return x            #(batch_size*num_channels, num_classes)
        
        #训练时需要将每个序列的标签reshape为大小为(batch_size * num_channels, num_classes)的列向量作为损失函数的目标
        #labels = labels.repeat(num_channels, 1)
        #labels = labels.view(-1, num_classes)