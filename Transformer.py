# Defining Scale Dot Product it is used to calculate the SELF ATTENTION 
# then we use it in Multi-Head Attention
class Scale_Dot_Product(nn.Module):
    def __init__(self):
        super().__init__()

    # Q = query, K = key, V = value " it's important 3 values for calculating SELF ATTENTION " and,
    # Mask is used in Decoder Layer then we pass.
    def forward(self, Q, K, V, Mask=None):
        dimension_of_K = K.size(-1)  # it's give the Dimension of K

        # Calculating Important part of attention "SCORE" 
        # " K.transpose(-2, -1)) " means Swapping value second last to last then,
        # Performed " Dot Product " and Divide by square root of " Dimension of K " to calculate the ATTENTION SCORE
        # And Attention Score have " Shape of (batch_size, num_heads, seq_length, seq_length)"
        attention_score = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(dimension_of_K, dtype=torch.float32))

        # Masking is same size of Attention Score
        # If any value in attention_score is equal to zero it's replaced to -1e9 and it is very small value
        # it is preventing the model from attending to these positions because the corresponding weights in the attention will be very small
        # basically for attention_score ignoring the certain positions.
        if Mask is not None:
            attention_score = attention_score.masked_fill(Mask == 0, -1e9)

        # After Getting attention_score we will apply SOFTMAX activation function on attention_score and obtained the attention_score WEIGHTS
        attention_score = torch.softmax(attention_score, dim=-1)

        # Then we multiply with V ( VALUE ) to get final Output.
        output = torch.matmul(attention_score, V)

        return output


# Here we calculate the MULTI-HEAD ATTENTION
# it takes two parameters dimension_of_model and num_heads
# dimension_of_model refers to the dimensionality of the input and output vectors for each layer and sub-layer, 
# including the embeddings, the attention mechanism,
# and the feed-forward network defines the size of the feature vectors used throughout the model.
# When input tokens (words or subwords) are fed into the Transformer, they are first converted into vectors of size d_model using an embedding layer.
# This means that each token is represented as a d_model-dimensional vector.
# Common values of d_model used in practice include 128, 256, 512, and 1024. The original Transformer model by Vaswani et al. used d_model = 512.
class Multi_Head_Attention(nn.Module):
    def __init__(self, dimension_of_model, num_heads):
        super().__init__()

        # Ensure that the model dimension ( dimension_of_model ) is divisible by the number of heads
        assert dimension_of_model % num_heads == 0, "dimension_of_model must be divisible by num_heads"

        self.num_heads = num_heads
        self.dimension_of_model = dimension_of_model

        # Getting the dimension of K ( key ) 
        self.dimension_of_k = dimension_of_model // num_heads

        # Defining the random weight and biases of dimension of same as dimension_of_model ( Input Vector is given by Embedding Layers )
        self.W_q = nn.Linear(dimension_of_model, dimension_of_model)  # Weights of q ( Query )
        self.W_k = nn.Linear(dimension_of_model, dimension_of_model)  # Weights of k ( Key )
        self.W_v = nn.Linear(dimension_of_model, dimension_of_model)  # Weights of v ( Value )
        self.W_o = nn.Linear(dimension_of_model, dimension_of_model)  # Weights of o ( Out )

        # Assign the Scale_Dot_Product for further use.
        self.attention = Scale_Dot_Product()

    # Split-Heads is used to reshape the size.
    def split_heads(self, x):
        batch_size, seq_length, dimension_of_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.dimension_of_k).transpose(1, 2)

    # Combine-Heads is used to reshape the size back to the original form.
    def combine_heads(self, x):
        batch_size, _, seq_length, dimension_of_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.dimension_of_model)

    def forward(self, Q, K, V, Mask=None):
        # Using the split_heads to reshape the size of Q, K, V
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))

        # After performing the split heads we calculate ATTENTION using Scale_Dot_Product
        attention_output = self.attention(Q, K, V, Mask)

        # Final output returned using combine_heads.
        output = self.W_o(self.combine_heads(attention_output))

        return output


class Point_wise_feedforward(nn.Module):
    def __init__(self, dimension_of_model, dimension_of_feedforward):
        super().__init__()

        self.linear1 = nn.Linear(dimension_of_model, dimension_of_feedforward)
        self.linear2 = nn.Linear(dimension_of_feedforward, dimension_of_model)

    def forward(self, x):
        return self.linear2(F.relu(self.linear1(x)))


class Encoder(nn.Module):
    def __init__(self, dimension_of_model, num_heads, dimension_of_feedforward, dropout=0.1):
        super(Encoder, self).__init__()

        self.multi_head_attention = Multi_Head_Attention(dimension_of_model=dimension_of_model, num_heads=num_heads)
        self.feedforward = Point_wise_feedforward(dimension_of_model=dimension_of_model, dimension_of_feedforward=dimension_of_feedforward)

        self.norm1 = nn.LayerNorm(dimension_of_model)
        self.norm2 = nn.LayerNorm(dimension_of_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attention_output = self.multi_head_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attention_output))

        ff_output = self.feedforward(x)
        x = self.norm2(x + self.dropout(ff_output))

        return x


class Decoder(nn.Module):
    def __init__(self, dimension_of_model, num_heads, dimension_of_feedforward, dropout=0.1):
        super(Decoder, self).__init__()

        self.multi_head_attention = Multi_Head_Attention(dimension_of_model=dimension_of_model, num_heads=num_heads)
        self.cross_multi_head_attention = Multi_Head_Attention(dimension_of_model=dimension_of_model, num_heads=num_heads)

        self.feedforward = Point_wise_feedforward(dimension_of_model=dimension_of_model, dimension_of_feedforward=dimension_of_feedforward)

        self.norm1 = nn.LayerNorm(dimension_of_model)
        self.norm2 = nn.LayerNorm(dimension_of_model)
        self.norm3 = nn.LayerNorm(dimension_of_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        attention_output = self.multi_head_attention(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attention_output))

        cross_attention_output = self.cross_multi_head_attention(x, encoder_output, encoder_output, src_mask)
        x = self.norm2(x + self.dropout(cross_attention_output))

        ff_output = self.feedforward(x)
        x = self.norm3(x + self.dropout(ff_output))

        return x


class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, dimension_of_model=512, num_heads=8, num_layers=6, dimension_of_feedforward=2048, dropout=0.1):
        super(Transformer, self).__init__()

        self.src_embedding = nn.Embedding(src_vocab_size, dimension_of_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, dimension_of_model)

        self.positional_encoding = self.create_positional_encoding(dimension_of_model)

        self.encoder_layers = nn.ModuleList([Encoder(dimension_of_model, num_heads, dimension_of_feedforward, dropout) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([Decoder(dimension_of_model, num_heads, dimension_of_feedforward, dropout) for _ in range(num_layers)])

        self.fc_out = nn.Linear(dimension_of_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def create_positional_encoding(self, dimension_of_model, max_len=5000):
        pe = torch.zeros(max_len, dimension_of_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dimension_of_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / dimension_of_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        return nn.Parameter(pe, requires_grad=False)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        src = self.src_embedding(src) + self.positional_encoding[:src.size(0), :]
        tgt = self.tgt_embedding(tgt) + self.positional_encoding[:tgt.size(0), :]
        src = self.dropout(src)
        tgt = self.dropout(tgt)

        for layer in self.encoder_layers:
            src = layer(src, src_mask)

        for layer in self.decoder_layers:
            tgt = layer(tgt, src, src_mask, tgt_mask)

        output = self.fc_out(tgt)
        return output
