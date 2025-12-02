from SwinTransformer import *
import math

class PPM(nn.ModuleList):
    def __init__(self, pool_scales, in_channels, channels):
        super(PPM, self).__init__()
        self.pool_scales = pool_scales
        self.in_channels = in_channels
        self.channels = channels
        for pool_scale in pool_scales:
            self.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(pool_scale),
                    nn.Conv2d(self.in_channels, self.channels, kernel_size=1),
                    nn.ReLU(inplace=False)
                )
            )
    def forward(self, x):
        ppm_outs = []
        for ppm in self:
            ppm_out = ppm(x)
            upsampled_ppm_out = F.interpolate(ppm_out, size=x.size()[2:], mode='bilinear', align_corners=False)
            ppm_outs.append(upsampled_ppm_out)
        return ppm_outs

class UPerHead(nn.Module):
    def __init__(self, in_channels=(96, 192, 384, 768), channels=256, pool_scales=(1, 2, 3, 6)):
        super(UPerHead, self).__init__()
        self.in_channels = in_channels
        self.channels = channels
        self.psp_modules = PPM(pool_scales, self.in_channels[-1], self.channels)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(self.in_channels[-1] + len(pool_scales) * self.channels, self.channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=False))
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for in_channels in self.in_channels[:-1]:
            l_conv = nn.Sequential(nn.Conv2d(in_channels, self.channels, kernel_size=1, padding=0), nn.ReLU(inplace=False))
            fpn_conv = nn.Sequential(nn.Conv2d(self.channels, self.channels, kernel_size=3, padding=1), nn.ReLU(inplace=False))
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)
        self.fpn_bottleneck = nn.Sequential(
            nn.Conv2d(len(self.in_channels) * self.channels, self.channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=False))

    def psp_forward(self, inputs):
        x = inputs[-1]
        psp_outs = [x]
        psp_outs.extend(self.psp_modules(x))
        psp_outs = torch.cat(psp_outs, dim=1)
        output = self.bottleneck(psp_outs)
        return output

    def forward(self, inputs):
        laterals = [lateral_conv(inputs[i]) for i, lateral_conv in enumerate(self.lateral_convs)]
        laterals.append(self.psp_forward(inputs))
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] = laterals[i - 1] + F.interpolate(laterals[i], size=prev_shape, mode='bilinear', align_corners=False)
        fpn_outs = [self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels - 1)]
        fpn_outs.append(laterals[-1])
        for i in range(used_backbone_levels - 1, 0, -1):
            fpn_outs[i] = F.interpolate(fpn_outs[i], size=fpn_outs[0].shape[2:], mode='bilinear', align_corners=False)
        fpn_outs = torch.cat(fpn_outs, dim=1)
        output = self.fpn_bottleneck(fpn_outs)
        return output

# --- Transformer Layers for Aggregation ---
class MultiHeadSelfAttentionBlock(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, ln=False, attention_dropout=0.1, dim_feedforward=512):
        super(MultiHeadSelfAttentionBlock, self).__init__()
        self.dim_V, self.dim_Q, self.dim_K, self.num_heads = dim_out, dim_in, dim_in, num_heads
        self.fc_q = nn.Linear(self.dim_Q, self.dim_V)
        self.fc_k = nn.Linear(self.dim_K, self.dim_V)
        self.fc_v = nn.Linear(self.dim_K, self.dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(self.dim_Q)
            self.ln1 = nn.LayerNorm(self.dim_V)
        self.dropout_attention = nn.Dropout(attention_dropout)
        self.fc_o1 = nn.Linear(self.dim_V, dim_feedforward)
        self.fc_o2 = nn.Linear(dim_feedforward, self.dim_V)
        self.dropout1 = nn.Dropout(attention_dropout)
        self.dropout2 = nn.Dropout(attention_dropout)

    def forward(self, x, y):
        x = x if getattr(self, 'ln0', None) is None else self.ln0(x)
        Q = self.fc_q(x)
        K, V = self.fc_k(y), self.fc_v(y)
        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(int(dim_split), 2), 0)
        K_ = torch.cat(K.split(int(dim_split), 2), 0)
        V_ = torch.cat(V.split(int(dim_split), 2), 0)
        A = self.dropout_attention(torch.softmax(Q_.bmm(K_.transpose(1, 2)) / math.sqrt(self.dim_V), 2))
        A = A.bmm(V_)
        O = torch.cat((Q_ + A).split(Q.size(0), 0), 2)
        O_ = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        O = O + self.dropout2(self.fc_o2(self.dropout1(F.gelu(self.fc_o1(O_)))))
        return O

class SAB(nn.Module):
    """
    Self-Attention Block (SAB).

    A special case of Multi-Head Attention where query, key, and value
    all come from the same input. Enables communication within a set.

    Mathematical Formulation:
        SAB(X) = MultiHeadAttention(X, X, X)

        where X ∈ ℝ^{N×D} is a set of N elements with dimension D.

    This is used for set-to-set operations, allowing each element to
    attend to all other elements in the set.

    Args:
        dim_in (int): Input dimension
        dim_out (int): Output dimension
        num_heads (int): Number of attention heads (default: 4)
        ln (bool): Use layer normalization (default: False)
        attention_dropout (float): Dropout rate (default: 0.1)
        dim_feedforward (int): FFN hidden dimension (default: 512)
    """
    def __init__(self, dim_in, dim_out, num_heads=4, ln=False, attention_dropout=0.1, dim_feedforward=512):
        super(SAB, self).__init__()
        self.mab = MultiHeadSelfAttentionBlock(dim_in, dim_out, num_heads, ln=ln, attention_dropout=attention_dropout, dim_feedforward=dim_feedforward)

    def forward(self, X):
        """
        Apply self-attention to input set.

        Args:
            X (torch.Tensor): Input set of shape [B, N, D]

        Returns:
            torch.Tensor: Output set of shape [B, N, D_out]
        """
        return self.mab(X, X)

class PMA(nn.Module):
    """
    Pooling by Multi-head Attention (PMA).

    Uses learnable seed vectors to aggregate information from a set
    into a fixed number of outputs. Acts as an attention-based pooling layer.

    Mathematical Formulation:
        Given input set X ∈ ℝ^{N×D} and learnable seeds S ∈ ℝ^{K×D}:

        PMA(X) = MultiHeadAttention(S, X, X) ∈ ℝ^{K×D}

        where:
        - S: Learnable seed vectors (queries)
        - X: Input set (keys and values)
        - K: Number of output elements (num_seeds)

    This reduces a variable-size set to a fixed-size representation.

    Args:
        dim (int): Feature dimension
        num_heads (int): Number of attention heads
        num_seeds (int): Number of output elements
        ln (bool): Use layer normalization (default: False)
    """
    def __init__(self, dim, num_heads, num_seeds, ln=False):
        super(PMA, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MultiHeadSelfAttentionBlock(dim, dim, num_heads, ln=ln)

    def forward(self, X):
        """
        Pool input set using attention.

        Args:
            X (torch.Tensor): Input set of shape [B, N, D]

        Returns:
            torch.Tensor: Pooled output of shape [B, num_seeds, D]
        """
        return self.mab(self.S.repeat(X.size(0), 1, 1), X)

class TransformerLayer(nn.Module):
    """
    Transformer-based Set Encoder and Decoder.

    Processes sets of features using stacked self-attention blocks (encoder)
    and pools them using PMA (decoder) to produce fixed-size outputs.

    Architecture:
        Encoder: SAB → SAB → ... → SAB (num_enc_sab blocks)
        Decoder: PMA (attention-based pooling)

    Mathematical Formulation:
        Given input set X ∈ ℝ^{N×D_in}:

        1. Encoder (stack of K SABs):
           H_0 = Linear(X) ∈ ℝ^{N×D_hidden}
           H_k = SAB(H_{k-1}) for k = 1,...,K

        2. Decoder (PMA pooling):
           Z = PMA(H_K) ∈ ℝ^{num_outputs×D_hidden}

        3. Flatten:
           Output = Flatten(Z) ∈ ℝ^{num_outputs × D_hidden}

    This is used in UniPS to aggregate features from multiple views.

    Args:
        dim_input (int): Input feature dimension (e.g., 256+3=259 for RGB+features)
        num_enc_sab (int): Number of SAB blocks in encoder (default: 3)
        num_outputs (int): Number of output tokens (default: 1)
        dim_hidden (int): Hidden dimension (default: 384)
        dim_feedforward (int): FFN dimension (default: 1024)
        num_heads (int): Number of attention heads (default: 8)
        ln (bool): Use layer normalization (default: False)
        attention_dropout (float): Dropout rate (default: 0.1)
    """
    def __init__(self, dim_input, num_enc_sab=3, num_outputs=1, dim_hidden=384, dim_feedforward=1024, num_heads=8, ln=False, attention_dropout=0.1):
        super(TransformerLayer, self).__init__()
        self.num_outputs = num_outputs
        self.dim_hidden = dim_hidden
        modules_enc = [SAB(dim_input, dim_hidden, num_heads, ln=ln, attention_dropout=attention_dropout, dim_feedforward=dim_feedforward)]
        for _ in range(num_enc_sab):
            modules_enc.append(SAB(dim_hidden, dim_hidden, num_heads, ln=ln, attention_dropout=attention_dropout, dim_feedforward=dim_feedforward))
        self.enc = nn.Sequential(*modules_enc)
        self.dec = nn.Sequential(PMA(dim_hidden, num_heads, num_outputs))

    def forward(self, x):
        """
        Forward pass through transformer encoder-decoder.

        Args:
            x (torch.Tensor): Input set of shape [B, N, D_in] where:
                - B: Batch size (number of pixels/samples)
                - N: Set size (number of views)
                - D_in: Input dimension per element

        Returns:
            torch.Tensor: Aggregated features of shape [B, num_outputs × D_hidden]

        Example:
            In UniPS, for each pixel p with N=10 views:
            x_p ∈ ℝ^{10×259} → output ∈ ℝ^{384}
        """
        x = self.enc(x)
        x = self.dec(x)
        feat = x.view(-1, self.num_outputs * self.dim_hidden)
        return feat