from Components import *

class PredictionHead(nn.Module):
    def __init__(self, dim_input, dim_output):
        super(PredictionHead, self).__init__()
        self.regression = nn.Sequential(
            nn.Linear(dim_input, dim_input // 2),
            nn.ReLU(inplace=False),
            nn.Linear(dim_input // 2, dim_output)
        )

    def forward(self, x):
        return self.regression(x)


class Encoder(nn.Module):
    def __init__(self, input_nc):
        super(Encoder, self).__init__()
        in_channels = (96, 192, 384, 768)
        self.backbone = nn.Sequential(SwinTransformer(in_chans=input_nc))
        self.fusion = nn.Sequential(UPerHead(in_channels=in_channels))
        attn = []
        for i in range(len(in_channels)):
            attn.append(self.attn_block(in_channels[i]))
        self.attn = nn.Sequential(*attn)

    def attn_block(self, dim, num_attn=1):
        attn = []
        for _ in range(num_attn):
            attn.append(SAB(dim, dim, num_heads=8, ln=False, attention_dropout=0.1, dim_feedforward=2 * dim))
        return nn.Sequential(*attn)

    def forward(self, x):
        # x: [B, N, Cin, H, W]
        feats = []
        for k in range(x.shape[1]):
            feats.append(self.backbone(x[:, k, :, :, :]))

        # Communication
        out = []
        for l in range(len(feats[0])):
            in_fuse = []
            for k in range(x.shape[1]):
                in_fuse.append(feats[k][l])
            in_fuse = torch.stack(in_fuse, dim=1)  # B, N, C, H//, W//
            B, N, C, H, W = in_fuse.size()
            in_fuse = in_fuse.permute(0, 3, 4, 1, 2).reshape(-1, N, C)
            out_fuse = self.attn[l](in_fuse).reshape(B, H, W, N, C).permute(0, 3, 4, 1, 2)
            out.append(out_fuse)

        # Fusion
        feats_fused = []
        for k in range(x.shape[1]):
            # Collect scales for k-th image
            f_k = (out[0][:, k], out[1][:, k], out[2][:, k], out[3][:, k])
            feats_fused.append(self.fusion(f_k))

        return torch.stack(feats_fused, 1)  # [B, N, C, H/4, W/4]


class UniPSNet():
    def __init__(self, device, lr=0.0001):
        self.device = device
        self.encoder = Encoder(4).to(device)
        self.aggregation = TransformerLayer(dim_input=256 + 3, num_enc_sab=3, dim_hidden=384, num_heads=8, ln=True).to(
            device)
        self.prediction = PredictionHead(384, 3).to(device)

        params = list(self.encoder.parameters()) + list(self.aggregation.parameters()) + list(
            self.prediction.parameters())
        self.optimizer = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.999), weight_decay=0.01)
        self.criterion = nn.MSELoss(reduction='sum').to(device)

    def train(self):
        self.encoder.train()
        self.aggregation.train()
        self.prediction.train()

    def eval(self):
        self.encoder.eval()
        self.aggregation.eval()
        self.prediction.eval()

    def step(self, batch, decoder_imgsize, encoder_imgsize=(256, 256), mode='Train', num_samples=2500):
        """
        Perform one forward pass (and optionally backward pass) through the network.

        This method handles both training and testing modes with different behaviors:
        - Train: Sample random pixels, compute loss, backpropagate, update weights
        - Test: Process all valid pixels, return predicted normal maps

        Mathematical Formulation:
            Given N images {I_k}_{k=1}^N and mask M:

            1. Feature Encoding (at canonical resolution):
               F_k = Encoder(I_1 ⊙ M, ..., I_N ⊙ M) ∈ ℝ^{256×H/4×W/4}

            2. For each pixel p = (i,j) in decoder resolution:
               a) Sample features: f_k(p) = BilinearSample(F_k, p)
               b) Aggregate: h_p = Transformer([I_1(p), f_1(p)], ..., [I_N(p), f_N(p)])
               c) Predict: n̂_p = Normalize(PredictionHead(h_p))

            3. Training Loss:
               L = (1/S) Σ_{p∈S} ||n̂_p - n_p^gt||₂²
               where S is a random sample of num_samples valid pixels

            4. Testing Output:
               Return normalized normal map: n̂ ∈ ℝ^{B×3×H×W}, ||n̂_p||₂ = 1

        Args:
            batch (tuple): Tuple of (images, normals, mask) where:
                - images: [B, C, H, W, N] RGB images
                - normals: [B, 3, H, W] Ground truth normal maps
                - mask: [B, 1, H, W] Valid pixel mask
            decoder_imgsize (tuple): (H, W) resolution for prediction
            encoder_imgsize (tuple): (H, W) resolution for feature encoding (default: 256×256)
            mode (str): 'Train' or 'Test' (default: 'Train')
            num_samples (int): Number of pixels to sample per object in training (default: 2500)

        Returns:
            If mode == 'Train':
                float: Average loss value over the batch
            If mode == 'Test':
                torch.Tensor: Predicted normal maps of shape [B, 3, H_dec, W_dec]

        Note:
            - All normals are L2-normalized: ||n||₂ = 1
            - Training uses random pixel sampling for efficiency
            - Testing processes pixels in chunks to avoid OOM
        """
        img, nml, mask = batch
        img = img.permute(0, 4, 1, 2, 3).to(self.device)  # B, N, C, H, W
        nml = nml.to(self.device)
        mask = mask.to(self.device)

        B, N, C_img, H, W = img.shape

        # --- Feature Encoding (Canonical Resolution) ---
        img_enc = img.reshape(-1, C_img, H, W)
        img_enc = F.interpolate(img_enc, size=encoder_imgsize, mode='bilinear', align_corners=False).reshape(B, N,
                                                                                                             C_img,
                                                                                                             encoder_imgsize[
                                                                                                                 0],
                                                                                                             encoder_imgsize[
                                                                                                                 1])
        mask_enc = F.interpolate(mask, size=encoder_imgsize, mode='nearest')

        data = torch.cat([img_enc * mask_enc.unsqueeze(1).expand(-1, N, -1, -1, -1),
                          mask_enc.unsqueeze(1).expand(-1, N, -1, -1, -1)], dim=2)
        feats = self.encoder(data)  # [B, N, 256, H/4, W/4]
        C_feat = feats.shape[2]  # 256

        # --- Decoding / Prediction ---
        img_dec = img.reshape(-1, C_img, H, W)
        img_dec = F.interpolate(img_dec, size=decoder_imgsize, mode='bilinear', align_corners=False).reshape(B, N,
                                                                                                             C_img,
                                                                                                             decoder_imgsize[
                                                                                                                 0],
                                                                                                             decoder_imgsize[
                                                                                                                 1])
        mask_dec = F.interpolate(mask, size=decoder_imgsize, mode='nearest')
        nml_dec = F.normalize(F.interpolate(nml, size=decoder_imgsize, mode='bilinear', align_corners=False), p=2,
                              dim=1)

        loss = 0
        H_dec, W_dec = decoder_imgsize

        if mode == 'Train':
            for b in range(B):
                m_ = mask_dec[b].view(-1)
                n_ = nml_dec[b].view(3, -1).permute(1, 0)
                ids = torch.nonzero(m_ > 0).squeeze()

                if ids.numel() > num_samples:
                    perm = torch.randperm(ids.numel())
                    ids = ids[perm[:num_samples]]

                if ids.numel() == 0: continue

                # Get Coords for Grid Sample
                coords = ind2coords((H_dec, W_dec), ids).to(self.device)

                x_agg_list = []
                feat_b = feats[b]  # N, 256, H/4, W/4

                for k in range(N):
                    # Sample feature
                    # FIX: Use view(C, -1) instead of squeeze(2).squeeze(2) to correctly handle S > 1
                    f = F.grid_sample(feat_b[k:k + 1], coords, mode='bilinear', align_corners=False).view(C_feat,
                                                                                                          -1).permute(1,
                                                                                                                      0)  # Samples, 256

                    # Sample raw image
                    o = img_dec[b, k].view(3, -1).permute(1, 0)[ids]  # Samples, 3
                    x_agg_list.append(torch.cat([o, f], dim=1))

                x_agg = torch.stack(x_agg_list, dim=1)  # Samples, N, 256+3

                feat_gg = self.aggregation(x_agg)
                out_nml = self.prediction(feat_gg)
                nout_ = F.normalize(out_nml, dim=1, p=2)

                loss += self.criterion(nout_, n_[ids]) / ids.numel()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            return loss.item()

        elif mode == 'Test':
            output_maps = []
            with torch.no_grad():
                for b in range(B):
                    m_ = mask_dec[b].view(-1)
                    ids = torch.nonzero(m_ > 0).squeeze()

                    # Process in chunks
                    chunk_size = 5000
                    nml_pred_full = torch.zeros(H_dec * W_dec, 3).to(self.device)

                    if ids.numel() > 0:
                        chunks = torch.split(ids, chunk_size)
                        for chunk_ids in chunks:
                            coords = ind2coords((H_dec, W_dec), chunk_ids).to(self.device)
                            x_agg_list = []
                            feat_b = feats[b]
                            for k in range(N):
                                # FIX: Same dimension fix for testing
                                f = F.grid_sample(feat_b[k:k + 1], coords, mode='bilinear', align_corners=False).view(
                                    C_feat, -1).permute(1, 0)
                                o = img_dec[b, k].view(3, -1).permute(1, 0)[chunk_ids]
                                x_agg_list.append(torch.cat([o, f], dim=1))
                            x_agg = torch.stack(x_agg_list, dim=1)
                            feat_gg = self.aggregation(x_agg)
                            out_nml = self.prediction(feat_gg)
                            nout_ = F.normalize(out_nml, dim=1, p=2)
                            nml_pred_full[chunk_ids] = nout_

                    output_maps.append(nml_pred_full.view(H_dec, W_dec, 3).permute(2, 0, 1))
                return torch.stack(output_maps)