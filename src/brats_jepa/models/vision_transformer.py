import torch
from torch import nn


class PatchEmbed2D(nn.Module):
    r"""
    2D Multi-Modal MRI Patch Embedding Layer.

    Mathematical Rationale & Defense Context:
    -----------------------------------------
    1. Patch Linear Projection:
       Splits a 4-channel 2D MRI slice x \in \mathbb{R}^{B \times 4 \times 240 \times 240}
       into a grid of non-overlapping patches of size P \times P (16 \times 16).
       Implemented as a 2D convolution with kernel_size = stride = P = 16:
           \text{grid\_size} = \left(\frac{240}{16}, \frac{240}{16}\right) = (15, 15) \implies N = 225 \text{ patches}.
       Linearly projects each 4 \times 16 \times 16 = 1024-dimensional raw voxel patch
       to the latent embedding dimension D = 384.

    References:
    -----------
    - Dosovitskiy, A., et al. (2020). "An Image is Worth 16x16 Words: Transformers for Image
      Recognition at Scale." ICLR 2021.
    """
    def __init__(self, img_size: int = 240, patch_size: int = 16, in_channels: int = 4, embed_dim: int = 384):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size // patch_size, img_size // patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W] -> [B, embed_dim, grid_h, grid_w] -> [B, num_patches, embed_dim]
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

class VisionTransformerEncoder2D(nn.Module):
    r"""
    Vision Transformer Encoder for 2D Multi-Modal MRI Representations.

    Mathematical Rationale & Defense Context:
    -----------------------------------------
    1. Pre-LayerNorm Transformer Architecture:
       Uses pre-LN Transformer layers (`norm_first=True`) with GELU activations.
       Pre-LN guarantees identity gradient paths across residual connections, eliminating
       the warm-up sensitivity and gradient dissipation typical of Post-LN ViTs (Xiong et al., 2020).

    2. Context Patch Selection (Attention Leakage Prevention):
       When patch_indices are provided, ONLY context patches are fed into self-attention blocks.
       This design choice has two major scientific justifications:
       a) **Zero Information Leakage**: Prevents attention keys/queries from attending to target
          patches, ensuring strict self-supervised prediction difficulty.
       b) **Quadratic Speedup**: Self-attention complexity scales as O(N_{ctx}^2) rather than
          O(N_{patches}^2). With N_{ctx} = 96 vs N = 225, attention FLOPs are reduced by ~82%.

    References:
    -----------
    - Dosovitskiy, A., et al. (2020). "An Image is Worth 16x16 Words." ICLR 2021.
    - Assran, M., et al. (2023). "Self-Supervised Learning from Images with a Joint-Embedding
      Predictive Architecture." IEEE/CVF CVPR 2023.
    - Xiong, R., et al. (2020). "On Layer Normalization in the Transformer Architecture." ICML 2020.
    """
    def __init__(
        self,
        img_size: int = 240,
        patch_size: int = 16,
        in_channels: int = 4,
        embed_dim: int = 384,
        depth: int = 8,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.patch_embed = PatchEmbed2D(img_size, patch_size, in_channels, embed_dim)
        self.num_patches = self.patch_embed.num_patches
        
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(encoder_layer, num_layers=depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor, patch_indices: torch.Tensor | None = None) -> torch.Tensor:
        """
        x: [B, C, H, W]
        patch_indices: Optional [B, N_ctx] context patch subset indices
        """
        B = x.shape[0]
        tokens = self.patch_embed(x) + self.pos_embed  # [B, N_patches, D]
        
        if patch_indices is not None:
            # Select subset of context patches
            # patch_indices can be [B, N_ctx] or 1D [N_ctx] shared across batch
            if patch_indices.dim() == 1:
                tokens = tokens[:, patch_indices, :]
            else:
                tokens = tokens.gather(1, patch_indices.unsqueeze(-1).expand(-1, -1, tokens.size(-1)))
                
        out = self.blocks(tokens)
        out = self.norm(out)
        return out

class JEPAPredictor(nn.Module):
    r"""
    JEPA Latent Space Target Representation Predictor.

    Mathematical Rationale & Defense Context:
    -----------------------------------------
    1. Capacity Asymmetry (Encoder vs Predictor):
       - Encoder: D = 384, depth = 8, 6 attention heads.
       - Predictor: D_{pred} = 192, depth = 4, 6 attention heads.
       In predictive self-supervised learning, if the predictor is given comparable or
       greater parameter capacity than the encoder, the system is susceptible to "representation
       offloading", where the encoder learns trivial identity features and the predictor does the
       heavy semantic reconstruction. Narrowing the predictor (D_{pred} = D / 2) and restricting
       its depth forces the encoder to maximize the semantic density and spatial linearly-separable
       structure of its representations.

    2. Target Spatial Querying via Positional Embeddings:
       The target tokens are formed by broadcasting a learned `mask_token` and adding the
       positional embeddings corresponding to target patch coordinates:
           t_{\text{query}}^{(j)} = m_{\text{token}} + p_{\text{pos}}[j] \quad \text{for } j \in \text{target\_indices}
       Context tokens (projected to D_{pred} + context position embeddings) and target queries
       are concatenated into full self-attention blocks. The target output slots predict
       the teacher's latent representation s_y at each queried patch coordinate.

    References:
    -----------
    - Assran, M., et al. (2023). "Self-Supervised Learning from Images with a Joint-Embedding
      Predictive Architecture." IEEE/CVF CVPR 2023.
    - He, K., et al. (2022). "Masked Autoencoders Are Scalable Vision Learners." CVPR 2022.
    """
    def __init__(
        self,
        embed_dim: int = 384,
        pred_embed_dim: int = 192,
        num_patches: int = 225,
        depth: int = 4,
        num_heads: int = 6,
    ):
        super().__init__()
        self.predictor_embed = nn.Linear(embed_dim, pred_embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, pred_embed_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, pred_embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        predictor_layer = nn.TransformerEncoderLayer(
            d_model=pred_embed_dim,
            nhead=num_heads,
            dim_feedforward=int(pred_embed_dim * 4),
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(predictor_layer, num_layers=depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(pred_embed_dim)
        self.predictor_proj = nn.Linear(pred_embed_dim, embed_dim)

    def forward(
        self,
        context_tokens: torch.Tensor,
        context_indices: torch.Tensor,
        target_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        context_tokens: [B, N_ctx, D]
        context_indices: [B, N_ctx] or [N_ctx]
        target_indices: [B, N_tgt] or [N_tgt]
        Returns predicted target tokens: [B, N_tgt, D]
        """
        B = context_tokens.shape[0]
        ctx_proj = self.predictor_embed(context_tokens)  # [B, N_ctx, D_pred]
        
        # Add positional embedding to context tokens
        if context_indices.dim() == 1:
            ctx_pos = self.pos_embed[:, context_indices, :].expand(B, -1, -1)
        else:
            ctx_pos = self.pos_embed.expand(B, -1, -1).gather(
                1, context_indices.unsqueeze(-1).expand(-1, -1, self.pos_embed.size(-1))
            )
        ctx_proj = ctx_proj + ctx_pos

        # Compute positional embedding for target tokens
        if target_indices.dim() == 1:
            tgt_pos = self.pos_embed[:, target_indices, :].expand(B, -1, -1)
        else:
            tgt_pos = self.pos_embed.expand(B, -1, -1).gather(
                1, target_indices.unsqueeze(-1).expand(-1, -1, self.pos_embed.size(-1))
            )
            
        N_tgt = target_indices.shape[-1] if target_indices.dim() > 0 else len(target_indices)
        tgt_tokens = self.mask_token.expand(B, N_tgt, -1) + tgt_pos
        
        # Concatenate context tokens and mask target tokens
        full_tokens = torch.cat([ctx_proj, tgt_tokens], dim=1)
        out = self.blocks(full_tokens)
        out = self.norm(out)
        
        # Extract predicted target tokens (last N_tgt tokens)
        pred_target = out[:, -N_tgt:, :]
        pred_target = self.predictor_proj(pred_target)  # [B, N_tgt, D]
        return pred_target
