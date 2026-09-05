import torch
from torch import nn


class PatchEmbed2D(nn.Module):
    """2D Image to Patch Embedding."""
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
    """Vision Transformer Encoder for 2D multi-modal images and JEPA context patches."""
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
    """Predicts target patch representations from context tokens and target mask position tokens."""
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
