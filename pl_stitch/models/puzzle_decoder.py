import torch.nn as nn

class AttentionBlock(nn.Module):
    def __init__(self, attn_mul=1, embed_dim=384, num_heads=16):
        super().__init__()
        self.attn_mul = attn_mul
        self.auxiliary_cross_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.auxiliary_layer_norm1 = nn.LayerNorm(embed_dim)
        self.auxiliary_self_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.auxiliary_layer_norm2 = nn.LayerNorm(embed_dim)
        self.auxiliary_linear = nn.Linear(embed_dim, embed_dim, bias=True)
        self.auxiliary_layer_norm3 = nn.LayerNorm(embed_dim)

    def forward(self, current_patch, neighbor_patch):
        # shape of aux_crop and stu_crop: (batch_size, seq_len, embed_dim)

        # MultiheadAttention takes in the query, key, value. Here we use stu_crop to attend to aux_crop.
        cross_attn_output, cross_attn_output_weights = self.auxiliary_cross_attn(current_patch, neighbor_patch, neighbor_patch)
        cross_attn_output = self.auxiliary_layer_norm1(self.attn_mul * cross_attn_output + current_patch) # layer norm with skip connection
        
        # Then we use cross_attn_output to attend to cross_attn_output itself
        self_attn_output, self_attn_output_weights = self.auxiliary_self_attn(cross_attn_output, cross_attn_output, cross_attn_output)
        self_attn_output = self.auxiliary_layer_norm2(self_attn_output + cross_attn_output) # layer norm with skip connection

        # Finally, apply feed forward.
        output = self.auxiliary_linear(self_attn_output)
        output = self.auxiliary_layer_norm3(output + self_attn_output) # layer norm with skip connection
        
        return output, cross_attn_output_weights, self_attn_output_weights



class PuzzleDecoder(nn.Module):
    def __init__(self, attn_mul=4, num_blocks=1, embed_dim=384, num_heads=16):
        super().__init__()
        self.decoder = nn.ModuleList([AttentionBlock(attn_mul, embed_dim, num_heads) for _ in range(num_blocks)])
        self.scorer = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 1),
        )
        nn.init.trunc_normal_(self.scorer[-1].weight, std=0.02)
        nn.init.zeros_(self.scorer[-1].bias)

    def forward(self, current_patch, neighbor_patch, return_feats=False, return_attn=False):
        x = current_patch
        cross_ws, self_ws = [], []
        for block in self.decoder:
            x, cw, sw = block(x, neighbor_patch)
            if return_attn:
                cross_ws.append(cw); self_ws.append(sw)

        scores = self.scorer(x).squeeze(-1)        # [B, N]
        if return_feats or return_attn:
            out = {"scores": scores}
            if return_feats: out["feats"] = x
            if return_attn:  out["cross_w"] = cross_ws; out["self_w"] = self_ws
            return out

        return scores

