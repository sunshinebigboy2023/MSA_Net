import torch
from torch import nn
import torch.nn.functional as F


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block_softmoe(nn.Module):
    def __init__(
            self,
            dim,
            num_heads=8,
            attn_drop=0,
            proj_drop=0,
            mlp_ratio=1,
            num_experts=1,
            use_layerwise_sparse_router=False,
            num_shared_experts=0,
            router_topk=2,
            router_noise_std=0.0,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.use_layerwise_sparse_router = use_layerwise_sparse_router
        self.num_shared_experts = max(0, int(num_shared_experts))
        self.router_topk = max(0, int(router_topk))
        self.router_noise_std = max(0.0, float(router_noise_std))
        self.transformer_a = nn.ModuleList(
            [
                Attention(
                    dim,
                    num_heads=num_heads,
                    attn_drop=attn_drop,
                    proj_drop=proj_drop,
                    mlp_ratio=mlp_ratio,
                )
                for _ in range(num_experts)
            ]
        )
        self.transformer_t = nn.ModuleList(
            [
                Attention(
                    dim,
                    num_heads=num_heads,
                    attn_drop=attn_drop,
                    proj_drop=proj_drop,
                    mlp_ratio=mlp_ratio,
                )
                for _ in range(num_experts)
            ]
        )
        self.transformer_v = nn.ModuleList(
            [
                Attention(
                    dim,
                    num_heads=num_heads,
                    attn_drop=attn_drop,
                    proj_drop=proj_drop,
                    mlp_ratio=mlp_ratio,
                )
                for _ in range(num_experts)
            ]
        )
        if self.use_layerwise_sparse_router:
            self.transformer_shared = nn.ModuleList(
                [
                    Attention(
                        dim,
                        num_heads=num_heads,
                        attn_drop=attn_drop,
                        proj_drop=proj_drop,
                        mlp_ratio=mlp_ratio,
                    )
                    for _ in range(self.num_shared_experts)
                ]
            )
            router_hidden = max(int(dim * mlp_ratio), dim)
            candidate_count = self.num_experts + self.num_shared_experts
            self.router_a = Mlp(dim, hidden_features=router_hidden, out_features=candidate_count, drop=proj_drop)
            self.router_t = Mlp(dim, hidden_features=router_hidden, out_features=candidate_count, drop=proj_drop)
            self.router_v = Mlp(dim, hidden_features=router_hidden, out_features=candidate_count, drop=proj_drop)
        else:
            self.transformer_shared = None
            self.router_a = None
            self.router_t = None
            self.router_v = None

    def forward(self, x, cross_modality='atv', mask_modality=None, mask=None):
        # x: [B, s, C]
        modality_to_experts = {
            'a': self.transformer_a,
            't': self.transformer_t,
            'v': self.transformer_v,
        }
        if cross_modality not in modality_to_experts:
            raise ValueError(f"Unsupported cross_modality '{cross_modality}'")

        expert_modules = list(modality_to_experts[cross_modality])
        if self.transformer_shared is not None:
            expert_modules = expert_modules + list(self.transformer_shared)
        expert_outputs = [
            expert(x, mask_modality, mask) for expert in expert_modules
        ]
        stacked = torch.stack(expert_outputs, dim=2)  # [B, s, num_candidates, C]

        if self.router_a is not None:
            router_map = {
                'a': self.router_a,
                't': self.router_t,
                'v': self.router_v,
            }
            router_logits = router_map[cross_modality](x)
            if self.training and self.router_noise_std > 0.0:
                router_logits = router_logits + torch.randn_like(router_logits) * self.router_noise_std
            router_weights = torch.softmax(router_logits, dim=-1)
            if 0 < self.router_topk < router_weights.size(-1):
                topk = min(self.router_topk, router_weights.size(-1))
                top_vals, top_idx = torch.topk(router_weights, k=topk, dim=-1)
                sparse_weights = torch.zeros_like(router_weights).scatter(-1, top_idx, top_vals)
                sparse_weights = sparse_weights / sparse_weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            else:
                sparse_weights = router_weights
            aggregated = torch.sum(sparse_weights.unsqueeze(-1) * stacked, dim=2)
            probs = sparse_weights.clamp_min(1e-8)
            entropy = -(probs * probs.log()).sum(dim=-1).mean()
            stats = {
                'active': float((sparse_weights > 0).float().sum(dim=-1).mean().detach().cpu().item()),
                'entropy': float(entropy.detach().cpu().item()),
            }
            return aggregated, stacked, sparse_weights, stats

        aggregated = stacked.mean(dim=2)
        return aggregated, stacked, None, None



class Attention(nn.Module):
    def __init__(
            self,
            dim,
            num_heads=8,
            attn_drop=0.0,
            proj_drop=0.0,
            mlp_ratio=1.0
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.scale = head_dim ** -0.5
        self.q, self.k, self.v = nn.Linear(dim, dim), nn.Linear(dim, dim), nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            drop=proj_drop,
        )


    def forward(self, x, mask_modality, mask=None):
        B, seq_len, C = x.shape

        q = self.q(x).reshape(B, seq_len, self.num_heads, -1).permute(0, 2, 1, 3)
        k = self.k(x).reshape(B, seq_len, self.num_heads, -1).permute(0, 2, 1, 3)
        v = self.v(x).reshape(B, seq_len, self.num_heads, -1).permute(0, 2, 1, 3)

        q = q * self.scale
        attn = (q.float() @ k.float().transpose(-2, -1))  # [B, heads, s, s]

        if mask is not None:
            mask = mask.bool()
            mask = {'a':mask[:, :seq_len], 't':mask[:, seq_len:2*seq_len], 'v':mask[:, 2*seq_len:3*seq_len]}
            mask = mask[mask_modality]
            attn = self.attn_drop(attn.masked_fill(~mask[:, None, None, :], float("-inf")).softmax(dim=-1).type_as(x))
            attn = torch.where(torch.isnan(attn), torch.full_like(attn, 0), attn)

        x_out = (attn @ v).transpose(1, 2).reshape(B, seq_len, C)
        x_out = x_out + self.mlp(x_out)

        return x_out


class Block(nn.Module):
    def __init__(
            self,
            dim,
            num_heads,
            mlp_ratio=4.0,
            drop=0.0,
            attn_drop=0.0,
            depth=4,
            num_experts=1,
            use_layerwise_sparse_router=False,
            num_shared_experts=0,
            router_topk=2,
            router_noise_std=0.0,
    ):
        super().__init__()
        self.drop = drop
        self.use_layerwise_sparse_router = use_layerwise_sparse_router
        self.latest_router_stats = {}

        self.blocks = nn.ModuleList(
            [
                Block_softmoe(dim,
                              num_heads=num_heads,
                              attn_drop=attn_drop,
                              proj_drop=drop,
                              mlp_ratio=mlp_ratio,
                              num_experts=num_experts,
                              use_layerwise_sparse_router=use_layerwise_sparse_router,
                              num_shared_experts=num_shared_experts,
                              router_topk=router_topk,
                              router_noise_std=router_noise_std,)
                for i in range(depth)
            ]
        )

    def forward(self, x, first_stage, mask=None, modality=None):
        if self.use_layerwise_sparse_router:
            if modality is None:
                raise ValueError("modality must be provided when layerwise sparse router is enabled")
            layer_stats = []
            for layer_idx, block in enumerate(self.blocks):
                aggregated, _, _, stats = block(x, cross_modality=modality, mask_modality=modality, mask=mask)
                x = x + aggregated
                if stats is not None:
                    layer_stats.append(stats)
            if layer_stats:
                self.latest_router_stats[modality] = {
                    'active': float(sum(item['active'] for item in layer_stats) / len(layer_stats)),
                    'entropy': float(sum(item['entropy'] for item in layer_stats) / len(layer_stats)),
                }
            return x
        if first_stage:
            for layer_idx, block in enumerate(self.blocks):
                aggregated, _, _, _ = block(x, cross_modality=modality, mask_modality=modality, mask=mask)
                x = x + aggregated
            return x
        else:
            x_cross_a, x_cross_t, x_cross_v = torch.clone(x), torch.clone(x), torch.clone(x)
            experts_out_a = experts_out_t = experts_out_v = None
            for layer_idx, block in enumerate(self.blocks):
                residual_a, residual_t, residual_v = x_cross_a, x_cross_t, x_cross_v
                aggregated_a, experts_stack_a, _, _ = block(residual_a, cross_modality='a', mask_modality=modality, mask=mask)
                aggregated_t, experts_stack_t, _, _ = block(residual_t, cross_modality='t', mask_modality=modality, mask=mask)
                aggregated_v, experts_stack_v, _, _ = block(residual_v, cross_modality='v', mask_modality=modality, mask=mask)
                x_cross_a = residual_a + aggregated_a
                x_cross_t = residual_t + aggregated_t
                x_cross_v = residual_v + aggregated_v
                experts_out_a = residual_a.unsqueeze(2) + experts_stack_a
                experts_out_t = residual_t.unsqueeze(2) + experts_stack_t
                experts_out_v = residual_v.unsqueeze(2) + experts_stack_v
            if experts_out_a is None or experts_out_t is None or experts_out_v is None:
                raise RuntimeError("No expert outputs were produced. Check Block depth.")
            experts_out_a = experts_out_a.flatten(2)
            experts_out_t = experts_out_t.flatten(2)
            experts_out_v = experts_out_v.flatten(2)
            return torch.cat([experts_out_a, experts_out_t, experts_out_v], dim=-1)
