import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.Attention_softmoe import *

try:
    from cross_attn_encoder import CMELayer, BertConfig
except ImportError:  # ensure parent directory (mul_div) is visible when run from MoMKE/
    import sys
    from pathlib import Path

    current_dir = Path(__file__).resolve().parent
    parent_dir = current_dir.parent
    if str(parent_dir) not in sys.path:
        sys.path.append(str(parent_dir))
    from cross_attn_encoder import CMELayer, BertConfig


class MoMKE(nn.Module):

    def __init__(self, args, adim, tdim, vdim, D_e, n_classes, depth=4, num_heads=4, mlp_ratio=1, drop_rate=0, attn_drop_rate=0, num_experts=2, no_cuda=False):
        super(MoMKE, self).__init__()
        self.n_classes = n_classes
        self.D_e = D_e
        self.num_heads = num_heads
        D = 3 * D_e
        self.device = args.device
        self.no_cuda = no_cuda
        self.adim, self.tdim, self.vdim = adim, tdim, vdim
        self.out_dropout = args.drop_rate
        self.num_experts = getattr(args, "num_experts", num_experts)
        self.num_experts_total = 3 * self.num_experts
        self.use_layerwise_sparse_router = getattr(args, "enable_layerwise_sparse_router", False)
        self.num_shared_experts = max(0, int(getattr(args, "num_shared_experts", 1)))
        self.router_topk = max(0, int(getattr(args, "router_topk", 2)))
        self.router_noise_std = max(0.0, float(getattr(args, "router_noise_std", 0.0)))
        self.latest_router_stats = None

        self.a_in_proj = nn.Sequential(nn.Linear(self.adim, D_e))
        self.t_in_proj = nn.Sequential(nn.Linear(self.tdim, D_e))
        self.v_in_proj = nn.Sequential(nn.Linear(self.vdim, D_e))
        audio_drop_rate = getattr(args, "audio_drop_rate", None)
        text_drop_rate = getattr(args, "text_drop_rate", None)
        visual_drop_rate = getattr(args, "visual_drop_rate", None)
        if audio_drop_rate is None:
            audio_drop_rate = min(0.5, args.drop_rate + 0.1)
        if text_drop_rate is None:
            text_drop_rate = args.drop_rate
        if visual_drop_rate is None:
            visual_drop_rate = min(0.5, args.drop_rate + 0.1)
        self.dropout_a = nn.Dropout(audio_drop_rate)
        self.dropout_t = nn.Dropout(text_drop_rate)
        self.dropout_v = nn.Dropout(visual_drop_rate)
        self.post_dropout_a = nn.Dropout(audio_drop_rate)
        self.post_dropout_t = nn.Dropout(text_drop_rate)
        self.post_dropout_v = nn.Dropout(visual_drop_rate)

        self.block = Block(
                    dim=D_e,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    depth=depth,
                    num_experts=self.num_experts,
                    use_layerwise_sparse_router=self.use_layerwise_sparse_router,
                    num_shared_experts=self.num_shared_experts,
                    router_topk=self.router_topk,
                    router_noise_std=self.router_noise_std,
                )
        self.proj1 = nn.Linear(D, D)
        self.nlp_head_a = nn.Linear(D_e, n_classes)
        self.nlp_head_t = nn.Linear(D_e, n_classes)
        self.nlp_head_v = nn.Linear(D_e, n_classes)
        self.nlp_head = nn.Linear(D, n_classes)
        self.enable_late_unimodal_calibration = getattr(args, "enable_late_unimodal_calibration", False)
        self.late_calib_scale = max(0.0, float(getattr(args, "late_calib_scale", 0.2)))
        if self.enable_late_unimodal_calibration and self.late_calib_scale > 0.0:
            self.late_calib_router = nn.Linear(D, 3)
            self.late_calib_alpha = nn.Parameter(torch.zeros(1))
            nn.init.zeros_(self.late_calib_router.weight)
            nn.init.zeros_(self.late_calib_router.bias)
        else:
            self.late_calib_router = None
            self.late_calib_alpha = None
        self.enable_fusion_gate = getattr(args, "enable_fusion_gate", False)
        self.fusion_gate_scale = max(0.0, float(getattr(args, "fusion_gate_scale", 0.1)))
        self.detach_fusion_text_source = self.enable_fusion_gate and not getattr(args, "disable_detach_fusion_text_source", False)
        if self.enable_fusion_gate and self.fusion_gate_scale > 0.0:
            self.fusion_gate = nn.Linear(D, D)
            self.fusion_shift = nn.Linear(2 * D_e, D)
            nn.init.zeros_(self.fusion_gate.weight)
            nn.init.zeros_(self.fusion_gate.bias)
            nn.init.zeros_(self.fusion_shift.weight)
            nn.init.zeros_(self.fusion_shift.bias)
        else:
            self.fusion_gate = None
            self.fusion_shift = None
        self.enable_conflict_head = getattr(args, "enable_conflict_head", False)
        self.conflict_head_scale = max(0.0, float(getattr(args, "conflict_head_scale", 0.1)))
        if self.enable_conflict_head and self.conflict_head_scale > 0.0:
            self.conflict_proj = nn.Linear(6 * D_e, D)
            self.conflict_gate = nn.Linear(2 * D, D)
            nn.init.zeros_(self.conflict_proj.weight)
            nn.init.zeros_(self.conflict_proj.bias)
            nn.init.zeros_(self.conflict_gate.weight)
            nn.init.zeros_(self.conflict_gate.bias)
        else:
            self.conflict_proj = None
            self.conflict_gate = None
        self.enable_shared_private_decomp = getattr(args, "enable_shared_private_decomp", False)
        self.latest_decomp_losses = None
        if self.enable_shared_private_decomp:
            def _make_proj_block(in_dim, out_dim):
                return nn.Sequential(
                    nn.Linear(in_dim, out_dim),
                    nn.GELU(),
                    nn.Dropout(drop_rate),
                    nn.Linear(out_dim, out_dim),
                )

            self.shared_proj_a = _make_proj_block(D_e, D_e)
            self.shared_proj_t = _make_proj_block(D_e, D_e)
            self.shared_proj_v = _make_proj_block(D_e, D_e)
            self.private_proj_a = _make_proj_block(D_e, D_e)
            self.private_proj_t = _make_proj_block(D_e, D_e)
            self.private_proj_v = _make_proj_block(D_e, D_e)
            self.recon_proj_a = _make_proj_block(2 * D_e, D_e)
            self.recon_proj_t = _make_proj_block(2 * D_e, D_e)
            self.recon_proj_v = _make_proj_block(2 * D_e, D_e)
        else:
            self.shared_proj_a = None
            self.shared_proj_t = None
            self.shared_proj_v = None
            self.private_proj_a = None
            self.private_proj_t = None
            self.private_proj_v = None
            self.recon_proj_a = None
            self.recon_proj_t = None
            self.recon_proj_v = None
        self.enable_intensity_contrastive = getattr(args, "enable_intensity_contrastive", False)
        self.contrastive_dim = max(2, int(getattr(args, "contrastive_dim", 32)))
        self.contrastive_label_sigma = max(1e-4, float(getattr(args, "contrastive_label_sigma", 0.75)))
        self.latest_contrastive_loss = None
        if self.enable_intensity_contrastive:
            self.contrastive_proj = nn.Sequential(
                nn.Linear(D, D),
                nn.GELU(),
                nn.Dropout(drop_rate),
                nn.Linear(D, self.contrastive_dim),
            )
        else:
            self.contrastive_proj = None
        self.enable_label_prototype = getattr(args, "enable_label_prototype", False)
        self.prototype_dim = max(2, int(getattr(args, "prototype_dim", 32)))
        self.prototype_bins = max(3, int(getattr(args, "prototype_bins", 7)))
        self.prototype_temperature = max(1e-4, float(getattr(args, "prototype_temperature", 0.5)))
        self.prototype_target_sigma = max(1e-4, float(getattr(args, "prototype_target_sigma", 0.75)))
        self.prototype_min = float(getattr(args, "prototype_min", -3.0))
        self.prototype_max = float(getattr(args, "prototype_max", 3.0))
        self.latest_prototype_losses = None
        if self.enable_label_prototype:
            self.prototype_proj = nn.Sequential(
                nn.Linear(D, D),
                nn.GELU(),
                nn.Dropout(drop_rate),
                nn.Linear(D, self.prototype_dim),
            )
            prototype_values = torch.linspace(self.prototype_min, self.prototype_max, steps=self.prototype_bins)
            self.register_buffer("prototype_values", prototype_values)
            init_prototypes = torch.zeros(self.prototype_bins, self.prototype_dim)
            init_prototypes[:, 0] = torch.linspace(-1.0, 1.0, steps=self.prototype_bins)
            self.sentiment_prototypes = nn.Parameter(init_prototypes)
        else:
            self.prototype_proj = None
            self.register_buffer("prototype_values", torch.empty(0))
            self.sentiment_prototypes = None

        self.router_a = Mlp(
            in_features=D_e,
            hidden_features=int(D_e * mlp_ratio),
            out_features=self.num_experts_total,
            drop=drop_rate,
        )
        self.router_t = Mlp(
            in_features=D_e,
            hidden_features=int(D_e * mlp_ratio),
            out_features=self.num_experts_total,
            drop=drop_rate,
        )
        self.router_v = Mlp(
            in_features=D_e,
            hidden_features=int(D_e * mlp_ratio),
            out_features=self.num_experts_total,
            drop=drop_rate,
        )

        self.use_cross_modal = getattr(args, "use_cross_modal", True)
        if self.use_cross_modal:
            cross_intermediate = max(int(D_e * mlp_ratio * 2), D_e)
            cross_config = BertConfig(
                hidden_size=D_e,
                num_hidden_layers=1,
                num_attention_heads=num_heads,
                intermediate_size=cross_intermediate,
                hidden_dropout_prob=drop_rate,
                attention_probs_dropout_prob=attn_drop_rate,
            )
            self.cross_at = CMELayer(cross_config)
            self.cross_vt = CMELayer(cross_config)
            # Directional cross-mix weights with backward-compatible defaults.
            def _clamp01(x):
                return float(torch.clamp(torch.tensor(x), 0.0, 1.0).item())

            def _make_gate(prior):
                gate = nn.Linear(2 * D_e, 1)
                nn.init.zeros_(gate.weight)
                prior = float(np.clip(prior, 1e-4, 1 - 1e-4))
                gate.bias.data.fill_(float(torch.logit(torch.tensor(prior)).item()))
                return gate

            def _make_delta_gate():
                gate = nn.Linear(2 * D_e, 1)
                nn.init.zeros_(gate.weight)
                nn.init.zeros_(gate.bias)
                return gate

            def _make_vector_gate():
                gate = nn.Linear(2 * D_e, D_e)
                nn.init.zeros_(gate.weight)
                nn.init.zeros_(gate.bias)
                return gate

            def _make_zero_proj():
                proj = nn.Linear(D_e, D_e)
                nn.init.zeros_(proj.weight)
                nn.init.zeros_(proj.bias)
                return proj

            base_mix = getattr(args, "cross_mix_weight", 0.7)
            w_ta = getattr(args, "cross_mix_ta", None)
            w_tv = getattr(args, "cross_mix_tv", None)
            w_at = getattr(args, "cross_mix_at", None)
            w_vt = getattr(args, "cross_mix_vt", None)
            self.cross_mix_t_from_a = _clamp01(base_mix if w_ta is None else w_ta)
            self.cross_mix_t_from_v = _clamp01(base_mix if w_tv is None else w_tv)
            self.cross_mix_a_from_t = _clamp01(base_mix if w_at is None else w_at)
            self.cross_mix_v_from_t = _clamp01(base_mix if w_vt is None else w_vt)
            self.enable_av_cross = getattr(args, "enable_av_cross", False)
            self.cross_mix_a_from_v = _clamp01(getattr(args, "cross_mix_av", 0.2))
            self.cross_mix_v_from_a = _clamp01(getattr(args, "cross_mix_va", 0.2))
            self.cross_av = CMELayer(cross_config) if self.enable_av_cross else None
            self.enable_text_stim = getattr(args, "enable_text_stim", False)
            self.text_stim_a_scale = max(0.0, float(getattr(args, "text_stim_a_scale", 0.0)))
            self.text_stim_v_scale = max(0.0, float(getattr(args, "text_stim_v_scale", 0.1)))
            self.detach_text_stim_source = self.enable_text_stim and not getattr(args, "disable_detach_text_stim_source", False)
            if self.enable_text_stim and (self.text_stim_a_scale > 0.0 or self.text_stim_v_scale > 0.0):
                self.text_stim_gate_a = _make_vector_gate() if self.text_stim_a_scale > 0.0 else None
                self.text_stim_gate_v = _make_vector_gate() if self.text_stim_v_scale > 0.0 else None
                self.text_stim_proj_a = _make_zero_proj() if self.text_stim_a_scale > 0.0 else None
                self.text_stim_proj_v = _make_zero_proj() if self.text_stim_v_scale > 0.0 else None
            else:
                self.text_stim_gate_a = None
                self.text_stim_gate_v = None
                self.text_stim_proj_a = None
                self.text_stim_proj_v = None
            self.use_clean_va_inputs = (
                self.enable_av_cross
                and self.cross_mix_a_from_v <= 1e-8
                and self.cross_mix_v_from_a > 0.0
                and not getattr(args, "disable_clean_va_path", False)
            )
            self.detach_va_source = self.use_clean_va_inputs and not getattr(args, "disable_detach_va_source", False)
            self.use_adaptive_va_gate = self.enable_av_cross and not getattr(args, "disable_adaptive_va_gate", False)
            self.va_gate_delta = max(0.0, float(getattr(args, "va_gate_delta", 0.03)))
            self.cross_gate_va = _make_delta_gate() if self.use_adaptive_va_gate else None
            self.use_adaptive_cross_gate = not getattr(args, "disable_adaptive_cross_gate", False)
            if self.use_adaptive_cross_gate:
                self.cross_gate_ta = _make_gate(self.cross_mix_t_from_a)
                self.cross_gate_tv = _make_gate(self.cross_mix_t_from_v)
                self.cross_gate_at = _make_gate(self.cross_mix_a_from_t)
                self.cross_gate_vt = _make_gate(self.cross_mix_v_from_t)
            else:
                self.cross_gate_ta = None
                self.cross_gate_tv = None
                self.cross_gate_at = None
                self.cross_gate_vt = None
            self.latest_cross_gate_stats = None
        else:
            self.cross_at = None
            self.cross_vt = None
            self.cross_mix_t_from_a = None
            self.cross_mix_t_from_v = None
            self.cross_mix_a_from_t = None
            self.cross_mix_v_from_t = None
            self.enable_av_cross = False
            self.cross_mix_a_from_v = None
            self.cross_mix_v_from_a = None
            self.cross_av = None
            self.enable_text_stim = False
            self.text_stim_a_scale = 0.0
            self.text_stim_v_scale = 0.0
            self.detach_text_stim_source = False
            self.text_stim_gate_a = None
            self.text_stim_gate_v = None
            self.text_stim_proj_a = None
            self.text_stim_proj_v = None
            self.use_clean_va_inputs = False
            self.detach_va_source = False
            self.use_adaptive_va_gate = False
            self.va_gate_delta = 0.0
            self.cross_gate_va = None
            self.use_adaptive_cross_gate = False
            self.cross_gate_ta = None
            self.cross_gate_tv = None
            self.cross_gate_at = None
            self.cross_gate_vt = None
            self.latest_cross_gate_stats = None

    @staticmethod
    def _masked_average(x, mask):
        weight = mask.unsqueeze(-1).float()
        denom = weight.sum(dim=1).clamp_min(1.0)
        return (x * weight).sum(dim=1) / denom

    @staticmethod
    def _masked_feature_mse(pred, target, mask):
        weight = mask.unsqueeze(-1).float()
        denom = weight.sum() * pred.size(-1)
        return ((pred - target) ** 2 * weight).sum() / denom.clamp_min(1.0)

    @staticmethod
    def _masked_orthogonality(shared, private, mask):
        shared_norm = F.normalize(shared, dim=-1)
        private_norm = F.normalize(private, dim=-1)
        score = (shared_norm * private_norm).sum(dim=-1).pow(2)
        weight = mask.float()
        return (score * weight).sum() / weight.sum().clamp_min(1.0)

    def _compute_decomp_losses(self, x_out_a, x_out_t, x_out_v, modality_masks):
        shared_a = self.shared_proj_a(x_out_a)
        shared_t = self.shared_proj_t(x_out_t)
        shared_v = self.shared_proj_v(x_out_v)
        private_a = self.private_proj_a(x_out_a)
        private_t = self.private_proj_t(x_out_t)
        private_v = self.private_proj_v(x_out_v)

        recon_a = self.recon_proj_a(torch.cat([shared_a, private_a], dim=-1))
        recon_t = self.recon_proj_t(torch.cat([shared_t, private_t], dim=-1))
        recon_v = self.recon_proj_v(torch.cat([shared_v, private_v], dim=-1))

        audio_mask = modality_masks['a'].float()
        text_mask = modality_masks['t'].float()
        video_mask = modality_masks['v'].float()

        recon_loss = (
            self._masked_feature_mse(recon_a, x_out_a, audio_mask)
            + self._masked_feature_mse(recon_t, x_out_t, text_mask)
            + self._masked_feature_mse(recon_v, x_out_v, video_mask)
        ) / 3.0

        orth_loss = (
            self._masked_orthogonality(shared_a, private_a, audio_mask)
            + self._masked_orthogonality(shared_t, private_t, text_mask)
            + self._masked_orthogonality(shared_v, private_v, video_mask)
        ) / 3.0

        pooled_shared_a = self._masked_average(shared_a, audio_mask)
        pooled_shared_t = self._masked_average(shared_t, text_mask)
        pooled_shared_v = self._masked_average(shared_v, video_mask)
        align_loss = (
            (1.0 - F.cosine_similarity(pooled_shared_a, pooled_shared_t, dim=-1)).mean()
            + (1.0 - F.cosine_similarity(pooled_shared_t, pooled_shared_v, dim=-1)).mean()
            + (1.0 - F.cosine_similarity(pooled_shared_a, pooled_shared_v, dim=-1)).mean()
        ) / 3.0

        return {
            'recon': recon_loss,
            'orth': orth_loss,
            'align': align_loss,
        }

    def compute_label_prototype_losses(self, hidden, labels, umask):
        if not self.enable_label_prototype or self.prototype_proj is None or self.sentiment_prototypes is None:
            self.latest_prototype_losses = None
            return None

        mask = umask.float()
        if mask.sum() <= 0:
            self.latest_prototype_losses = None
            return None

        proto_hidden = F.normalize(self.prototype_proj(hidden), dim=-1)
        prototypes = F.normalize(self.sentiment_prototypes, dim=-1)
        logits = torch.einsum("bsd,kd->bsk", proto_hidden, prototypes) / self.prototype_temperature
        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()

        anchor_values = self.prototype_values.view(1, 1, -1)
        clipped_labels = labels.float().clamp(self.prototype_min, self.prototype_max).unsqueeze(-1)
        target_dist = torch.exp(-0.5 * ((anchor_values - clipped_labels) / self.prototype_target_sigma) ** 2)
        target_dist = target_dist / target_dist.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        kl = (target_dist * (torch.log(target_dist.clamp_min(1e-8)) - log_probs)).sum(dim=-1)
        kl = (kl * mask).sum() / mask.sum().clamp_min(1.0)

        proto_pred = (probs * anchor_values).sum(dim=-1)
        reg = (((proto_pred - labels.float()) ** 2) * mask).sum() / mask.sum().clamp_min(1.0)

        target_proto = torch.einsum("bsk,kd->bsd", target_dist, prototypes)
        align = 1.0 - F.cosine_similarity(proto_hidden, F.normalize(target_proto, dim=-1), dim=-1)
        align = (align * mask).sum() / mask.sum().clamp_min(1.0)

        self.latest_prototype_losses = {
            'kl': kl,
            'reg': reg,
            'align': align,
            'pred_mean': (proto_pred * mask).sum() / mask.sum().clamp_min(1.0),
        }
        return self.latest_prototype_losses

    def compute_intensity_contrastive_loss(self, hidden, labels, umask):
        if not self.enable_intensity_contrastive or self.contrastive_proj is None:
            self.latest_contrastive_loss = None
            return None

        valid_mask = umask.bool()
        if valid_mask.sum() < 2:
            self.latest_contrastive_loss = None
            return None

        features = hidden[valid_mask]
        scores = labels.float()[valid_mask]
        if features.size(0) < 2:
            self.latest_contrastive_loss = None
            return None

        proj = F.normalize(self.contrastive_proj(features), dim=-1)
        cosine = torch.matmul(proj, proj.transpose(0, 1))
        pred_sim = (cosine + 1.0) * 0.5

        score_diff = scores.unsqueeze(1) - scores.unsqueeze(0)
        target_sim = torch.exp(-0.5 * (score_diff / self.contrastive_label_sigma).pow(2))

        pair_mask = ~torch.eye(features.size(0), device=features.device, dtype=torch.bool)
        if pair_mask.sum() <= 0:
            self.latest_contrastive_loss = None
            return None

        loss = F.mse_loss(pred_sim[pair_mask], target_sim[pair_mask])
        self.latest_contrastive_loss = loss
        return loss

    def forward(self, inputfeats, input_features_mask=None, umask=None, first_stage=False):
        """
        inputfeats -> ?*[seqlen, batch, dim]
        qmask -> [batch, seqlen]
        umask -> [batch, seqlen]
        seq_lengths -> each conversation lens
        input_features_mask -> ?*[seqlen, batch, 3]
        """
        # print(inputfeats[:,:,:])
        # print(input_features_mask[:,:,1])
        weight_save = []
        # sequence modeling
        audio, text, video = inputfeats[:, :, :self.adim], inputfeats[:, :, self.adim:self.adim + self.tdim], \
        inputfeats[:, :, self.adim + self.tdim:]
        seq_len, B, C = audio.shape

        # --> [batch, seqlen, dim]
        audio, text, video = audio.permute(1, 0, 2), text.permute(1, 0, 2), video.permute(1, 0, 2)
        proj_a = self.dropout_a(self.a_in_proj(audio))
        proj_t = self.dropout_t(self.t_in_proj(text))
        proj_v = self.dropout_v(self.v_in_proj(video))

        # --> [batch, seqlen, 3]
        input_mask = torch.clone(input_features_mask.permute(1, 0, 2))
        input_mask[umask == 0] = 0
        modality_masks = {
            'a': input_mask[:, :, 0],
            't': input_mask[:, :, 1],
            'v': input_mask[:, :, 2],
        }
        # --> [batch, 3, seqlen] -> [batch, 3*seqlen]
        attn_mask = input_mask.transpose(1, 2).reshape(B, -1)

        self.latest_cross_gate_stats = None
        self.latest_router_stats = None
        self.latest_decomp_losses = None
        self.latest_contrastive_loss = None
        self.latest_prototype_losses = None
        if self.use_layerwise_sparse_router:
            self.block.latest_router_stats = {}
            weight_a = weight_t = weight_v = None
        else:
            # Legacy global dense router.
            weight_a, weight_t, weight_v = self.router_a(proj_a), self.router_t(proj_t), self.router_v(proj_v)
            weight_a = torch.softmax(weight_a, dim=-1)
            weight_t = torch.softmax(weight_t, dim=-1)
            weight_v = torch.softmax(weight_v, dim=-1)
            # weight_save.append(np.array([weight_a.cpu().detach().numpy(), weight_t.cpu().detach().numpy(), weight_v.cpu().detach().numpy()]))
            # reshape weights later to align with experts

        # --> [batch, 3*seqlen, dim]
        x_a = self.block(proj_a, first_stage, attn_mask, 'a')
        x_t = self.block(proj_t, first_stage, attn_mask, 't')
        x_v = self.block(proj_v, first_stage, attn_mask, 'v')
        if first_stage:
            x_a = self.post_dropout_a(x_a)
            x_t = self.post_dropout_t(x_t)
            x_v = self.post_dropout_v(x_v)
            out_a = self.nlp_head_a(x_a)
            out_t = self.nlp_head_t(x_t)
            out_v = self.nlp_head_v(x_v)
            x = torch.cat([x_a, x_t, x_v], dim=1)
        else:
            if self.use_layerwise_sparse_router:
                x_out_a, x_out_t, x_out_v = x_a, x_t, x_v
                if getattr(self.block, "latest_router_stats", None):
                    self.latest_router_stats = dict(self.block.latest_router_stats)
            else:
                x_unweighted_a = x_a.reshape(B, seq_len, 3, self.num_experts, self.D_e).reshape(B, seq_len, self.num_experts_total, self.D_e)
                x_unweighted_t = x_t.reshape(B, seq_len, 3, self.num_experts, self.D_e).reshape(B, seq_len, self.num_experts_total, self.D_e)
                x_unweighted_v = x_v.reshape(B, seq_len, 3, self.num_experts, self.D_e).reshape(B, seq_len, self.num_experts_total, self.D_e)
                x_out_a = torch.sum(weight_a.unsqueeze(-1) * x_unweighted_a, dim=2)
                x_out_t = torch.sum(weight_t.unsqueeze(-1) * x_unweighted_t, dim=2)
                x_out_v = torch.sum(weight_v.unsqueeze(-1) * x_unweighted_v, dim=2)
            x_out_a_base = x_out_a
            x_out_v_base = x_out_v
            out_a = self.nlp_head_a(x_out_a)
            out_t = self.nlp_head_t(x_out_t)
            out_v = self.nlp_head_v(x_out_v)

            if self.enable_shared_private_decomp:
                self.latest_decomp_losses = self._compute_decomp_losses(
                    x_out_a, x_out_t, x_out_v, modality_masks
                )

            if self.use_cross_modal:
                text_mask = modality_masks['t'].float()
                audio_mask = modality_masks['a'].float()
                video_mask = modality_masks['v'].float()
                # Directional mixing: t<-a (ta), t<-v (tv), a<-t (at), v<-t (vt)
                w_ta = self.cross_mix_t_from_a if self.cross_mix_t_from_a is not None else 0.5
                w_tv = self.cross_mix_t_from_v if self.cross_mix_t_from_v is not None else 0.5
                w_at = self.cross_mix_a_from_t if self.cross_mix_a_from_t is not None else 0.5
                w_vt = self.cross_mix_v_from_t if self.cross_mix_v_from_t is not None else 0.5

                ta_pair_mask = (text_mask * audio_mask).unsqueeze(-1)
                tv_pair_mask = (text_mask * video_mask).unsqueeze(-1)

                if self.enable_text_stim and (self.text_stim_gate_a is not None or self.text_stim_gate_v is not None):
                    stim_text = x_out_t.detach() if self.detach_text_stim_source else x_out_t
                    if self.text_stim_gate_a is not None:
                        g_sa = torch.sigmoid(self.text_stim_gate_a(torch.cat([x_out_a, stim_text], dim=-1))) * ta_pair_mask
                        stim_a = torch.tanh(self.text_stim_proj_a(stim_text))
                        x_out_a = x_out_a + self.text_stim_a_scale * g_sa * stim_a
                    if self.text_stim_gate_v is not None:
                        g_sv = torch.sigmoid(self.text_stim_gate_v(torch.cat([x_out_v, stim_text], dim=-1))) * tv_pair_mask
                        stim_v = torch.tanh(self.text_stim_proj_v(stim_text))
                        x_out_v = x_out_v + self.text_stim_v_scale * g_sv * stim_v

                cross_t_from_a, cross_a_from_t = self.cross_at(
                    x_out_t, text_mask, x_out_a, audio_mask
                )
                cross_t_from_v, cross_v_from_t = self.cross_vt(
                    x_out_t, text_mask, x_out_v, video_mask
                )

                if self.use_adaptive_cross_gate:
                    g_ta = torch.sigmoid(self.cross_gate_ta(torch.cat([x_out_t, cross_t_from_a], dim=-1))) * ta_pair_mask
                    g_tv = torch.sigmoid(self.cross_gate_tv(torch.cat([x_out_t, cross_t_from_v], dim=-1))) * tv_pair_mask
                    g_at = torch.sigmoid(self.cross_gate_at(torch.cat([x_out_a, cross_a_from_t], dim=-1))) * ta_pair_mask
                    g_vt = torch.sigmoid(self.cross_gate_vt(torch.cat([x_out_v, cross_v_from_t], dim=-1))) * tv_pair_mask
                else:
                    g_ta = ta_pair_mask * w_ta
                    g_tv = tv_pair_mask * w_tv
                    g_at = ta_pair_mask * w_at
                    g_vt = tv_pair_mask * w_vt

                self.latest_cross_gate_stats = {
                    'ta': float(g_ta.detach().mean().cpu().item()),
                    'tv': float(g_tv.detach().mean().cpu().item()),
                    'at': float(g_at.detach().mean().cpu().item()),
                    'vt': float(g_vt.detach().mean().cpu().item()),
                }
                if self.enable_text_stim and self.text_stim_gate_a is not None:
                    self.latest_cross_gate_stats['sa'] = float((self.text_stim_a_scale * g_sa).detach().mean().cpu().item())
                if self.enable_text_stim and self.text_stim_gate_v is not None:
                    self.latest_cross_gate_stats['sv'] = float((self.text_stim_v_scale * g_sv).detach().mean().cpu().item())

                # Use the maximum of incoming weights to rescale the residual for t,
                # so defaults (all 0.7) reproduce previous behavior exactly.
                rescale_t = torch.maximum(g_ta, g_tv)
                x_out_t = (1 - rescale_t) * x_out_t + 0.5 * (g_ta * cross_t_from_a + g_tv * cross_t_from_v)
                x_out_a = (1 - g_at) * x_out_a + g_at * cross_a_from_t
                x_out_v = (1 - g_vt) * x_out_v + g_vt * cross_v_from_t

                if self.enable_av_cross:
                    av_pair_mask = (audio_mask * video_mask).unsqueeze(-1)
                    av_audio_state = x_out_a_base if self.use_clean_va_inputs else x_out_a
                    av_video_state = x_out_v_base if self.use_clean_va_inputs else x_out_v
                    if self.detach_va_source:
                        av_audio_state = av_audio_state.detach()
                    cross_a_from_v, cross_v_from_a = self.cross_av(
                        av_audio_state, audio_mask, av_video_state, video_mask
                    )
                    w_av = av_pair_mask * self.cross_mix_a_from_v
                    if self.use_adaptive_va_gate:
                        va_delta = self.va_gate_delta * torch.tanh(
                            self.cross_gate_va(torch.cat([x_out_v, cross_v_from_a], dim=-1))
                        )
                        va_weight = torch.clamp(self.cross_mix_v_from_a + va_delta, 0.0, 1.0)
                        w_va = av_pair_mask * va_weight
                    else:
                        w_va = av_pair_mask * self.cross_mix_v_from_a
                    self.latest_cross_gate_stats.update({
                        'av': float(w_av.detach().mean().cpu().item()),
                        'va': float(w_va.detach().mean().cpu().item()),
                    })
                    x_out_a = (1 - w_av) * x_out_a + w_av * cross_a_from_v
                    x_out_v = (1 - w_va) * x_out_v + w_va * cross_v_from_a

            x = torch.cat([x_out_a, x_out_t, x_out_v], dim=1)

        x[attn_mask == 0] = 0

        x_a, x_t, x_v = x[:, :seq_len, :], x[:, seq_len:2*seq_len, :], x[:, 2*seq_len:, :]
        x_joint = torch.cat([x_a, x_t, x_v], dim=-1)
        if self.enable_conflict_head and self.conflict_proj is not None and self.conflict_gate is not None:
            diff_ta = torch.abs(x_t - x_a)
            diff_tv = torch.abs(x_t - x_v)
            diff_av = torch.abs(x_a - x_v)
            prod_ta = x_t * x_a
            prod_tv = x_t * x_v
            prod_av = x_a * x_v
            conflict_feat = torch.cat([diff_ta, diff_tv, diff_av, prod_ta, prod_tv, prod_av], dim=-1)
            conflict_shift = torch.tanh(self.conflict_proj(conflict_feat))
            conflict_gate = torch.sigmoid(self.conflict_gate(torch.cat([x_joint, conflict_shift], dim=-1)))
            x_joint = x_joint + self.conflict_head_scale * conflict_gate * conflict_shift
            if self.latest_cross_gate_stats is None:
                self.latest_cross_gate_stats = {}
            self.latest_cross_gate_stats['cf'] = float((self.conflict_head_scale * conflict_gate).detach().mean().cpu().item())
        if self.enable_fusion_gate and self.fusion_gate is not None and self.fusion_shift is not None:
            fusion_text = x_t.detach() if self.detach_fusion_text_source else x_t
            fusion_aux = torch.cat([x_a, x_v], dim=-1)
            fusion_gate = torch.sigmoid(self.fusion_gate(torch.cat([fusion_text, fusion_aux], dim=-1)))
            fusion_shift = torch.tanh(self.fusion_shift(fusion_aux))
            x_joint = x_joint + self.fusion_gate_scale * fusion_gate * fusion_shift
            if self.latest_cross_gate_stats is None:
                self.latest_cross_gate_stats = {}
            self.latest_cross_gate_stats['fg'] = float((self.fusion_gate_scale * fusion_gate).detach().mean().cpu().item())
        res = x_joint
        u = F.relu(self.proj1(x_joint))
        u = F.dropout(u, p=self.out_dropout, training=self.training)
        hidden = u + res
        out = self.nlp_head(hidden)
        if (not first_stage) and self.enable_late_unimodal_calibration and self.late_calib_router is not None:
            late_weights = torch.softmax(self.late_calib_router(hidden), dim=-1)
            unimodal_preds = torch.cat([out_a, out_t, out_v], dim=-1)
            late_residual = (late_weights * unimodal_preds).sum(dim=-1, keepdim=True)
            late_alpha = self.late_calib_scale * torch.tanh(self.late_calib_alpha)
            out = out + late_alpha * late_residual
            if self.latest_cross_gate_stats is None:
                self.latest_cross_gate_stats = {}
            self.latest_cross_gate_stats['lc'] = float(late_alpha.detach().cpu().item())

        return hidden, out, out_a, out_t, out_v, np.array(weight_save)

if __name__ == '__main__':
    input = [torch.randn(61, 32, 300)]
    model = MoMKE(100, 100, 100, 128, 1)
    anchor = torch.randn(32, 61, 128)
    hidden, out, _ = model(input)
