import time
import datetime
import random
import argparse
import math
import numpy as np
import torch
import torch.optim as optim
from sklearn.metrics import f1_score, accuracy_score, recall_score
import sys
from utils import Logger, get_loaders, build_model, generate_inputs, generate_mask_tensor
from loss import MaskedCELoss, MaskedMSELoss, MaskedBCEWithLogitsLoss
from training_utils import (
    apply_stage2_modality_dropout,
    compute_monitor_value,
    resolve_cpu_thread_settings,
)
import os
import warnings
sys.path.append('./')
warnings.filterwarnings("ignore")
import config


def get_cross_mix_weight(model):
    if hasattr(model, "latest_cross_gate_stats") and model.latest_cross_gate_stats:
        gate_stats = model.latest_cross_gate_stats
        order = ["ta", "tv", "at", "vt", "sa", "sv", "av", "va", "cf", "fg", "lc"]
        parts = [f"{name}={gate_stats[name]:.3f}" for name in order if name in gate_stats]
        if parts:
            return ",".join(parts)
    # New directional weights formatting if available
    if hasattr(model, "use_cross_modal") and getattr(model, "use_cross_modal"):
        names = [
            ("ta", "cross_mix_t_from_a"),
            ("tv", "cross_mix_t_from_v"),
            ("at", "cross_mix_a_from_t"),
            ("vt", "cross_mix_v_from_t"),
            ("av", "cross_mix_a_from_v"),
            ("va", "cross_mix_v_from_a"),
        ]
        if all(hasattr(model, attr_name) for _, attr_name in names):
            vals = []
            for short_name, attr_name in names:
                v = getattr(model, attr_name)
                if short_name in ("av", "va") and not getattr(model, "enable_av_cross", False):
                    continue
                if v is None:
                    continue
                if isinstance(v, torch.Tensor):
                    v = v.detach().cpu().item()
                vals.append(f"{short_name}={float(v):.3f}")
            if vals:
                return ",".join(vals)
        # Backward compatibility for the single scalar weight
        if hasattr(model, "cross_mix_weight") and model.cross_mix_weight is not None:
            value = model.cross_mix_weight
            if isinstance(value, torch.Tensor):
                value = torch.sigmoid(value).detach().cpu().item()
            return f"{float(value):.3f}"
    return None


def get_router_summary(model):
    stats = getattr(model, "latest_router_stats", None)
    if not stats:
        return None
    parts = []
    for name in ["a", "t", "v"]:
        if name not in stats:
            continue
        item = stats[name]
        parts.append(f"{name}:k={item['active']:.2f},h={item['entropy']:.2f}")
    return "; ".join(parts) if parts else None


def get_decomp_summary(model):
    losses = getattr(model, "latest_decomp_losses", None)
    if not losses:
        return None
    return "recon={recon:.4f},orth={orth:.4f},align={align:.4f}".format(
        recon=float(losses["recon"].detach().cpu().item()),
        orth=float(losses["orth"].detach().cpu().item()),
        align=float(losses["align"].detach().cpu().item()),
    )


def get_prototype_summary(model):
    losses = getattr(model, "latest_prototype_losses", None)
    if not losses:
        return None
    return "kl={kl:.4f},reg={reg:.4f},align={align:.4f},pm={pm:.3f}".format(
        kl=float(losses["kl"].detach().cpu().item()),
        reg=float(losses["reg"].detach().cpu().item()),
        align=float(losses["align"].detach().cpu().item()),
        pm=float(losses["pred_mean"].detach().cpu().item()),
    )


def get_contrastive_summary(model):
    loss = getattr(model, "latest_contrastive_loss", None)
    if loss is None:
        return None
    return f"icl={float(loss.detach().cpu().item()):.4f}"


def snapshot_model_state(model):
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def average_model_states(states):
    if not states:
        raise ValueError("states must not be empty")
    avg_state = {}
    ref_keys = states[0].keys()
    for key in ref_keys:
        tensors = [state[key].float() for state in states]
        avg_state[key] = torch.stack(tensors, dim=0).mean(dim=0)
    return avg_state


def build_optimizer(model, lr, l2):
    return optim.Adam([{'params': model.parameters(), 'lr': lr, 'weight_decay': l2}])


def set_optimizer_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def scale_optimizer_state(optimizer, scale):
    if scale == 1.0:
        return
    for state in optimizer.state.values():
        for key in ('exp_avg', 'exp_avg_sq', 'max_exp_avg_sq'):
            tensor = state.get(key)
            if torch.is_tensor(tensor):
                tensor.mul_(scale)


def get_epoch_lr(args, epoch):
    stage_epoch = int(args.stage_epoch)
    if epoch < stage_epoch:
        return args.lr

    base_lr = args.stage2_lr if args.stage2_lr is not None else args.lr
    if not getattr(args, "enable_stage2_cosine_lr", False):
        return base_lr

    total_stage2_epochs = max(1, int(args.epochs - stage_epoch))
    if total_stage2_epochs <= 1:
        return base_lr

    stage2_epoch_idx = min(max(epoch - stage_epoch, 0), total_stage2_epochs - 1)
    progress = stage2_epoch_idx / float(total_stage2_epochs - 1)
    min_lr = base_lr * args.stage2_min_lr_ratio
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def load_branch_states(model, branch_states):
    model_state = model.state_dict()
    for branch_name, branch_state in branch_states.items():
        branch_key = f"transformer_{branch_name}"
        router_key = f"router_{branch_name}"
        matched = {
            key: value
            for key, value in branch_state.items()
            if branch_key in key or router_key in key
        }
        if not matched:
            raise RuntimeError(f"No parameters found for branch '{branch_key}'")
        model_state.update(matched)
    model.load_state_dict(model_state)


def train_or_eval_model(args, model, reg_loss, cls_loss, polarity_loss, dataloader, optimizer=None, train=False, first_stage=True, mark='train'):
    collect_analysis = getattr(args, "collect_analysis", False)
    weight = [] if collect_analysis else None
    collect_unimodal = first_stage
    preds, preds_a, preds_t, preds_v, masks, labels = [], [], [], [], [], []
    losses, losses1, losses2, losses3 = [], [], [], []
    dataset = args.dataset
    cuda = torch.cuda.is_available() and not args.no_cuda

    assert not train or optimizer!=None
    if train:
        model.train()
    else:
        model.eval()

    for data_idx, data in enumerate(dataloader):
        vidnames = []
        if train: optimizer.zero_grad()
        
        ## read dataloader and generate all missing conditions
        """
        audio_host, text_host, visual_host: [seqlen, batch, dim]
        audio_guest, text_guest, visual_guest: [seqlen, batch, dim]
        qmask: speakers, [batch, seqlen]
        umask: has utt, [batch, seqlen]
        label: [batch, seqlen]
        """
        audio_host, text_host, visual_host = data[0], data[1], data[2]
        audio_guest, text_guest, visual_guest = data[3], data[4], data[5]
        qmask, umask, label = data[6], data[7], data[8]
        vidnames += data[-1]
        seqlen = audio_host.size(0)
        batch = audio_host.size(1)

        if cuda:
            audio_host = audio_host.to(device, non_blocking=True)
            text_host = text_host.to(device, non_blocking=True)
            visual_host = visual_host.to(device, non_blocking=True)
            audio_guest = audio_guest.to(device, non_blocking=True)
            text_guest = text_guest.to(device, non_blocking=True)
            visual_guest = visual_guest.to(device, non_blocking=True)
            qmask = qmask.to(device, non_blocking=True)
            umask = umask.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)

        input_features_mask_tensor = generate_mask_tensor(
            seqlen,
            batch,
            args.test_condition,
            first_stage,
            device=audio_host.device,
            dtype=audio_host.dtype,
        )
        input_features_mask_tensor = apply_stage2_modality_dropout(
            base_mask=input_features_mask_tensor,
            umask=umask,
            train=train,
            first_stage=first_stage,
            enable_dropout=getattr(args, "enable_stage2_modality_dropout", False),
            audio_drop_prob=getattr(args, "audio_modality_dropout", 0.0),
            text_drop_prob=getattr(args, "text_modality_dropout", 0.0),
            visual_drop_prob=getattr(args, "visual_modality_dropout", 0.0),
        )
        audio_host_mask = input_features_mask_tensor[:, :, 0:1]
        text_host_mask = input_features_mask_tensor[:, :, 1:2]
        visual_host_mask = input_features_mask_tensor[:, :, 2:3]
        audio_guest_mask = audio_host_mask
        text_guest_mask = text_host_mask
        visual_guest_mask = visual_host_mask

        masked_audio_host = audio_host * audio_host_mask
        masked_audio_guest = audio_guest * audio_guest_mask
        masked_text_host = text_host * text_host_mask
        masked_text_guest = text_guest * text_guest_mask
        masked_visual_host = visual_host * visual_host_mask
        masked_visual_guest = visual_guest * visual_guest_mask

        ## generate mask_input_features: ? * [seqlen, batch, dim], input_features_mask: ? * [seq_len, batch, 3]
        masked_input_features = generate_inputs(masked_audio_host, masked_text_host, masked_visual_host, \
                                                masked_audio_guest, masked_text_guest, masked_visual_guest, qmask)
        input_features_mask = [input_features_mask_tensor]
        '''
        # masked_input_features, input_features_mask: ?*[seqlen, batch, dim]
        # qmask: speakers, [batch, seqlen]
        # umask: has utt, [batch, seqlen]
        # label: [batch, seqlen]
        # log_prob: [seqlen, batch, num_classes]
        '''
        ## forward
        hidden, out, out_a, out_t, out_v, weight_save = model(masked_input_features[0], input_features_mask[0], umask, first_stage)

        ## save analysis result
        if collect_analysis:
            weight.append(weight_save)
            in_mask = torch.clone(input_features_mask[0].permute(1, 0, 2))
            in_mask[umask == 0] = 0
            weight.append(np.array(in_mask.detach().cpu()))
            weight.append(label.detach().cpu().numpy())
            weight.append(vidnames)

        ## calculate loss
        lp_ = out.view(-1, out.size(2)) # [batch*seq_len, n_classes]
        lp_a, lp_t, lp_v = out_a.view(-1, out_a.size(2)), out_t.view(-1, out_t.size(2)), out_v.view(-1, out_v.size(2))
        labels_ = label.view(-1) # [batch*seq_len]
        valid_count = umask.sum().item()

        if dataset in ['IEMOCAPFour', 'IEMOCAPSix']:
            if first_stage:
                loss_a = cls_loss(lp_a, labels_, umask)
                loss_t = cls_loss(lp_t, labels_, umask)
                loss_v = cls_loss(lp_v, labels_, umask)
                batch_loss = (loss_a + loss_t + loss_v) / 3.0
                losses.append(batch_loss.item() * valid_count)
                losses1.append(loss_a.item())
                losses2.append(loss_t.item())
                losses3.append(loss_v.item())
            else:
                loss = cls_loss(lp_, labels_, umask)
                if getattr(args, "enable_shared_private_decomp", False) and getattr(model, "latest_decomp_losses", None):
                    losses_map = model.latest_decomp_losses
                    loss = loss \
                        + args.decomp_recon_weight * losses_map['recon'] \
                        + args.decomp_orth_weight * losses_map['orth'] \
                        + args.decomp_align_weight * losses_map['align']
                losses.append(loss.item() * valid_count)
        if dataset in ['CMUMOSI', 'CMUMOSEI']:
            if first_stage:
                loss_a = reg_loss(lp_a, labels_, umask)
                loss_t = reg_loss(lp_t, labels_, umask)
                loss_v = reg_loss(lp_v, labels_, umask)
                batch_loss = (loss_a + loss_t + loss_v) / 3.0
                losses.append(batch_loss.item() * valid_count)
                losses1.append(loss_a.item())
                losses2.append(loss_t.item())
                losses3.append(loss_v.item())
            else:
                loss = reg_loss(lp_, labels_, umask)
                if getattr(args, "enable_polarity_loss", False):
                    loss = loss + args.polarity_loss_weight * polarity_loss(lp_.squeeze(-1), labels_, umask)
                if getattr(args, "enable_shared_private_decomp", False) and getattr(model, "latest_decomp_losses", None):
                    losses_map = model.latest_decomp_losses
                    loss = loss \
                        + args.decomp_recon_weight * losses_map['recon'] \
                        + args.decomp_orth_weight * losses_map['orth'] \
                        + args.decomp_align_weight * losses_map['align']
                if getattr(args, "enable_label_prototype", False):
                    proto_losses = model.compute_label_prototype_losses(hidden, label, umask)
                    if proto_losses:
                        loss = loss \
                            + args.prototype_kl_weight * proto_losses['kl'] \
                            + args.prototype_reg_weight * proto_losses['reg'] \
                            + args.prototype_align_weight * proto_losses['align']
                if getattr(args, "enable_intensity_contrastive", False):
                    contrastive_loss = model.compute_intensity_contrastive_loss(hidden, label, umask)
                    if contrastive_loss is not None:
                        loss = loss + args.intensity_contrastive_weight * contrastive_loss
                losses.append(loss.item() * valid_count)

        ## save batch results
        if collect_unimodal:
            preds_a.append(lp_a.data.cpu().numpy())
            preds_t.append(lp_t.data.cpu().numpy())
            preds_v.append(lp_v.data.cpu().numpy())
        preds.append(lp_.data.cpu().numpy())
        labels.append(labels_.data.cpu().numpy())
        masks.append(umask.view(-1).cpu().numpy())

        if train and first_stage:
            loss_a.backward()
            loss_t.backward()
            loss_v.backward()
            optimizer.step()
        if train and not first_stage:
            loss.backward()
            optimizer.step()

    assert preds!=[], f'Error: no dataset in dataloader'
    preds  = np.concatenate(preds)
    if collect_unimodal:
        preds_a = np.concatenate(preds_a)
        preds_t = np.concatenate(preds_t)
        preds_v = np.concatenate(preds_v)
    else:
        preds_a = preds_t = preds_v = None
    labels = np.concatenate(labels)
    masks  = np.concatenate(masks)

    # all
    if dataset in ['IEMOCAPFour', 'IEMOCAPSix']:
        preds = np.argmax(preds, 1)
        preds_a = np.argmax(preds_a, 1)
        preds_t = np.argmax(preds_t, 1)
        preds_v = np.argmax(preds_v, 1)
        total_mask = np.sum(masks)
        avg_loss = round(np.sum(losses)/total_mask, 4) if total_mask > 0 else 0.0
        avg_accuracy = accuracy_score(labels, preds, sample_weight=masks)
        avg_fscore = f1_score(labels, preds, sample_weight=masks, average='weighted')
        mae = 0
        ua = recall_score(labels, preds, sample_weight=masks, average='macro')
        if collect_unimodal:
            avg_acc_a = accuracy_score(labels, preds_a, sample_weight=masks)
            avg_acc_t = accuracy_score(labels, preds_t, sample_weight=masks)
            avg_acc_v = accuracy_score(labels, preds_v, sample_weight=masks)
        else:
            avg_acc_a = avg_acc_t = avg_acc_v = 0.0
        return mae, ua, avg_accuracy, avg_fscore, [avg_acc_a, avg_acc_t, avg_acc_v], vidnames, avg_loss, weight

    elif dataset in ['CMUMOSI', 'CMUMOSEI']:
        non_zeros = np.array([i for i, e in enumerate(labels) if e != 0]) # remove 0, and remove mask
        total_mask = np.sum(masks)
        avg_loss = round(np.sum(losses)/total_mask, 4) if total_mask > 0 else 0.0
        avg_accuracy = accuracy_score((labels[non_zeros] > 0), (preds[non_zeros] > 0))
        avg_fscore = f1_score((labels[non_zeros] > 0), (preds[non_zeros] > 0), average='weighted')
        mae = np.mean(np.absolute(labels[non_zeros] - preds[non_zeros].squeeze()))
        corr = np.corrcoef(labels[non_zeros], preds[non_zeros].squeeze())[0][1]
        if collect_unimodal:
            avg_acc_a = accuracy_score((labels[non_zeros] > 0), (preds_a[non_zeros] > 0))
            avg_acc_t = accuracy_score((labels[non_zeros] > 0), (preds_t[non_zeros] > 0))
            avg_acc_v = accuracy_score((labels[non_zeros] > 0), (preds_v[non_zeros] > 0))
        else:
            avg_acc_a = avg_acc_t = avg_acc_v = 0.0
        return mae, corr, avg_accuracy, avg_fscore, [avg_acc_a, avg_acc_t, avg_acc_v], vidnames, avg_loss, weight




if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    ## Params for input
    parser.add_argument('--audio-feature', type=str, default=None, help='audio feature name')
    parser.add_argument('--text-feature', type=str, default=None, help='text feature name')
    parser.add_argument('--video-feature', type=str, default=None, help='video feature name')
    parser.add_argument('--dataset', type=str, default='IEMOCAPFour', help='dataset type')

    ## Params for model
    parser.add_argument('--time-attn', action='store_true', default=False, help='whether to use nodal attention in graph model: Equation 4,5,6 in Paper')
    parser.add_argument('--depth', type=int, default=4, help='')
    parser.add_argument('--num_heads', type=int, default=2, help='')
    parser.add_argument('--drop_rate', type=float, default=0.0, help='')
    parser.add_argument('--attn_drop_rate', type=float, default=0.0, help='')
    parser.add_argument('--audio-drop-rate', type=float, default=None, help='dropout applied to projected audio states')
    parser.add_argument('--text-drop-rate', type=float, default=None, help='dropout applied to projected text states')
    parser.add_argument('--visual-drop-rate', type=float, default=None, help='dropout applied to projected visual states')
    parser.add_argument('--hidden', type=int, default=100, help='hidden size in model training')
    parser.add_argument('--n_classes', type=int, default=2, help='number of classes [defined by args.dataset]')
    parser.add_argument('--n_speakers', type=int, default=2, help='number of speakers [defined by args.dataset]')
    parser.add_argument('--num-experts', type=int, default=2, help='number of experts per modality branch')
    parser.add_argument('--enable-layerwise-sparse-router', action='store_true', default=False,
                        help='route tokens inside each MoE layer over modality-specific plus shared experts')
    parser.add_argument('--num-shared-experts', type=int, default=1,
                        help='number of shared experts available to every modality when layerwise sparse routing is enabled')
    parser.add_argument('--router-topk', type=int, default=2,
                        help='top-k experts kept per token by the layerwise sparse router; 0 keeps dense routing')
    parser.add_argument('--router-noise-std', type=float, default=0.0,
                        help='gaussian noise std added to layerwise router logits during training')
    # Cross-modal directional mix weights
    parser.add_argument('--cross-mix-weight', type=float, default=0.7, help='base cross-modal mix weight (default 0.7)')
    parser.add_argument('--cross-mix-ta', type=float, default=None, help='directional mix: t<-a (overrides base if set)')
    parser.add_argument('--cross-mix-tv', type=float, default=None, help='directional mix: t<-v (overrides base if set)')
    parser.add_argument('--cross-mix-at', type=float, default=None, help='directional mix: a<-t (overrides base if set)')
    parser.add_argument('--cross-mix-vt', type=float, default=None, help='directional mix: v<-t (overrides base if set)')
    parser.add_argument('--enable-av-cross', action='store_true', default=False,
                        help='enable direct audio-video cross-modal interaction in the second stage')
    parser.add_argument('--cross-mix-av', type=float, default=0.2, help='directional mix: a<-v when av cross is enabled')
    parser.add_argument('--cross-mix-va', type=float, default=0.2, help='directional mix: v<-a when av cross is enabled')
    parser.add_argument('--enable-text-stim', action='store_true', default=False,
                        help='use text-guided stimulation to filter and enhance auxiliary modalities before cross fusion')
    parser.add_argument('--text-stim-a-scale', type=float, default=0.0,
                        help='residual scale for text-guided stimulation applied to audio states')
    parser.add_argument('--text-stim-v-scale', type=float, default=0.1,
                        help='residual scale for text-guided stimulation applied to visual states')
    parser.add_argument('--disable-detach-text-stim-source', action='store_true', default=False,
                        help='allow gradients from text-guided stimulation to flow back into text states')
    parser.add_argument('--enable-fusion-gate', action='store_true', default=False,
                        help='apply a MAG-style gated residual from audio/video to the final joint representation')
    parser.add_argument('--fusion-gate-scale', type=float, default=0.1,
                        help='residual scale for the final gated fusion head')
    parser.add_argument('--disable-detach-fusion-text-source', action='store_true', default=False,
                        help='allow gradients from the final gated fusion head to flow back into text states')
    parser.add_argument('--enable-conflict-head', action='store_true', default=False,
                        help='add an explicit pairwise conflict residual built from modality differences/products at the final fusion stage')
    parser.add_argument('--conflict-head-scale', type=float, default=0.1,
                        help='residual scale for the explicit modality-conflict fusion head')
    parser.add_argument('--enable-late-unimodal-calibration', action='store_true', default=False,
                        help='use unimodal predictions as a small residual calibration on top of the final joint head')
    parser.add_argument('--late-calib-scale', type=float, default=0.2,
                        help='maximum residual scale for late unimodal calibration')
    parser.add_argument('--disable-clean-va-path', action='store_true', default=False,
                        help='use the legacy post-text v<-a path instead of the clean pre-text auxiliary path')
    parser.add_argument('--disable-detach-va-source', action='store_true', default=False,
                        help='allow gradients from the v<-a branch to flow back into the source audio states')
    parser.add_argument('--disable-adaptive-va-gate', action='store_true', default=False,
                        help='keep v<-a audio-video mixing fixed instead of using a bounded adaptive gate')
    parser.add_argument('--va-gate-delta', type=float, default=0.03,
                        help='maximum signed adjustment applied around cross-mix-va by the adaptive v<-a gate')
    parser.add_argument('--disable-adaptive-cross-gate', action='store_true', default=False,
                        help='use the original fixed cross-modal mix weights instead of learned gates')
    parser.add_argument('--enable-shared-private-decomp', action='store_true', default=False,
                        help='apply shared/private decomposition regularization to the stage-2 modality representations')
    parser.add_argument('--decomp-recon-weight', type=float, default=0.02,
                        help='weight for shared/private reconstruction regularization')
    parser.add_argument('--decomp-orth-weight', type=float, default=0.01,
                        help='weight for shared/private orthogonality regularization')
    parser.add_argument('--decomp-align-weight', type=float, default=0.01,
                        help='weight for cross-modal shared-space alignment regularization')
    parser.add_argument('--enable-label-prototype', action='store_true', default=False,
                        help='add a label-prototype auxiliary branch on the final joint representation for MOSI/MOSEI')
    parser.add_argument('--prototype-dim', type=int, default=32,
                        help='embedding dimension for the label-prototype auxiliary space')
    parser.add_argument('--prototype-bins', type=int, default=7,
                        help='number of ordered sentiment anchors spanning the regression range')
    parser.add_argument('--prototype-temperature', type=float, default=0.5,
                        help='temperature applied to prototype similarity logits')
    parser.add_argument('--prototype-target-sigma', type=float, default=0.75,
                        help='gaussian smoothing width used to build soft ordinal targets around the gold score')
    parser.add_argument('--prototype-min', type=float, default=-3.0,
                        help='minimum sentiment value covered by the prototype anchors')
    parser.add_argument('--prototype-max', type=float, default=3.0,
                        help='maximum sentiment value covered by the prototype anchors')
    parser.add_argument('--prototype-kl-weight', type=float, default=0.03,
                        help='weight for the prototype soft-target KL loss')
    parser.add_argument('--prototype-reg-weight', type=float, default=0.05,
                        help='weight for the prototype expected-score regression loss')
    parser.add_argument('--prototype-align-weight', type=float, default=0.01,
                        help='weight for aligning the hidden state with the soft target prototype mixture')
    parser.add_argument('--enable-intensity-contrastive', action='store_true', default=False,
                        help='apply a sentiment-intensity-guided contrastive regularizer on the final hidden states')
    parser.add_argument('--contrastive-dim', type=int, default=32,
                        help='projection size used by the intensity-guided contrastive head')
    parser.add_argument('--contrastive-label-sigma', type=float, default=0.75,
                        help='gaussian width that maps label differences to target representation similarity')
    parser.add_argument('--intensity-contrastive-weight', type=float, default=0.03,
                        help='weight for the sentiment-intensity-guided contrastive loss')
    parser.add_argument('--enable-polarity-loss', action='store_true', default=False,
                        help='optimize a polarity BCE auxiliary loss on top of the scalar sentiment output for MOSI/MOSEI')
    parser.add_argument('--polarity-loss-weight', type=float, default=0.25,
                        help='weight for the polarity BCE auxiliary loss')
    parser.add_argument('--enable-stage2-modality-dropout', action='store_true', default=False,
                        help='randomly drop active modalities during stage2 training while keeping at least one modality per utterance')
    parser.add_argument('--audio-modality-dropout', type=float, default=0.15,
                        help='stage2 train-time audio modality dropout probability')
    parser.add_argument('--text-modality-dropout', type=float, default=0.05,
                        help='stage2 train-time text modality dropout probability')
    parser.add_argument('--visual-modality-dropout', type=float, default=0.25,
                        help='stage2 train-time visual modality dropout probability')
    parser.add_argument('--monitor-metric', type=str, default='auto',
                        choices=['auto', 'f1', 'acc', 'f1_acc', 'corr', 'neg_mae'],
                        help='metric used to select the best checkpoint and checkpoint soup candidates')

    ## Params for training
    parser.add_argument('--no-cuda', action='store_true', default=False, help='does not use GPU')
    parser.add_argument('--gpu', type=int, default=2, help='index of gpu')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR', help='learning rate')
    parser.add_argument('--stage2-lr', type=float, default=None,
                        help='optional stage2 base learning rate; defaults to --lr when omitted')
    parser.add_argument('--reset-optimizer-at-stage2', action='store_true', default=False,
                        help='rebuild Adam when switching to stage2 after restoring the best stage1 branches')
    parser.add_argument('--stage2-adam-buffer-scale', type=float, default=1.0,
                        help='multiply Adam momentum/variance buffers by this factor at the stage2 transition; 1.0 keeps the original optimizer state')
    parser.add_argument('--enable-stage2-cosine-lr', action='store_true', default=False,
                        help='apply cosine decay during stage2 using --stage2-lr/--lr as the starting LR')
    parser.add_argument('--stage2-min-lr-ratio', type=float, default=0.3,
                        help='final stage2 LR ratio for cosine decay, relative to the stage2 base LR')
    parser.add_argument('--l2', type=float, default=0.00001, metavar='L2', help='L2 regularization weight')
    parser.add_argument('--batch-size', type=int, default=32, metavar='BS', help='batch size')
    parser.add_argument('--epochs', type=int, default=100, metavar='E', help='number of epochs')
    parser.add_argument('--num-folder', type=int, default=5, help='folders for cross-validation [defined by args.dataset]')
    parser.add_argument('--seed', type=int, default=100, help='make split manner is same with same seed')
    parser.add_argument('--test_condition', type=str, default='atv', choices=['a', 't', 'v', 'at', 'av', 'tv', 'atv'], help='test conditions')
    parser.add_argument('--stage_epoch', type=float, default=100, help='number of epochs of the first stage')
    parser.add_argument('--best-epoch-scope', type=str, default='stage2', choices=['all', 'stage2'],
                        help='scope used to pick the best checkpoint/metrics; stage2 ignores pre-transition epochs')
    parser.add_argument('--checkpoint-soup-topk', type=int, default=0,
                        help='if >1, average the top-k stage2 checkpoints ranked by the monitor metric and evaluate the soup at the end')
    parser.add_argument('--exp-name', type=str, default='', help='optional experiment name used in saved model filenames')
    parser.add_argument('--cpu-threads', type=int, default=1, help='cap intra-op CPU threads; default is 1 to keep CPU usage low')
    parser.add_argument('--cpu-interop-threads', type=int, default=1, help='cap inter-op CPU threads; default is 1 to keep CPU usage low')
    parser.add_argument('--collect-analysis', action='store_true', default=False, help='collect per-batch analysis arrays on CPU')
    parser.add_argument('--save-log', action='store_true', default=False,
                        help='persist stdout logs under ./saved/log; disabled by default')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='persist the final model checkpoint under ./saved/model; disabled by default')

    args = parser.parse_args()
    args.stage2_min_lr_ratio = min(1.0, max(0.0, args.stage2_min_lr_ratio))
    args.stage2_adam_buffer_scale = min(1.0, max(0.0, args.stage2_adam_buffer_scale))
    resolved_cpu_threads, resolved_cpu_interop_threads = resolve_cpu_thread_settings(
        args.cpu_threads,
        args.cpu_interop_threads,
    )
    args.cpu_threads = resolved_cpu_threads
    args.cpu_interop_threads = resolved_cpu_interop_threads
    os.environ['OMP_NUM_THREADS'] = str(args.cpu_threads)
    os.environ['MKL_NUM_THREADS'] = str(args.cpu_threads)
    os.environ['OPENBLAS_NUM_THREADS'] = str(args.cpu_threads)
    os.environ['NUMEXPR_NUM_THREADS'] = str(args.cpu_threads)
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    args.device = device
    torch.set_num_threads(args.cpu_threads)
    torch.set_num_interop_threads(args.cpu_interop_threads)
    save_folder_name = f'{args.dataset}'
    time_dataset = f"{datetime.datetime.now().strftime('%Y-%m-%d_%H_%M_%S')}_{args.dataset}"
    if args.save_log:
        save_log = os.path.join(config.get_save_dir('log'), 'main_result', f'{save_folder_name}')
        if not os.path.exists(save_log):
            os.makedirs(save_log)
        sys.stdout = Logger(
            filename=f"{save_log}/{time_dataset}_batchsize-{args.batch_size}_lr-{args.lr}_seed-{args.seed}_test-condition-{args.test_condition}.txt",
            stream=sys.stdout,
        )

    ## seed
    def seed_torch(seed):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    seed_torch(args.seed)


    ## dataset
    if args.dataset in ['CMUMOSI', 'CMUMOSEI']:
        args.num_folder = 1
        args.n_classes = 1
        args.n_speakers = 1
    elif args.dataset == 'IEMOCAPFour':
        args.num_folder = 5
        args.n_classes = 4
        args.n_speakers = 2
    elif args.dataset == 'IEMOCAPSix':
        args.num_folder = 5
        args.n_classes = 6
        args.n_speakers = 2
    cuda = torch.cuda.is_available() and not args.no_cuda
    print(args)

    ## reading data
    print (f'====== Reading Data =======')
    audio_feature, text_feature, video_feature = args.audio_feature, args.text_feature, args.video_feature
    audio_root = os.path.join(config.PATH_TO_FEATURES[args.dataset], audio_feature)
    text_root = os.path.join(config.PATH_TO_FEATURES[args.dataset], text_feature)
    video_root = os.path.join(config.PATH_TO_FEATURES[args.dataset], video_feature)
    print(audio_root)
    print(text_root)
    print(video_root)
    assert os.path.exists(audio_root) and os.path.exists(text_root) and os.path.exists(video_root), f'features not exist!'
    train_loaders, test_loaders, adim, tdim, vdim = get_loaders(audio_root=audio_root,
                                                                             text_root=text_root,
                                                                             video_root=video_root,
                                                                             num_folder=args.num_folder,
                                                                             batch_size=args.batch_size,
                                                                             dataset=args.dataset,
                                                                             num_workers=0)
    assert len(train_loaders) == args.num_folder, f'Error: folder number'

    
    print (f'====== Training and Testing =======')
    folder_mae = []
    folder_corr = []
    folder_acc = []
    folder_f1 = []
    folder_model = []
    for ii in range(args.num_folder):
        print (f'>>>>> Cross-validation: training on the {ii+1} folder >>>>>')
        train_loader = train_loaders[ii]
        test_loader = test_loaders[ii]
        start_time = time.time()

        print('-'*80)
        print (f'Step1: build model (each folder has its own model)')
        model = build_model(args, adim, tdim, vdim)
        reg_loss = MaskedMSELoss()
        cls_loss = MaskedCELoss()
        polarity_loss = MaskedBCEWithLogitsLoss(ignore_zero_targets=True)
        if cuda:
            model.to(device)
        optimizer = build_optimizer(model, args.lr, args.l2)
        print('-'*80)


        print (f'Step2: training (multiple epoches)')
        train_acc_as, train_acc_ts, train_acc_vs = [], [], []
        test_fscores, test_accs, test_maes, test_corrs = [], [], [], []
        best_stage1_states = {'a': None, 't': None, 'v': None}
        best_stage1_scores = {'a': float('-inf'), 't': float('-inf'), 'v': float('-inf')}
        best_monitor = float('-inf')
        soup_candidates = []
        bestmodel = None
        best_index_test = None
        bestmae = bestcorr = bestf1 = bestacc = None
        start_first_stage_time = time.time()

        print("------- Starting the first stage! -------")
        for epoch in range(args.epochs):
            first_stage = True if epoch < args.stage_epoch else False
            current_lr = get_epoch_lr(args, epoch)
            set_optimizer_lr(optimizer, current_lr)
            ## training and testing (!!! if IEMOCAP, the ua is equal to corr !!!)
            train_mae, train_corr, train_acc, train_fscore, train_acc_atv, train_names, train_loss, weight_train = train_or_eval_model(args, model, reg_loss, cls_loss, polarity_loss, train_loader, \
                                                                            optimizer=optimizer, train=True, first_stage=first_stage, mark='train')
            test_mae, test_corr, test_acc, test_fscore, test_acc_atv, test_names, test_loss, weight_test = train_or_eval_model(args, model, reg_loss, cls_loss, polarity_loss, test_loader, \
                                                                            optimizer=None, train=False, first_stage=first_stage, mark='test')


            ## save
            test_accs.append(test_acc)
            test_fscores.append(test_fscore)
            test_maes.append(test_mae)
            test_corrs.append(test_corr)
            current_state = snapshot_model_state(model)
            train_acc_as.append(train_acc_atv[0])
            train_acc_ts.append(train_acc_atv[1])
            train_acc_vs.append(train_acc_atv[2])

            if first_stage:
                stage1_scores = {
                    'a': train_acc_atv[0],
                    't': train_acc_atv[1],
                    'v': train_acc_atv[2],
                }
                for branch_name, score in stage1_scores.items():
                    if score >= best_stage1_scores[branch_name]:
                        best_stage1_scores[branch_name] = score
                        best_stage1_states[branch_name] = current_state

            monitor = compute_monitor_value(
                dataset=args.dataset,
                metric_mode=args.monitor_metric,
                f1=test_fscore,
                acc=test_acc,
                corr=test_corr,
                mae=test_mae,
            )
            eligible_for_best = (args.best_epoch_scope == 'all') or (not first_stage)
            if eligible_for_best and monitor >= best_monitor:
                best_monitor = monitor
                bestmodel = current_state
                best_index_test = epoch
                bestmae = test_mae
                bestcorr = test_corr
                bestf1 = test_fscore
                bestacc = test_acc
            if (not first_stage) and args.checkpoint_soup_topk and args.checkpoint_soup_topk > 1:
                soup_candidates.append({
                    'monitor': float(monitor),
                    'epoch': epoch,
                    'state': current_state,
                })
                soup_candidates = sorted(soup_candidates, key=lambda item: item['monitor'], reverse=True)[:args.checkpoint_soup_topk]

            if first_stage:
                print(f'epoch:{epoch}; a_acc_train:{train_acc_atv[0]:.3f}; t_acc_train:{train_acc_atv[1]:.3f}; v_acc_train:{train_acc_atv[2]:.3f}')
                print(f'epoch:{epoch}; a_acc_test:{test_acc_atv[0]:.3f}; t_acc_test:{test_acc_atv[1]:.3f}; v_acc_test:{test_acc_atv[2]:.3f}')
            else:
                lr_str = f'; lr={current_lr:.6g}'
                mix_weight_val = get_cross_mix_weight(model)
                mix_str = f'; mix_w:{mix_weight_val}' if mix_weight_val is not None else ''
                router_summary = get_router_summary(model)
                router_str = f'; router:{router_summary}' if router_summary is not None else ''
                decomp_summary = get_decomp_summary(model)
                decomp_str = f'; decomp:{decomp_summary}' if decomp_summary is not None else ''
                proto_summary = get_prototype_summary(model)
                proto_str = f'; proto:{proto_summary}' if proto_summary is not None else ''
                contrastive_summary = get_contrastive_summary(model)
                contrastive_str = f'; {contrastive_summary}' if contrastive_summary is not None else ''
                print(f'epoch:{epoch}; train_mae_{args.test_condition}:{train_mae:.3f}; train_corr_{args.test_condition}:{train_corr:.3f}; '
                      f'train_fscore_{args.test_condition}:{train_fscore:2.2%}; train_acc_{args.test_condition}:{train_acc:2.2%}; '
                      f'train_loss_{args.test_condition}:{train_loss}{lr_str}{mix_str}{router_str}{decomp_str}{proto_str}{contrastive_str}')
                print(f'epoch:{epoch}; test_mae_{args.test_condition}:{test_mae:.3f}; test_corr_{args.test_condition}:{test_corr:.3f}; '
                      f'test_fscore_{args.test_condition}:{test_fscore:2.2%}; test_acc_{args.test_condition}:{test_acc:2.2%}; '
                      f'test_loss_{args.test_condition}:{test_loss}{lr_str}{mix_str}{router_str}{decomp_str}{proto_str}{contrastive_str}')
            print('-'*10)
            ## update the parameter for the 2nd stage
            if epoch == args.stage_epoch-1:
                model_idx_a = int(torch.argmax(torch.Tensor(train_acc_as)))
                print(f'best_epoch_a: {model_idx_a}')

                model_idx_t = int(torch.argmax(torch.Tensor(train_acc_ts)))
                print(f'best_epoch_t: {model_idx_t}')

                model_idx_v = int(torch.argmax(torch.Tensor(train_acc_vs)))
                print(f'best_epoch_v: {model_idx_v}')
                load_branch_states(model, best_stage1_states)
                if args.reset_optimizer_at_stage2:
                    stage2_lr = args.stage2_lr if args.stage2_lr is not None else args.lr
                    optimizer = build_optimizer(model, stage2_lr, args.l2)
                    print(f"Reset optimizer for stage2 with lr={stage2_lr:.6g}")
                elif args.stage2_lr is not None:
                    set_optimizer_lr(optimizer, args.stage2_lr)
                    print(f"Switch stage2 lr to {args.stage2_lr:.6g}")
                if (not args.reset_optimizer_at_stage2) and args.stage2_adam_buffer_scale < 1.0:
                    scale_optimizer_state(optimizer, args.stage2_adam_buffer_scale)
                    print(f"Scale Adam buffers for stage2 by {args.stage2_adam_buffer_scale:.3f}")

                end_first_stage_time = time.time()
                print("------- Starting the second stage! -------")

        end_second_stage_time = time.time()
        print("-"*80)
        print(f"Time of first stage: {end_first_stage_time - start_first_stage_time}s")
        print(f"Time of second stage: {end_second_stage_time - end_first_stage_time}s")
        print("-" * 80)

        if args.checkpoint_soup_topk and args.checkpoint_soup_topk > 1 and len(soup_candidates) >= 2:
            soup_state = average_model_states([item['state'] for item in soup_candidates])
            model.load_state_dict(soup_state)
            soup_mae, soup_corr, soup_acc, soup_fscore, _, _, soup_loss, _ = train_or_eval_model(
                args,
                model,
                reg_loss,
                cls_loss,
                polarity_loss,
                test_loader,
                optimizer=None,
                train=False,
                first_stage=False,
                mark='test',
            )
            soup_monitor = compute_monitor_value(
                dataset=args.dataset,
                metric_mode=args.monitor_metric,
                f1=soup_fscore,
                acc=soup_acc,
                corr=soup_corr,
                mae=soup_mae,
            )
            soup_epochs = ",".join(str(item['epoch']) for item in soup_candidates)
            if args.dataset in ['CMUMOSI', 'CMUMOSEI']:
                print(f"Checkpoint soup(top{len(soup_candidates)} @ epochs {soup_epochs}) --test_mae {soup_mae} --test_corr {soup_corr} --test_fscores {soup_fscore} --test_acc {soup_acc} --test_loss {soup_loss}")
            else:
                print(f"Checkpoint soup(top{len(soup_candidates)} @ epochs {soup_epochs}) --test_acc {soup_acc} --test_ua {soup_corr} --test_loss {soup_loss}")
            if soup_monitor >= best_monitor:
                best_monitor = soup_monitor
                bestmodel = snapshot_model_state(model)
                best_index_test = f"soup[{soup_epochs}]"
                bestmae = soup_mae
                bestcorr = soup_corr
                bestf1 = soup_fscore
                bestacc = soup_acc
            elif bestmodel is not None:
                model.load_state_dict(bestmodel)
        elif bestmodel is not None:
            model.load_state_dict(bestmodel)

        print(f'Step3: finalizing and testing on the {ii+1} folder')
        if bestmodel is None:
            raise RuntimeError('No best model state was recorded during training. Check best-epoch-scope against stage_epoch/epochs.')

        folder_mae.append(bestmae)
        folder_corr.append(bestcorr)
        folder_f1.append(bestf1)
        folder_acc.append(bestacc)
        folder_model.append(bestmodel)
        end_time = time.time()


        if args.dataset in ['CMUMOSI', 'CMUMOSEI']:
            print(f"The best(acc) epoch of test_condition ({args.test_condition}): {best_index_test} --test_mae {bestmae} --test_corr {bestcorr} --test_fscores {bestf1} --test_acc {bestacc}.")
            print(f'>>>>> Finish: training on the {ii+1} folder, duration: {end_time - start_time} >>>>>')
        if args.dataset in ['IEMOCAPFour', 'IEMOCAPSix']:
            print(f"The best(acc) epoch of test_condition ({args.test_condition}): {best_index_test} --test_acc {bestacc} --test_ua {bestcorr}.")
            print(f'>>>>> Finish: training on the {ii+1} folder, duration: {end_time - start_time} >>>>>')

    print('-'*80)
    if args.dataset in ['CMUMOSI', 'CMUMOSEI']:
        print(f"Folder avg: test_condition ({args.test_condition}) --test_mae {np.mean(folder_mae)} --test_corr {np.mean(folder_corr)} --test_fscores {np.mean(folder_f1)} --test_acc{np.mean(folder_acc)}")
    if args.dataset in ['IEMOCAPFour', 'IEMOCAPSix']:
        print(f"Folder avg: test_condition ({args.test_condition}) --test_acc{np.mean(folder_acc)} --test_ua {np.mean(folder_corr)}")

    if args.save_model:
        print (f'====== Saving =======')
        save_model = os.path.join(config.get_save_dir('model'), 'main_result', f'{save_folder_name}')
        if not os.path.exists(save_model):
            os.makedirs(save_model)
        ## gain suffix_name
        suffix_name = f"{time_dataset}_hidden-{args.hidden}_bs-{args.batch_size}"
        ## gain feature_name
        feature_name = f'{audio_feature};{text_feature};{video_feature}'
        ## gain res_name
        mean_mae = np.mean(np.array(folder_mae))
        mean_corr = np.mean(np.array(folder_corr))
        mean_f1 = np.mean(np.array(folder_f1))
        mean_acc = np.mean(np.array(folder_acc))
        if args.dataset in ['CMUMOSI', 'CMUMOSEI']:
            primary_score = mean_f1 * 100.0
        if args.dataset in ['IEMOCAPFour', 'IEMOCAPSix']:
            primary_score = mean_acc * 100.0
        exp_name = args.exp_name.strip().replace(' ', '-') if args.exp_name else ''
        file_prefix = f'{primary_score:.2f}'
        if exp_name:
            file_prefix = f'{file_prefix}_{exp_name}'
        save_path = f'{save_model}/{file_prefix}_{suffix_name}_features-{feature_name}_test-condition-{args.test_condition}.pth'
        torch.save({'model': folder_model[-1], 'fold_models': folder_model}, save_path)
        print(save_path)
    else:
        print('====== Skipping checkpoint save (--save-model is disabled) =======')
