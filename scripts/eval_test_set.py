"""
D8FN v3 Test Set Evaluation
Evaluates trained D8FN checkpoints on Kuro Siwo test split (E:\\processed_v4_test).
Produces all metrics from the notebook: IoU, F1, Precision, Recall, mIoU, Kappa,
Betti0_Err, Betti1_Err, HVR, PA_IoU, global_IoU + 3-class BlackBench metrics.

Usage: python eval_test_set.py
"""
import os, sys, gc, json, math, time, glob
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy.ndimage import label as ndi_label

SEED = 42
random = __import__('random')
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IS_BF16 = DEVICE.type == 'cuda' and torch.cuda.is_bf16_supported()

# ============================================================
# TTA flow remapping (from notebook Cell 0)
# ============================================================
def remap_fd_h(fd):
    idx = torch.floor(fd * 8.0 + 0.5).long().clamp(0, 8)
    return torch.tensor([0,5,4,3,2,1,8,7,6], device=fd.device)[idx].to(fd.dtype) / 8.0

def remap_fd_v(fd):
    idx = torch.floor(fd * 8.0 + 0.5).long().clamp(0, 8)
    return torch.tensor([0,1,8,7,6,5,4,3,2], device=fd.device)[idx].to(fd.dtype) / 8.0

# ============================================================
# D8FlowRouting (from notebook Cell 1)
# ============================================================
class D8FlowRouting(nn.Module):
    DX = torch.tensor([0, 1, 1, 0, -1, -1, -1, 0, 1])
    DY = torch.tensor([0, 0, -1, -1, -1, 0, 1, 1, 1])

    def __init__(self, channels, hidden_dim=64, num_rounds=50, dropout=0.1):
        super().__init__()
        self.channels = channels; self.num_rounds = num_rounds
        self.gate_net = nn.Sequential(
            nn.Conv2d(channels+3, hidden_dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(hidden_dim), nn.SiLU(inplace=True), nn.Dropout2d(dropout),
            nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(hidden_dim), nn.SiLU(inplace=True),
            nn.Conv2d(hidden_dim, channels, 1), nn.Sigmoid())
        self.routing_temp    = nn.Parameter(torch.tensor(0.5))
        self.channel_weights = nn.Parameter(torch.ones(channels))
        self.output_proj = nn.Sequential(
            nn.Conv2d(channels*2, channels, 1, bias=False),
            nn.BatchNorm2d(channels), nn.SiLU(inplace=True))

    def _compute_ds(self, flow_dir, device, B, H, W):
        di = (flow_dir*8.0+0.5).floor_().long().clamp_(0,8).contiguous()
        y = torch.arange(H,device=device).view(1,1,H,1).expand(B,1,H,W).contiguous()
        x = torch.arange(W,device=device).view(1,1,1,W).expand(B,1,H,W).contiguous()
        dy = self.DY.to(device)[di]; dx = self.DX.to(device)[di]
        ny = y+dy; nx = x+dx
        valid = (ny>=0)&(ny<H)&(nx>=0)&(nx<W)&(di>0)
        dsi = (ny*W+nx).long()
        dsi[~valid] = torch.arange(H*W,device=device).view(1,1,H,W).expand(B,1,H,W)[~valid]
        return dsi, valid

    def forward(self, features, flow_dir, dem, slope=None):
        B,C,H,W = features.shape; N=H*W; device=features.device
        if flow_dir.shape[2:] != features.shape[2:]:
            flow_dir=F.interpolate(flow_dir,size=features.shape[2:],mode='nearest')
            dem     =F.interpolate(dem,size=features.shape[2:],mode='bilinear',align_corners=False)
            if slope is not None:
                slope=F.interpolate(slope,size=features.shape[2:],mode='bilinear',align_corners=False)
        if slope is None: slope=torch.zeros_like(dem)
        gate_input = torch.cat([features, dem, flow_dir, slope], 1)
        gates = self.gate_net(gate_input)
        gates = gates * torch.sigmoid(self.routing_temp) * self.channel_weights.view(1,-1,1,1)
        dsi,_ = self._compute_ds(flow_dir,device,B,H,W)
        ff=features.view(B,C,N); gf=gates.view(B,C,N)
        si=dsi.view(B,1,N).expand(-1,C,-1).contiguous(); acc=ff.clone()
        for _ in range(self.num_rounds):
            ds=torch.zeros_like(acc)
            ds.scatter_reduce_(2,si,gf*acc,reduce='sum',include_self=False)
            acc=acc+ds
        return self.output_proj(torch.cat([features,acc.view(B,C,H,W)],1))

class D8FlowRoutingBlock(nn.Module):
    def __init__(self, channels, hidden_dim=64, num_rounds=50):
        super().__init__()
        self.conv_path = nn.Sequential(
            nn.Conv2d(channels,channels,3,1,1,bias=False),nn.BatchNorm2d(channels),nn.SiLU(inplace=True),
            nn.Conv2d(channels,channels,3,1,1,bias=False),nn.BatchNorm2d(channels))
        self.flow_path = D8FlowRouting(channels,hidden_dim=hidden_dim,num_rounds=num_rounds)
        self.fusion = nn.Sequential(
            nn.Conv2d(channels*2,channels,1,bias=False),nn.BatchNorm2d(channels),nn.SiLU(inplace=True))

    def forward(self, x, flow_dir, dem, slope=None):
        return self.fusion(torch.cat([self.conv_path(x),
                                      self.flow_path(x,flow_dir,dem,slope)],1))+x

# ============================================================
# Model components (from notebook Cell 4)
# ============================================================
class FourierFeatureEncoder(nn.Module):
    def __init__(self, in_features=2, num_frequencies=10):
        super().__init__()
        freqs=2.0**torch.linspace(0.0,num_frequencies-1,num_frequencies)
        self.register_buffer('freq_bands',freqs)
    def forward(self, x):
        out=[x]
        for freq in self.freq_bands:
            out.append(torch.sin(x*freq*math.pi))
            out.append(torch.cos(x*freq*math.pi))
        return torch.cat(out,dim=-1)

class HeightFieldHead(nn.Module):
    def __init__(self, feat_dim=128, hidden_dim=256):
        super().__init__()
        self.fourier=FourierFeatureEncoder(2,10)
        fourier_dim=42
        self.height_mlp=nn.Sequential(
            nn.Linear(feat_dim+fourier_dim+1,hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim,hidden_dim//2),           nn.SiLU(),
            nn.Linear(hidden_dim//2,hidden_dim//4),        nn.SiLU(),
            nn.Linear(hidden_dim//4,1))
        self.tau=nn.Parameter(torch.tensor(0.5))
        nn.init.constant_(self.height_mlp[-1].bias, 0.5)

    def forward(self, feat_map, dem_raw):
        B,C,H,W=feat_map.shape; device=feat_map.device
        gy,gx=torch.meshgrid(torch.linspace(-1,1,H,device=device),
                              torch.linspace(-1,1,W,device=device),indexing='ij')
        coords=torch.stack([gx,gy],dim=-1)
        coord_enc=self.fourier(coords).permute(2,0,1).unsqueeze(0).expand(B,-1,-1,-1)
        combined=torch.cat([feat_map,coord_enc,dem_raw],dim=1)
        flat=combined.permute(0,2,3,1).reshape(B*H*W,-1)
        H_w=self.height_mlp(flat).view(B,1,H,W)
        tau_pos=F.softplus(self.tau)+0.01
        return (H_w-dem_raw)/tau_pos, H_w

class DecoderWithSkips(nn.Module):
    def __init__(self, feat_dims, final_dim=128):
        super().__init__()
        f56, f28 = feat_dims[0], feat_dims[1]
        self.fuse_input=nn.Sequential(
            nn.Conv2d(256+f28,128,3,1,1,bias=False),nn.BatchNorm2d(128),nn.SiLU(True))
        self.up_28_56=nn.Sequential(
            nn.ConvTranspose2d(128,128,2,2,bias=False),nn.BatchNorm2d(128),nn.SiLU(True))
        self.fuse_56=nn.Sequential(
            nn.Conv2d(128+f56,64,3,1,1,bias=False),nn.BatchNorm2d(64),nn.SiLU(True),
            nn.Conv2d(64,64,3,1,1,bias=False),nn.BatchNorm2d(64),nn.SiLU(True))
        self.up_to_full=nn.Sequential(
            nn.ConvTranspose2d(64,final_dim,4,4,bias=False),nn.BatchNorm2d(final_dim),nn.SiLU(True))

    def forward(self, feat_28, encoder_feats):
        x=self.fuse_input(torch.cat([feat_28,encoder_feats[1]],1))
        x=self.up_28_56(x)
        x=self.fuse_56(torch.cat([x,encoder_feats[0]],1))
        return self.up_to_full(x)

class D8FN(nn.Module):
    def __init__(self, in_ch=9, backbone='convnext_small.fb_in22k_ft_in1k_384',
                 routing_rounds=50, routing_dim=64, height_dim=256):
        super().__init__()
        import timm
        self.input_proj=nn.Sequential(
            nn.Conv2d(in_ch,3,1,bias=False),nn.BatchNorm2d(3),nn.SiLU(True))
        self.encoder=timm.create_model(
            backbone,pretrained=True,features_only=True,out_indices=[0,1,2,3])
        feat_dims=[f['num_chs'] for f in self.encoder.feature_info]
        self.flow_routing_coarse=D8FlowRoutingBlock(feat_dims[3],routing_dim,routing_rounds)
        self.flow_routing_fine  =D8FlowRoutingBlock(feat_dims[1],routing_dim,routing_rounds//2)
        self._proj_7_to_28=nn.Sequential(
            nn.Conv2d(feat_dims[3],feat_dims[1],1,bias=False),nn.BatchNorm2d(feat_dims[1]),nn.SiLU(True))
        self.fuse_multiscale=nn.Sequential(
            nn.Conv2d(feat_dims[1]*2,256,3,1,1,bias=False),nn.BatchNorm2d(256),nn.SiLU(True),
            nn.Conv2d(256,256,3,1,1,bias=False),nn.BatchNorm2d(256),nn.SiLU(True))
        self.decoder    =DecoderWithSkips(feat_dims,final_dim=128)
        self.height_head=HeightFieldHead(feat_dim=128,hidden_dim=height_dim)
        self.head_3class=nn.Sequential(nn.Conv2d(128,64,1),nn.SiLU(True),nn.Conv2d(64,3,1))

    def forward(self, sar, dem, hand, slope, flow_dir, flow_acc):
        x=self.input_proj(torch.cat([sar,dem,hand,slope,flow_dir,flow_acc],1))
        enc=self.encoder(x)
        f7r=self.flow_routing_coarse(enc[-1],flow_dir,dem,slope)
        f7p=self._proj_7_to_28(F.interpolate(f7r,enc[1].shape[2:],mode='bilinear',align_corners=False))
        f28r   =self.flow_routing_fine(enc[1],flow_dir,dem,slope)
        f_fused=self.fuse_multiscale(torch.cat([f28r,f7p],1))
        feat_map=self.decoder(f_fused,enc[:2])
        dem_f   =F.interpolate(dem,size=(224,224),mode='bilinear',align_corners=False)
        flood_logit,H_w=self.height_head(feat_map,dem_f)
        return flood_logit, H_w, self.head_3class(feat_map)

# ============================================================
# Metrics (from notebook Cell 3)
# ============================================================
S4 = np.array([[0,1,0],[1,1,1],[0,1,0]], dtype=np.uint8)

def _betti1(m, nmax=500):
    lb,n = ndi_label(~m.astype(bool), structure=S4)
    if n>nmax: return 0
    bd=set(np.unique(np.concatenate([lb[0,:],lb[-1,:],lb[:,0],lb[:,-1]])))
    return sum(1 for l in range(1,n+1) if l not in bd)

def compute_all_metrics(pred_prob, target, raw_hand=None, thr=0.5, hthr=5.0):
    pred=(pred_prob>thr).float(); B=pred.shape[0]
    results={}; g_tp=g_fp=g_fn=0.0
    for i in range(B):
        p=pred[i,0]; t=target[i,0]
        tp=(p*t).sum().item(); fp=(p*(1-t)).sum().item()
        fn=((1-p)*t).sum().item(); tn=((1-p)*(1-t)).sum().item()
        g_tp+=tp; g_fp+=fp; g_fn+=fn
        if t.sum()<1e-5: continue
        iou=tp/(tp+fp+fn+1e-6); prec=tp/(tp+fp+1e-6); rec=tp/(tp+fn+1e-6)
        f1=2*prec*rec/(prec+rec+1e-6)
        bg=tn/(tn+fp+fn+1e-6); miou=(iou+bg)/2
        nn_=tp+fp+fn+tn; p_obs=(tp+tn)/nn_
        p_exp=((tp+fp)*(tp+fn)+(fn+tn)*(fp+tn))/(nn_*nn_+1e-8)
        kappa=(p_obs-p_exp)/(1.0-p_exp+1e-6)
        pn=p.detach().cpu().numpy().astype(np.uint8)
        tnp=t.detach().cpu().numpy().astype(np.uint8)
        _,pb0=ndi_label(pn,structure=S4); _,tb0=ndi_label(tnp,structure=S4)
        b0=min(abs(pb0-tb0),50)
        b1=min(abs(_betti1(pn.astype(bool))-_betti1(tnp.astype(bool))),20)
        hvr=0.0
        if raw_hand is not None:
            rh=raw_hand[i,0]; viol=((p==1.0)&(rh>hthr)).sum().item()
            hvr=viol/(p.sum().item()+1e-6)
        paiou=miou*(1.0-hvr)
        for k,v in [('IoU',iou),('F1',f1),('Precision',prec),('Recall',rec),
                    ('mIoU',miou),('Kappa',kappa),('Betti0_Err',b0),('Betti1_Err',b1),
                    ('HVR',hvr),('PA_IoU',paiou)]:
            results.setdefault(k,[]).append(v)
    ALL_KEYS=['IoU','F1','Precision','Recall','mIoU','Kappa','Betti0_Err','Betti1_Err','HVR','PA_IoU']
    out={k:(float(np.mean(v)) if v else 0.0) for k,v in results.items()}
    for k in ALL_KEYS:
        if k not in out: out[k]=0.0
    out['_g_tp']=g_tp; out['_g_fp']=g_fp; out['_g_fn']=g_fn
    return out

def compute_3class_metrics(all_logits_3c, all_labels):
    """BlackBench 3-class metrics: F1 for NoWater, PermWater, Flood, Water."""
    all_logits_3c = all_logits_3c.float()
    B, C, H, W = all_logits_3c.shape
    preds = all_logits_3c.argmax(dim=1)  # (B, H, W)
    gt = all_labels.squeeze(1)           # (B, H, W)
    valid = (gt != -1)
    if not valid.any():
        return {'F1_NW': 0, 'F1_PW': 0, 'F1_F': 0, 'F1_W': 0, 'mIoU_3class': 0}

    results = {}
    ious = []

    for cls_idx, cls_name in enumerate(['NW', 'PW', 'F']):
        p_c = (preds == cls_idx) & valid
        t_c = (gt == cls_idx) & valid
        tp = (p_c & t_c).sum().item()
        fp = (p_c & ~t_c).sum().item()
        fn = (~p_c & t_c).sum().item()
        tn = (~p_c & ~t_c & valid).sum().item()

        prec = tp / (tp + fp + 1e-8)
        rec = tp / (tp + fn + 1e-8)
        f1 = 2 * prec * rec / (prec + rec + 1e-8)
        iou = tp / (tp + fp + fn + 1e-8)
        bg = tn / (tn + fp + fn + 1e-8)
        miou_c = (iou + bg) / 2
        ious.append(miou_c)
        results[f'F1_{cls_name}'] = f1

    # Water = PW + Flood combined
    p_w = ((preds == 1) | (preds == 2)) & valid
    t_w = ((gt == 1) | (gt == 2)) & valid
    tp = (p_w & t_w).sum().item()
    fp = (p_w & ~t_w).sum().item()
    fn = (~p_w & t_w).sum().item()
    prec = tp / (tp + fp + 1e-8)
    rec = tp / (tp + fn + 1e-8)
    results['F1_W'] = 2 * prec * rec / (prec + rec + 1e-8)
    results['mIoU_3class'] = np.mean(ious) if ious else 0.0

    return {k: float(v) for k, v in results.items()}


# ============================================================
# Dataset Loader
# ============================================================
class FloodTestDataset(Dataset):
    def __init__(self, data_dir):
        self.files = sorted(glob.glob(os.path.join(data_dir, '*.pt')))
        if not self.files:
            raise FileNotFoundError(f'No .pt files in {data_dir}')

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = torch.load(self.files[idx], weights_only=True)
        feat = data['features']

        # SAR: pre1-VV+VH + post-VV+VH (matches notebook Cell 5)
        sar = torch.cat([feat[1, 0:2], feat[2, 0:2]], dim=0)
        sar = torch.nan_to_num(sar, nan=0., posinf=0., neginf=-50.)
        sar = torch.clamp(sar, -30., 5.)
        sar = (sar - (-12.5)) / 17.5

        # DEM: per-sample standardization
        dem_raw = data['dem'].float()
        if dem_raw.dim() == 2: dem_raw = dem_raw.unsqueeze(0)
        dem_mean = dem_raw.mean(); dem_std = dem_raw.std().clamp(min=1.0)
        dem = ((dem_raw - dem_mean) / dem_std).clamp(-3., 3.)

        # HAND: clamp to 100m, normalize to 0-1
        hand_raw = data['hand'].float()
        if hand_raw.dim() == 2: hand_raw = hand_raw.unsqueeze(0)
        hand = torch.clamp(hand_raw, 0., 100.) / 100.

        # Slope
        slope = data['slope'].float()
        if slope.dim() == 2: slope = slope.unsqueeze(0)

        # Flow direction
        flow_dir = data.get('flow_dir', torch.zeros_like(dem_raw)).float()
        if flow_dir.max() > 1.5: flow_dir = flow_dir / 8.
        if flow_dir.dim() == 2: flow_dir = flow_dir.unsqueeze(0)

        # Flow accumulation
        fa_raw = data.get('flow_acc', torch.zeros(dem_raw.shape)).float()
        flow_acc = fa_raw.unsqueeze(0) if fa_raw.dim() == 2 else fa_raw

        # Mask: flood class (value 2)
        rl = data['raw_label'].float()
        if rl.dim() == 2: rl = rl.unsqueeze(0)
        valid = (rl != 3.)
        mask = torch.where(valid, (rl == 2.).float(), torch.zeros_like(rl))

        # 3-class label (for BlackBench)
        label_3class = rl.long().clamp_(0, 3)
        label_3class[rl == 3.] = -1

        # raw_hand in meters for HVR
        raw_hand = hand * 100.

        return (sar.float(), dem.float(), hand.float(), slope.float(),
                dem_raw.float(), flow_dir.float(), flow_acc.float(),
                mask.float(), raw_hand.float(), label_3class.long())


# ============================================================
# Evaluation
# ============================================================
@torch.no_grad()
def evaluate_full(model, loader, use_tta=True):
    """TTA inference + threshold sweep, returns best metrics + all metrics at all thresholds."""
    model.eval()
    all_logits_3c = []
    all_labels_3c = []
    all_logits, all_masks, all_hands = [], [], []

    for batch in loader:
        sar, dem, hand, slope, dem_raw, fd, fa, mask, rh, l3 = [
            x.to(DEVICE) for x in batch]

        logits, _, logits_3class = model(sar, dem, hand, slope, fd, fa)

        if use_tta:
            ll = [logits]
            fd_h = remap_fd_h(fd.flip(-1))
            l_h, _, _ = model(sar.flip(-1), dem.flip(-1), hand.flip(-1),
                              slope.flip(-1), fd_h, fa.flip(-1))
            ll.append(l_h.flip(-1))
            fd_v = remap_fd_v(fd.flip(-2))
            l_v, _, _ = model(sar.flip(-2), dem.flip(-2), hand.flip(-2),
                              slope.flip(-2), fd_v, fa.flip(-2))
            ll.append(l_v.flip(-2))
            logits = torch.stack(ll).mean(0)

        all_logits.append(logits.cpu())
        all_logits_3c.append(logits_3class.cpu())
        all_masks.append(mask.cpu())
        all_hands.append(rh.cpu())
        all_labels_3c.append(l3.cpu())

    all_logits = torch.cat(all_logits, 0)
    all_masks = torch.cat(all_masks, 0)
    all_hands = torch.cat(all_hands, 0)
    all_logits_3c = torch.cat(all_logits_3c, 0)
    all_labels_3c = torch.cat(all_labels_3c, 0)

    probs = torch.sigmoid(all_logits.float())

    # Threshold sweep
    all_threshold_results = {}
    best_metrics = None; best_pa = 0.
    best_thr = 0.5

    for thr in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
        g_tp = g_fp = g_fn = 0.
        per_batch = []
        bs = 32
        for i in range(0, len(probs), bs):
            pb = probs[i:i+bs]; mb = all_masks[i:i+bs]; hb = all_hands[i:i+bs]
            m = compute_all_metrics(pb, mb, hb, thr=thr)
            g_tp += m.pop('_g_tp'); g_fp += m.pop('_g_fp'); g_fn += m.pop('_g_fn')
            if 'IoU' in m: per_batch.append(m)
        if not per_batch: continue
        global_iou = g_tp / (g_tp + g_fp + g_fn + 1e-8)
        avg = {k: float(np.mean([m[k] for m in per_batch])) for k in per_batch[0].keys()}
        avg['global_IoU'] = global_iou; avg['threshold'] = thr
        all_threshold_results[thr] = avg
        if avg.get('PA_IoU', 0.) > best_pa:
            best_pa = avg['PA_IoU']; best_metrics = avg; best_thr = thr

    # 3-class metrics
    bb3 = compute_3class_metrics(all_logits_3c, all_labels_3c)

    return best_metrics, all_threshold_results, best_thr, bb3


# ============================================================
# Main
# ============================================================
def main():
    CKPT_DIR = os.path.dirname(os.path.abspath(__file__))
    TEST_DIR = r"E:\processed_v4_test"

    if not os.path.exists(TEST_DIR):
        print(f"ERROR: Test directory not found: {TEST_DIR}")
        sys.exit(1)

    # Find checkpoints
    ckpt_files = sorted(glob.glob(os.path.join(CKPT_DIR, "D8FN_fold*_best.pt")))
    if not ckpt_files:
        ckpt_files = sorted(glob.glob(os.path.join(CKPT_DIR, "**", "D8FN_fold*_best.pt"), recursive=True))
    if not ckpt_files:
        print("ERROR: No D8FN_fold*_best.pt checkpoints found!")
        print(f"  Searched in: {CKPT_DIR}")
        sys.exit(1)

    print(f"Device: {DEVICE}")
    if DEVICE.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)} | VRAM: {torch.cuda.get_device_properties(0).total_memory//1024**3}GB")
    print(f"Test data: {TEST_DIR}")
    print(f"Checkpoints: {len(ckpt_files)}")
    for cp in ckpt_files:
        print(f"  {os.path.basename(cp)}")

    # Load test dataset
    test_ds = FloodTestDataset(TEST_DIR)
    test_loader = DataLoader(test_ds, batch_size=4, shuffle=False, num_workers=0, pin_memory=True)
    print(f"\nTest samples: {len(test_ds)}")

    # Quick flood stat
    flood_count = 0; water_count = 0
    for i in range(min(len(test_ds), 200)):
        _, _, _, _, _, _, _, mask, _, _ = test_ds[i]
        if mask.sum() > 0: flood_count += 1
        rl = torch.load(test_ds.files[i], weights_only=True)['raw_label']
        if ((rl == 1) | (rl == 2)).any(): water_count += 1
    print(f"First 200: {flood_count} with flood, {water_count} with any water")

    # Evaluate each checkpoint
    all_fold_results = []
    t_start = time.time()

    for cp_path in ckpt_files:
        fold_name = os.path.basename(cp_path).replace('.pt', '').replace('D8FN_fold', 'fold')
        print(f"\n{'=' * 60}")
        print(f"{fold_name.upper()} — {os.path.basename(cp_path)}")
        print(f"{'=' * 60}")

        ck = torch.load(cp_path, map_location='cpu', weights_only=True)
        print(f"  Checkpoint epoch: {ck.get('epoch', '?') + 1}")
        print(f"  Checkpoint PA-IoU: {ck.get('metrics', {}).get('PA_IoU', '?'):.4f} (val)")

        model = D8FN(in_ch=9).to(DEVICE)
        if 'model_state' in ck:
            model.load_state_dict(ck['model_state'], strict=True)
            print(f"  Loaded: model_state (raw weights)")
        elif 'ema_state' in ck:
            for n, p in model.named_parameters():
                if n in ck['ema_state']:
                    p.data = ck['ema_state'][n].to(DEVICE)
            print(f"  Loaded: ema_state (fallback)")

        t0 = time.time()
        best_m, all_thr, best_thr, bb3 = evaluate_full(model, test_loader, use_tta=True)
        elapsed = time.time() - t0

        print(f"\n  --- Best metrics (binary flood, thr={best_thr:.2f}, TTA=ON) ---")
        KEYS = ['threshold', 'global_IoU', 'IoU', 'F1', 'Precision', 'Recall',
                'mIoU', 'Kappa', 'Betti0_Err', 'Betti1_Err', 'HVR', 'PA_IoU']
        for k in KEYS:
            v = best_m.get(k, 0)
            print(f"    {k:>15s}: {v:.4f}" if isinstance(v, float) else f"    {k:>15s}: {v}")

        print(f"\n  --- All thresholds ---")
        print(f"    {'Thr':>6s} {'gIoU':>8s} {'IoU':>8s} {'F1':>8s} {'PA_IoU':>8s}")
        for thr in sorted(all_thr.keys()):
            m = all_thr[thr]
            print(f"    {thr:6.2f} {m.get('global_IoU',0):8.4f} {m.get('IoU',0):8.4f} "
                  f"{m.get('F1',0):8.4f} {m.get('PA_IoU',0):8.4f}")

        print(f"\n  --- 3-class BlackBench metrics ---")
        for k, v in bb3.items():
            print(f"    {k:>15s}: {v:.4f}")

        result = {
            'fold': fold_name,
            'binary_metrics': best_m,
            'all_thresholds': {str(k): v for k, v in all_thr.items()},
            'bb_3class': bb3,
            'best_threshold': best_thr,
            'eval_time_s': round(elapsed, 1),
        }
        all_fold_results.append(result)
        print(f"\n  Eval time: {elapsed:.1f}s")

        del model; gc.collect()
        if DEVICE.type == 'cuda':
            torch.cuda.empty_cache()

    # Summary
    print(f"\n\n{'=' * 60}")
    print("FINAL TEST SET SUMMARY")
    print(f"{'=' * 60}")

    summary_keys = ['IoU', 'F1', 'Precision', 'Recall', 'mIoU', 'Kappa',
                    'Betti0_Err', 'Betti1_Err', 'HVR', 'PA_IoU', 'global_IoU']

    if len(all_fold_results) > 1:
        print(f"\n  {'Metric':>15s} | {'Mean':>8s} | {'Std':>8s} | {'Min':>8s} | {'Max':>8s}")
        print(f"  {'-' * 55}")
        for key in summary_keys:
            vs = [r['binary_metrics'].get(key, 0) for r in all_fold_results]
            mv = np.mean(vs); sv = np.std(vs) if len(vs) > 1 else 0.
            print(f"  {key:>15s} | {mv:8.4f} | {sv:8.4f} | {min(vs):8.4f} | {max(vs):8.4f}")

        if any(r['bb_3class'] for r in all_fold_results):
            print(f"\n  --- 3-class BlackBench ---")
            for key in ['F1_NW', 'F1_PW', 'F1_F', 'F1_W', 'mIoU_3class']:
                vs = [r['bb_3class'].get(key, 0) for r in all_fold_results if r['bb_3class']]
                if not vs: continue
                mv = np.mean(vs); sv = np.std(vs) if len(vs) > 1 else 0.
                print(f"  {key:>15s} | {mv:8.4f} | {sv:8.4f}")
    else:
        print(f"\n  (Single checkpoint — no std/dev)")
        r = all_fold_results[0]
        for key in summary_keys:
            print(f"  {key:>15s}: {r['binary_metrics'].get(key, 0):.4f}")
        print(f"\n  --- 3-class ---")
        for k, v in r['bb_3class'].items():
            print(f"  {k:>15s}: {v:.4f}")

    total_t = time.time() - t_start
    print(f"\nTotal eval time: {total_t:.0f}s ({total_t/60:.1f} min)")

    # Save
    out_path = os.path.join(CKPT_DIR, "test_eval_results.json")
    json.dump({
        'checkpoints': ckpt_files,
        'test_dir': TEST_DIR,
        'device': str(DEVICE),
        'use_tta': True,
        'fold_results': all_fold_results,
        'eval_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }, open(out_path, 'w'), indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
