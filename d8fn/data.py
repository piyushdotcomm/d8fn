"""Data loading and preprocessing for Kuro Siwo dataset."""

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.ndimage import minimum_filter
import albumentations as A


# Flow direction mapping: 1=E, 2=NE, 3=N, 4=NW, 5=W, 6=SW, 7=S, 8=SE
DX = np.array([0, 1, 1, 0, -1, -1, -1, 0, 1])
DY = np.array([0, 0, -1, -1, -1, 0, 1, 1, 1])


def compute_d8_flow(dem_np):
    """Compute D8 flow direction and accumulation from DEM.

    Uses Jenson & Domingue (1988) algorithm.
    """
    h, w = dem_np.shape
    dem = np.nan_to_num(dem_np, nan=1e6, posinf=1e6, neginf=1e6)
    padded = np.pad(dem, 1, mode='constant', constant_values=1e6)

    # Flow direction (1-8)
    flow_dir = np.zeros((h, w), dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            neighbors = padded[i:i+3, j:j+3]
            dirs = [neighbors[1, 2], neighbors[0, 2], neighbors[0, 1],
                    neighbors[0, 0], neighbors[1, 0], neighbors[2, 0],
                    neighbors[2, 1], neighbors[2, 2]]
            min_idx = np.argmin(dirs)
            if dirs[min_idx] < dem[i, j]:
                flow_dir[i, j] = min_idx + 1

    # Flow accumulation: Jenson & Domingue algorithm
    acc = np.ones((h, w), dtype=np.float32)
    order = np.argsort(-dem.ravel())
    for idx in order:
        i, j = divmod(idx, w)
        d = int(flow_dir[i, j])
        if d < 1 or d > 8:
            continue
        ni, nj = i + DY[d], j + DX[d]
        if 0 <= ni < h and 0 <= nj < w:
            acc[ni, nj] += acc[i, j]

    return flow_dir.astype(np.float32), acc.astype(np.float32)


def preprocess_sample(data_dir, idx, processed_dir):
    """Preprocess a single sample from the tortilla dataset."""
    # This function would be called during dataset creation
    # For Kaggle execution, preprocessing is done once and cached
    pass


def compute_terrain_features(dem_np):
    """Compute HAND and slope from DEM."""
    # HAND (Height Above Nearest Drainage)
    local_min = minimum_filter(dem_np, size=15)
    hand = (dem_np - local_min).astype(np.float32)

    # Slope
    gy, gx = np.gradient(dem_np)
    slope = np.sqrt(gx**2 + gy**2)
    slope_deg = np.degrees(np.arctan(slope))
    slope_norm = np.clip(slope_deg / 45.0, 0, 1).astype(np.float32)

    return hand, slope_norm


train_aug = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.GaussianBlur(blur_limit=(3, 5), p=0.2),
    A.RandomBrightnessContrast(brightness_limit=0.05, contrast_limit=0.05, p=0.2),
])


class FloodDataset(Dataset):
    """Kuro Siwo flood mapping dataset with D8 flow features."""

    def __init__(self, data_dir, augment=False, fold=None, fold_idx=None,
                 num_folds=5, cache=True):
        """
        Args:
            data_dir: Path to preprocessed .pt files.
            augment: Enable augmentation.
            fold: Fold to use (0 to num_folds-1) or None for all.
            fold_idx: Which fold this dataset represents ('train' or 'val').
            num_folds: Number of CV folds.
            cache: Cache data in memory.
        """
        self.files = sorted(glob.glob(os.path.join(data_dir, '*.pt')))
        if len(self.files) == 0:
            raise FileNotFoundError(f'No .pt files in {data_dir}')

        # 5-fold cross validation
        if fold is not None:
            np.random.seed(42)
            indices = np.random.permutation(len(self.files))
            fold_size = len(self.files) // num_folds
            val_start = fold * fold_size
            val_end = (fold + 1) * fold_size if fold < num_folds - 1 else len(self.files)
            val_indices = set(indices[val_start:val_end].tolist())

            if fold_idx == 'train':
                self.files = [f for i, f in enumerate(self.files) if i not in val_indices]
            else:
                self.files = [f for i, f in enumerate(self.files) if i in val_indices]

        self.augment = augment
        self.cache = cache
        self._cache = {} if cache else None

        print(f'Dataset: {len(self.files)} samples (augment={augment})')

    def __len__(self):
        return len(self.files)

    def _load(self, idx):
        path = self.files[idx]
        if self.cache and path in self._cache:
            return self._cache[path]

        data = torch.load(path, weights_only=True)

        feat = data['features']

        if feat.dim() == 4:
            # Format (3, 4, H, W): [pre1, pre2, post] with [VV, VH, DEM, zeros]
            sar = torch.cat([feat[1, 0:2], feat[2, 0:2]], dim=0)  # (4, H, W)
            dem_raw = feat[2, 2:3]  # (1, H, W)
        else:
            sar = feat[0:4]
            dem_raw = feat[4:5]

        # Normalize SAR
        sar = torch.nan_to_num(sar, nan=0.0, posinf=0.0, neginf=-50.0)
        sar = torch.clamp(sar, -30.0, 5.0)
        sar = (sar - (-12.5)) / 17.5

        # DEM
        dem = torch.clamp(dem_raw, -50.0, 3000.0)
        dem = (dem - (-50.0)) / 3050.0

        # HAND, Slope, Flow direction, Flow accumulation
        hand_raw = data['hand'].unsqueeze(0)
        hand = torch.clamp(hand_raw, 0.0, 50.0) / 50.0

        slope = data['slope'].unsqueeze(0)
        flow_dir = data.get('flow_dir', torch.zeros_like(dem)).float()
        if flow_dir.max() > 1.5:
            flow_dir = flow_dir / 8.0
        flow_dir = flow_dir.unsqueeze(0)
        flow_acc = data.get('flow_acc', torch.zeros_like(dem)).unsqueeze(0)

        # Mask: flood class (value 2 in Kuro Siwo labels: 0=NW, 1=PW, 2=Flood)
        mask = (data['raw_label'] == 2.0).float()[:1]
        mask = torch.nan_to_num(mask, nan=0.0)

        # Full 3-class label for BlackBench-compatible evaluation
        label_3class = data['raw_label'].long().clamp_(0, 2)  # 0=NW, 1=PW, 2=Flood

        if self.cache:
            self._cache[path] = (sar, dem, hand, slope, dem_raw, flow_dir, flow_acc, mask, label_3class)

        return sar, dem, hand, slope, dem_raw, flow_dir, flow_acc, mask, label_3class

    def __getitem__(self, idx):
        sar, dem, hand, slope, dem_raw, flow_dir, flow_acc, mask, label_3class = self._load(idx)

        if self.augment:
            # All channels: sar(4) + dem(1) + hand(1) + slope(1) + flow_dir(1) + flow_acc(1) = 9
            cat_img = torch.cat([sar, dem, hand, slope, flow_dir, flow_acc], dim=0)
            img_np = cat_img.permute(1, 2, 0).numpy()
            mask_np = mask[0].numpy()

            augmented = train_aug(image=img_np, mask=mask_np)
            img_t = torch.from_numpy(augmented['image']).permute(2, 0, 1).float()
            mask = torch.from_numpy(augmented['mask']).unsqueeze(0).float()

            # Slice back
            sar = img_t[0:4]
            dem = img_t[4:5]
            hand = img_t[5:6]
            slope = img_t[6:7]
            flow_dir = img_t[7:8]
            flow_acc = img_t[8:9]

        # Raw hand for HVR metric
        raw_hand = hand * 50.0

        return sar, dem, hand, slope, dem_raw, flow_dir, flow_acc, mask, raw_hand, label_3class


def ensure_data(data_dir):
    """Auto-preprocess data from tortilla if .pt files don't exist."""
    val_pt = os.path.join(data_dir, 'sample_0.pt')
    if os.path.exists(val_pt):
        print(f'Preprocessed data found at {data_dir}')
        return True

    print(f'No .pt files in {data_dir}. Searching for tortilla...')
    try:
        import zipfile, rasterio
        from scipy.ndimage import minimum_filter
        from tqdm import tqdm
    except ImportError:
        print('Missing rasterio or tqdm. Cannot preprocess.')
        return False

    # Find tortilla
    known = [
        '/kaggle/input/datasets/piyushdotcom/my-kuro-siwo-raw/kuro_siwo/geobench_kuro_siwo.tortilla',
        '/kaggle/input/my-kuro-siwo-raw/kuro_siwo/geobench_kuro_siwo.tortilla',
        '/kaggle/working/kuro_siwo/geobench_kuro_siwo.tortilla',
    ]
    tp = None
    for p in known:
        if os.path.exists(p):
            tp = p
            break
    if tp is None:
        for sd in ['/kaggle/input', '/kaggle/working']:
            if os.path.exists(sd):
                for root, dirs, files in os.walk(sd):
                    for f in files:
                        if f.endswith('.tortilla'):
                            tp = os.path.join(root, f)
                            break
                    if tp: break
            if tp: break

    if tp is None:
        print('No tortilla file found. Add Kuro Siwo dataset.')
        return False

    print(f'Found: {tp}')
    try:
        import tacoreader.v1 as tr
    except ImportError:
        import subprocess, sys
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'tacoreader'])
        import tacoreader.v1 as tr

    df = tr.load(tp)
    total_samples = 0
    from concurrent.futures import ProcessPoolExecutor, as_completed

    def _process_one(args):
        ix, split_type = args
        try:
            import rasterio
            import numpy as np
            from scipy.ndimage import minimum_filter
            import tacoreader.v1 as _tr
            _df = _tr.load(tp)
            sub = _df.read(ix)
            sh = (224, 224)
            for eid in ['pre_event_1', 'pre_event_2', 'post_event', 'dem', 'mask']:
                r = sub[sub['tortilla:id'] == eid]
                if len(r):
                    with rasterio.open(r.iloc[0]['internal:subfile']) as s:
                        sh = (s.height, s.width)
                    break
            seq = []
            for eid in ['pre_event_1', 'pre_event_2', 'post_event']:
                r = sub[sub['tortilla:id'] == eid]
                seq.append(rasterio.open(r.iloc[0]['internal:subfile']).read() if len(r) else np.zeros((2, *sh)))
            img = np.nan_to_num(np.stack(seq), nan=1e-5)
            sar = 10 * np.log10(np.clip(img, 1e-5, None))
            dr = sub[sub['tortilla:id'] == 'dem']
            dem = np.nan_to_num(rasterio.open(dr.iloc[0]['internal:subfile']).read(1) if len(dr) else np.zeros(sh), nan=0.0)
            dem_stack = np.repeat(np.stack([dem, np.zeros_like(dem)])[None], 3, 0)
            feat = np.concatenate([sar, dem_stack.reshape(3, 2, *sh)], 1)
            mr = sub[sub['tortilla:id'] == 'mask']
            lbl = np.nan_to_num(rasterio.open(mr.iloc[0]['internal:subfile']).read() if len(mr) else np.zeros((1, *sh)), nan=0.0)
            lmin = minimum_filter(dem, size=15)
            hand = (dem - lmin).astype(np.float32)
            gy, gx = np.gradient(dem)
            slope = np.clip(np.degrees(np.arctan(np.sqrt(gx**2 + gy**2))) / 45.0, 0, 1).astype(np.float32)
            fd, fa = compute_d8_flow(dem)
            fd = (fd / 8.0).astype(np.float32)
            fa_log = np.log1p(fa).astype(np.float32)
            fa = (fa_log / (fa_log.max() + 1e-6)).astype(np.float32)
            return 1, {
                'features': torch.from_numpy(feat).float(),
                'raw_label': torch.from_numpy(lbl).float(),
                'hand': torch.from_numpy(hand),
                'slope': torch.from_numpy(slope),
                'dem': torch.from_numpy(dem.astype(np.float32)),
                'flow_dir': torch.from_numpy(fd),
                'flow_acc': torch.from_numpy(fa),
            }, ix
        except Exception as e:
            return 0, str(e), ix

    for split, subdir in [('train', 'train'), ('validation', 'val')]:
        idxs = [i for i, r in df.iterrows() if r['tortilla:data_split'] == split]
        print(f'Processing {len(idxs)} {split} with {min(os.cpu_count(), 8)} workers...')
        os.makedirs(data_dir, exist_ok=True)
        with ProcessPoolExecutor(max_workers=min(os.cpu_count(), 8)) as pool:
            futures = {pool.submit(_process_one, (ix, split)): ix for ix in idxs}
            for f in tqdm(as_completed(futures), total=len(idxs), desc=split):
                status, result, ix = f.result()
                if status:
                    torch.save(result, f'{data_dir}/sample_{total_samples}_{ix}.pt')
                    total_samples += 1
                else:
                    print(f'  Skipped sample {ix}: {result}')
    print(f'Preprocessing complete! {total_samples} samples saved to {data_dir}')
    return True


def create_dataloaders(data_dir, batch_size=8, num_workers=2, fold=None, pin_memory=True):
    """Create train and validation dataloaders for a given fold."""
    train_ds = FloodDataset(data_dir, augment=True, fold=fold, fold_idx='train')
    val_ds = FloodDataset(data_dir, augment=False, fold=fold, fold_idx='val')

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, drop_last=True, pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size // 2, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory
    )

    return train_loader, val_loader
