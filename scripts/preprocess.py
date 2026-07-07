import os
import numpy as np
import torch
import rasterio
from tqdm import tqdm

def _sar_to_db(sar_array, min_db=-30, max_db=0):
    epsilon = 1e-10
    db_array = 10 * np.log10(sar_array ** 2 + epsilon)
    db_array = np.clip(db_array, min_db, max_db)
    norm_array = (db_array - min_db) / (max_db - min_db)
    return norm_array

def _calculate_physics_mask(dem_array):
    gradient_y, gradient_x = np.gradient(dem_array)
    slope = np.sqrt(gradient_x**2 + gradient_y**2)
    slope_norm = np.clip(slope / (np.max(slope) + 1e-8), 0, 1)
    return slope_norm

def preprocess_dataset(data_root="data/kuro_siwo", output_dir="data/processed", max_samples=None):
    """
    Extracts and preprocesses data from the GEO-Bench .tortilla file.
    Saves optimized PyTorch tensors to disk for fast training.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'val'), exist_ok=True)
    
    tortilla_path = os.path.join(data_root, "geobench_kuro_siwo.tortilla")
    if not os.path.exists(tortilla_path):
        raise FileNotFoundError(f"Tortilla file not found at {tortilla_path}")
        
    print(f"Loading tortilla file from {tortilla_path}...")
    import tacoreader
    df = tacoreader.load(tortilla_path)
    
    for split_name, output_subdir in [("train", "train"), ("validation", "val")]:
        indices = [i for i, row in df.iterrows() if row["tortilla:data_split"] == split_name]
        
        if max_samples:
            indices = indices[:max_samples]
            
        print(f"Processing {split_name} split... ({len(indices)} samples)")
        
        for i, sample_idx in enumerate(tqdm(indices)):
            sub_ds = df.read(sample_idx)
            
            # 1. Read SAR sequences
            sar_seq = []
            for event_id in ['pre_event_1', 'pre_event_2', 'post_event']:
                row = sub_ds[sub_ds['tortilla:id'] == event_id]
                if len(row) > 0:
                    path = row.iloc[0]['internal:subfile']
                    with rasterio.open(path) as src:
                        img = src.read() # (2, H, W) -> VV, VH
                        sar_seq.append(img)
                else:
                    sar_seq.append(np.zeros((2, 224, 224), dtype=np.float32))
                    
            raw_sar_seq = np.stack(sar_seq, axis=0).astype(np.float32) # (3, 2, 224, 224)
            raw_sar_seq = np.nan_to_num(raw_sar_seq, nan=0.0, posinf=0.0, neginf=0.0)
            sar_db_seq = _sar_to_db(raw_sar_seq)
            
            # 2. Read DEM and calculate Physics Mask
            dem_row = sub_ds[sub_ds['tortilla:id'] == 'dem']
            if len(dem_row) > 0:
                dem_path = dem_row.iloc[0]['internal:subfile']
                with rasterio.open(dem_path) as src:
                    dem = src.read(1)
            else:
                dem = np.zeros(target_shape, dtype=np.float32)
                
            dem = np.nan_to_num(dem, nan=0.0, posinf=0.0, neginf=0.0)
            physics_feat = _calculate_physics_mask(dem)
            physics_feat = np.nan_to_num(physics_feat, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Expand dims to (1, H, W)
            dem = np.expand_dims(dem, axis=0).astype(np.float32)
            physics_feat = np.expand_dims(physics_feat, axis=0).astype(np.float32)
            
            # Stack DEM and Physics across time
            phys_combined = np.concatenate([dem, physics_feat], axis=0) # (2, H, W)
            phys_expanded = np.repeat(phys_combined[np.newaxis, ...], raw_sar_seq.shape[0], axis=0) # (3, 2, H, W)
            
            # Combine SAR and Physical features
            # (Time, 4, H, W) where Channels = [VV, VH, DEM, Physics]
            features = np.concatenate([sar_db_seq, phys_expanded], axis=1)
            
            # 3. Read Mask
            mask_row = sub_ds[sub_ds['tortilla:id'] == 'mask']
            if len(mask_row) > 0:
                mask_path = mask_row.iloc[0]['internal:subfile']
                with rasterio.open(mask_path) as src:
                    mask = src.read(1)
            else:
                mask = np.zeros((224, 224), dtype=np.float32)
                
            mask = np.nan_to_num(mask, nan=0.0, posinf=0.0, neginf=0.0)
                
            mask = np.expand_dims(mask, axis=0).astype(np.float32) # (1, H, W)
            
            # 4. Save to disk
            tensor_features = torch.from_numpy(features)
            tensor_label = torch.from_numpy(mask)
            
            sample_path = os.path.join(output_dir, output_subdir, f"sample_{i}.pt")
            torch.save({'features': tensor_features, 'label': tensor_label}, sample_path)

if __name__ == "__main__":
    # Process the entire dataset
    preprocess_dataset()
    print("Preprocessing complete!")dims(mask, axis=0).astype(np.float32) # (1, H, W)
            
            # 4. Save to disk
            tensor_features = torch.from_numpy(features)
            tensor_label = torch.from_numpy(mask)
            
            sample_path = os.path.join(output_dir, output_subdir, f"sample_{i}.pt")
            torch.save({'features': tensor_features, 'label': tensor_label}, sample_path)

if __name__ == "__main__":
    # Process the entire dataset
    preprocess_dataset()
    print("Preprocessing complete!")