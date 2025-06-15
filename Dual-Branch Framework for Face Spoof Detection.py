
import os
import torch
import timm
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from PIL import Image
import numpy as np
import cv2
from tqdm import tqdm
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
import pandas as pd
import matplotlib.pyplot as plt

# Spatial Attention Module
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return torch.sigmoid(x)

# CNN Feature Extractor with Spatial Attention
class CNNFeatureExtractor(nn.Module):
    def __init__(self, use_attention=True):
        super(CNNFeatureExtractor, self).__init__()
        self.model = timm.create_model("efficientnet_b0", pretrained=True)
        self.attention = SpatialAttention() if use_attention else None

    def forward(self, x):
        feat = self.model.forward_features(x)  # (B, 1280, 7, 7)
        if self.attention is not None:
            attention_map = self.attention(feat)  # (B, 1, 7, 7)
            feat = feat * attention_map  # (B, 1280, 7, 7)
        pooled = feat.mean(dim=[2, 3])  # (B, 1280)
        return pooled

# ELBP + FFT Feature Extractor
def extract_ELBP_FFT(image, use_elbp=True, use_fft=True):
    image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    combined = []
    if use_elbp:
        lbp = cv2.equalizeHist(image_gray)
        lbp_resized = cv2.resize(lbp, (32, 32)).flatten()
        combined.append(lbp_resized)
    if use_fft:
        fft = np.abs(np.fft.fft2(image_gray))
        fft_shifted = np.fft.fftshift(fft)
        fft_resized = cv2.resize(fft_shifted, (32, 32)).flatten()
        combined.append(fft_resized)
    if not combined:
        return np.zeros(2048)  # Default size for compatibility
    combined = np.concatenate(combined)
    return combined / combined.max()  # Max normalization

# Custom Dataset for CASIA-FASD
class CASIAFusionDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

        self.image_paths = []
        self.labels = []

        for label_name, label in zip(['train_release', 'test_release'], [0, 1]):
            label_dir = os.path.join(root_dir, label_name)
            for subdir, _, files in os.walk(label_dir):
                for file in files:
                    if file.endswith((".jpg", ".png")):
                        self.image_paths.append(os.path.join(subdir, file))
                        self.labels.append(label)

        train_paths, temp_paths, train_labels, temp_labels = train_test_split(
            self.image_paths, self.labels, test_size=0.4, stratify=self.labels, random_state=42
        )
        val_paths, test_paths, val_labels, test_labels = train_test_split(
            temp_paths, temp_labels, test_size=0.5, stratify=temp_labels, random_state=42
        )

        if split == 'train':
            self.image_paths, self.labels = train_paths, train_labels
        elif split == 'val':
            self.image_paths, self.labels = val_paths, val_labels
        else:
            self.image_paths, self.labels = test_paths, test_labels

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return None
        image_np = np.array(image)
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        rgb_tensor = self.transform(image)
        elbp_fft_feat = torch.tensor(extract_ELBP_FFT(image_np), dtype=torch.float32)

        return rgb_tensor, elbp_fft_feat, label

# Custom Dataset for MSU-MFSD
class MSUMFSDDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

        self.image_paths = []
        self.labels = []

        for label_name, label in zip(['real', 'attack'], [0, 1]):
            label_dir = os.path.join(root_dir, 'scene01', label_name)
            if not os.path.exists(label_dir):
                raise FileNotFoundError(f"Directory {label_dir} does not exist")
            for subdir, _, files in os.walk(label_dir):
                for file in files:
                    if file.endswith(".jpg"):
                        self.image_paths.append(os.path.join(subdir, file))
                        self.labels.append(label)

        if not self.image_paths:
            raise ValueError("No images found in the dataset")

        train_paths, temp_paths, train_labels, temp_labels = train_test_split(
            self.image_paths, self.labels, test_size=0.4, stratify=self.labels, random_state=42
        )
        val_paths, test_paths, val_labels, test_labels = train_test_split(
            temp_paths, temp_labels, test_size=0.5, stratify=temp_labels, random_state=42
        )

        if split == 'train':
            self.image_paths, self.labels = train_paths, train_labels
        elif split == 'val':
            self.image_paths, self.labels = val_paths, val_labels
        else:
            self.image_paths, self.labels = test_paths, test_labels

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return None
        image_np = np.array(image)
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        rgb_tensor = self.transform(image)
        elbp_fft_feat = torch.tensor(extract_ELBP_FFT(image_np), dtype=torch.float32)

        return rgb_tensor, elbp_fft_feat, label

# Custom Dataset for Replay
class ReplayDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

        self.image_paths = []
        self.labels = []

        split_dir = os.path.join(root_dir, split)
        if not os.path.exists(split_dir):
            raise FileNotFoundError(f"Directory {split_dir} does not exist")

        real_dir = os.path.join(split_dir, 'real')
        if os.path.exists(real_dir):
            for subdir, _, files in os.walk(real_dir):
                for file in files:
                    if file.endswith(".jpg"):
                        self.image_paths.append(os.path.join(subdir, file))
                        self.labels.append(0)

        attack_dir = os.path.join(split_dir, 'attack')
        if os.path.exists(attack_dir):
            for attack_type in ['fixed', 'hand']:
                attack_type_dir = os.path.join(attack_dir, attack_type)
                if os.path.exists(attack_type_dir):
                    for subdir, _, files in os.walk(attack_type_dir):
                        for file in files:
                            if file.endswith(".jpg"):
                                self.image_paths.append(os.path.join(subdir, file))
                                self.labels.append(1)

        if not self.image_paths:
            raise ValueError(f"No images found in {split} split")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return None
        image_np = np.array(image)
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        rgb_tensor = self.transform(image)
        elbp_fft_feat = torch.tensor(extract_ELBP_FFT(image_np), dtype=torch.float32)

        return rgb_tensor, elbp_fft_feat, label

def custom_collate(batch):
    batch = [item for item in batch if item is not None]
    if not batch:
        return None
    rgb, feats, labels = zip(*batch)
    return torch.stack(rgb), torch.stack(feats), torch.stack(labels)

# Dual Branch Model with Configurable Features
class DualBranchDeepFakeDetector(nn.Module):
    def __init__(self, use_rgb=True, use_handcrafted=True, use_elbp=True, use_fft=True, use_attention=True):
        super(DualBranchDeepFakeDetector, self).__init__()
        self.use_rgb = use_rgb
        self.use_handcrafted = use_handcrafted
        self.use_elbp = use_elbp
        self.use_fft = use_fft

        if use_rgb:
            self.rgb_branch = CNNFeatureExtractor(use_attention=use_attention)
        else:
            self.rgb_branch = None

        if use_handcrafted:
            input_dim = (1024 if use_elbp else 0) + (1024 if use_fft else 0)
            input_dim = max(input_dim, 1)  # Ensure non-zero input
            self.feature_branch = nn.Sequential(
                nn.Linear(input_dim, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, 128)
            )
        else:
            self.feature_branch = None

        # Classifier input size
        rgb_dim = 1280 if use_rgb else 0
        feat_dim = 128 if use_handcrafted else 0
        classifier_input = rgb_dim + feat_dim
        classifier_input = max(classifier_input, 1)  # Avoid zero input
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x_rgb, x_feats):
        rgb_out = torch.zeros(x_rgb.size(0), 1280, device=x_rgb.device) if self.use_rgb else None
        feat_out = torch.zeros(x_rgb.size(0), 128, device=x_rgb.device) if self.use_handcrafted else None

        if self.use_rgb and self.rgb_branch is not None:
            rgb_out = self.rgb_branch(x_rgb)

        if self.use_handcrafted and self.feature_branch is not None:
            feat_out = self.feature_branch(x_feats)

        combined = []
        if rgb_out is not None:
            combined.append(rgb_out)
        if feat_out is not None:
            combined.append(feat_out)
        if not combined:
            combined = [torch.zeros(x_rgb.size(0), 1, device=x_rgb.device)]
        combined = torch.cat(combined, dim=1)
        return self.classifier(combined)

# Calculate EER
def calculate_eer(fpr, tpr, thresholds):
    frr = 1 - tpr
    abs_diffs = np.abs(fpr - frr)
    eer_idx = np.argmin(abs_diffs)
    eer = fpr[eer_idx]
    return eer

# Training Function
def train(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    valid_batches = 0
    for batch in tqdm(loader, desc="Training"):
        if batch is None:
            continue
        rgb, features, labels = batch
        rgb, features, labels = rgb.to(device), features.to(device), labels.to(device)

        optimizer.zero_grad()
        output = model(rgb, features)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        valid_batches += 1
    return total_loss / valid_batches if valid_batches > 0 else float('inf')

# Evaluation Function
def evaluate(model, loader, criterion, acc_metric, f1_metric, device, desc="Evaluating"):
    model.eval()
    acc_metric.reset()
    f1_metric.reset()
    total_loss = 0
    valid_batches = 0
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(loader, desc=desc):
            if batch is None:
                continue
            rgb, features, labels = batch
            rgb, features, labels = rgb.to(device), features.to(device), labels.to(device)
            output = model(rgb, features)
            loss = criterion(output, labels)
            total_loss += loss.item()
            valid_batches += 1
            preds = torch.argmax(output, dim=1)
            acc_metric.update(preds, labels)
            f1_metric.update(preds, labels)
            probs = torch.softmax(output, dim=1)[:, 1]
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / valid_batches if valid_batches > 0 else float('inf')
    accuracy = acc_metric.compute().item()
    f1 = f1_metric.compute().item()

    fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)
    eer = calculate_eer(fpr, tpr, thresholds)

    return avg_loss, accuracy, f1, roc_auc, eer

# Main Execution
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    casia_root = "/kaggle/input/camsre/CASIA_FASD_FacesCropped-20240113T133738Z-001/CASIA_FASD_FacesCropped"
    msu_root = "/kaggle/input/camsre/MSU-MFSD_FACES/content/MSU-MFSD_FACES"
    replay_root = "/kaggle/input/camsre/replay_data_every_10th/replay_data_every_10th"

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor()
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # Initialize Datasets
    casia_train = CASIAFusionDataset(casia_root, split='train', transform=train_transform)
    casia_val = CASIAFusionDataset(casia_root, split='val', transform=eval_transform)
    casia_test = CASIAFusionDataset(casia_root, split='test', transform=eval_transform)

    msu_train = MSUMFSDDataset(msu_root, split='train', transform=train_transform)
    msu_val = MSUMFSDDataset(msu_root, split='val', transform=eval_transform)
    msu_test = MSUMFSDDataset(msu_root, split='test', transform=eval_transform)

    replay_train = ReplayDataset(replay_root, split='train', transform=train_transform)
    replay_val = ReplayDataset(replay_root, split='dev', transform=eval_transform)
    replay_test = ReplayDataset(replay_root, split='test', transform=eval_transform)

    # Combine Datasets
    train_dataset = ConcatDataset([casia_train, msu_train, replay_train])
    val_dataset = ConcatDataset([casia_val, msu_val, replay_val])
    test_dataset = ConcatDataset([casia_test, msu_test, replay_test])

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0, collate_fn=custom_collate)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0, collate_fn=custom_collate)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0, collate_fn=custom_collate)

    # Ablation Configurations
    configs = [
        {"name": "RGB_only", "use_rgb": True, "use_handcrafted": False, "use_elbp": False, "use_fft": False, "use_attention": True},
        {"name": "Handcrafted_only", "use_rgb": False, "use_handcrafted": True, "use_elbp": True, "use_fft": True, "use_attention": False},
        {"name": "RGB+Handcrafted", "use_rgb": True, "use_handcrafted": True, "use_elbp": True, "use_fft": True, "use_attention": True},
        {"name": "RGB+ELBP+Attention", "use_rgb": True, "use_handcrafted": True, "use_elbp": True, "use_fft": False, "use_attention": True},
        {"name": "RGB+FFT+Attention", "use_rgb": True, "use_handcrafted": True, "use_elbp": False, "use_fft": True, "use_attention": True},
    ]

    results = []
    criterion = CrossEntropyLoss()
    acc_metric = BinaryAccuracy().to(device)
    f1_metric = BinaryF1Score().to(device)

    for config in configs:
        print(f"\nTraining {config['name']}...")
        model = DualBranchDeepFakeDetector(
            use_rgb=config['use_rgb'],
            use_handcrafted=config['use_handcrafted'],
            use_elbp=config['use_elbp'],
            use_fft=config['use_fft'],
            use_attention=config['use_attention']
        ).to(device)
        optimizer = Adam(model.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)

        best_val_f1 = 0
        patience = 5
        no_improve = 0
        for epoch in range(50):  # Max epochs, early stopping will halt earlier
            train_loss = train(model, train_loader, optimizer, criterion, device)
            val_loss, val_acc, val_f1, val_auc, val_eer = evaluate(model, val_loader, criterion, acc_metric, f1_metric, device, desc="Validating")
            print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f} | Val AUC: {val_auc:.4f} | Val EER: {val_eer:.4f}")
            scheduler.step(val_loss)
            if val_f1 > best_val_f1 + 0.001:
                best_val_f1 = val_f1
                no_improve = 0
                torch.save(model.state_dict(), f"{config['name']}.pth")
                print(f"Saved best model: {config['name']}.pth")
            else:
                no_improve += 1
            if no_improve >= patience:
                print("Early stopping triggered")
                break

        model.load_state_dict(torch.load(f"{config['name']}.pth", map_location=device))
        test_loss, test_acc, test_f1, test_auc, test_eer = evaluate(model, test_loader, criterion, acc_metric, f1_metric, device, desc=f"Testing {config['name']}")
        print(f"{config['name']} Test Results: Loss: {test_loss:.4f} | Acc: {test_acc:.4f} | F1: {test_f1:.4f} | AUC: {test_auc:.4f} | EER: {test_eer:.4f}")

        results.append({
            "Configuration": config['name'],
            "Test_Accuracy": test_acc,
            "Test_F1": test_f1,
            "Test_AUC": test_auc,
            "Test_EER": test_eer
        })

    # Save Results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv("ablation_study_results.csv", index=False)
    print("\nAblation study results saved to 'ablation_study_results.csv'")