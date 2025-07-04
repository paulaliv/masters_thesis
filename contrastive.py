import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
from sklearn.metrics import classification_report, confusion_matrix
import copy
from monai.metrics import ConfusionMatrixMetric
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from torch.nn.functional import cross_entropy
from torch.utils.data import DataLoader, random_split,ConcatDataset
from monai.data import pad_list_data_collate
from torch.utils.data import Dataset
import torch.optim as optim
import numpy as np
import pandas as pd
import sys
import os
from torch.cuda.amp import GradScaler, autocast
import matplotlib.pyplot as plt
import umap
from sklearn.metrics.pairwise import rbf_kernel
from monai.data import Dataset, DataLoader
from scipy.spatial import distance
import seaborn as sns
from monai.losses import FocalLoss

from monai.networks.nets import DenseNet121, DenseNet169

from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, RandFlipd, RandRotate90d, RandGaussianNoised, NormalizeIntensityd,
    ToTensord
)
train_transforms = Compose([
    # Don't use LoadImaged since data is already loaded
    EnsureChannelFirstd(keys=["image"],channel_dim=0),
    NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    RandFlipd(keys=["image"], prob=0.5, spatial_axis=0),
    RandRotate90d(keys=["image"], prob=0.5, max_k=3),
    RandGaussianNoised(keys="image", prob=0.2),
    ToTensord(keys=["image"]),
])

val_transforms = Compose([
    EnsureChannelFirstd(keys=['image'],channel_dim=0),
    NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    ToTensord(keys=['image'])
])



os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

tumor_to_idx = {
    "MyxofibroSarcomas": 0,
    "LeiomyoSarcomas": 1,
    "DTF": 2,
    "MyxoidlipoSarcoma": 3,
    "WDLPS": 4,
}



class TumorClassifier(nn.Module):
    def __init__(self, model_depth=18,in_channels=1, num_classes=5):  # change num_classes to match your setting
        super().__init__()
        # Feature extractor
        # self.encoder = ResNet(
        #     block="basic",  # "BASIC" for ResNet18/34, "BOTTLE" for ResNet50+
        #     layers=[2, 2, 2, 2],  # ResNet18 config
        #     n_input_channels=in_channels,
        #     num_classes=5,
        #     feed_forward=False,
        #     block_inplanes=[64, 128, 256, 512]
        #
        #     # we'll define our own classifier head
        # )
        self.encoder = DenseNet121(
            spatial_dims=3,  # 3D input
            in_channels=in_channels,
            out_channels=512  # number of features before classifier head
        )
        #for feature extraction
        self.pool = nn.AdaptiveAvgPool3d(1)

        self.embedding_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        #self.classifier = nn.Linear(128, num_classes)


    def forward(self, x):
        x = self.encoder(x)
        embeddings = self.embedding_head(x)
        #logits = self.classifier(embeddings)

        return embeddings

    def extract_encoder_features(self, x):
        x = self.encoder(x)
        print(f'shape encoder output: {x.shape}')

        #embeddings = self.embedding_head(x)
        return x

class QADataset(Dataset):
    def __init__(self, fold, preprocessed_dir, fold_paths, transform=None):
        """
        fold: str, e.g. 'fold_0'
        preprocessed_dir: base preprocessed path with .npz images
        pred_fold_paths: dict with predicted mask folder paths per fold
        fold_paths: dict with fold folder paths containing Dice scores & case IDs
        """
        self.fold = fold
        self.preprocessed_dir = preprocessed_dir
        self.fold_dir = fold_paths[fold]


        # Load Dice scores & case IDs from a CSV or JSON
        # Example CSV: case_id,dice

        self.metadata = pd.read_csv(self.fold_dir)

        # List of case_ids
        self.case_ids = self.metadata['case_id'].tolist()
        #self.dice_scores = self.metadata['dice'].tolist()
        self.subtypes = self.metadata['subtype'].tolist()
        #self.ds = nnUNetDatasetBlosc2(self.preprocessed_dir)

        self.transform = transform

        self.tumor_to_idx = tumor_to_idx

    def __len__(self):
        return len(self.case_ids)

    def __getitem__(self, idx):
        case_id = self.case_ids[idx]
        #dice_score = self.dice_scores[idx]
        subtype = self.subtypes[idx]
        subtype = subtype.strip()

        label_idx = self.tumor_to_idx[subtype]
        # Load preprocessed image (.npz)
        #Check if its not case_id_0000
        # data, seg, seg_prev, properties = self.ds.load_case(case_id)
        #print("Data shape:", data.shape)
        file = f'{case_id}_resized.pt'
        image = torch.load(os.path.join(self.preprocessed_dir, file))

        # file = f'{case_id}_features_roi.npz'
        # feat1 = np.load(os.path.join(self.preprocessed_dir, file))
        # image = feat1[feat1.files[0]]


        assert image.ndim == 4 and image.shape[0] == 1, f"Expected shape (1, H, W, D), but got {image.shape}"
        #image = np.asarray(image)
        #print(f'Image Shape {image.shape}')

        data_dict = {'image': image }
        # nnU-Net raw images usually have multiple channels; choose accordingly:
        # Here, just take channel 0 for simplicity:
        #input_image = np.stack([image[0], pred_mask], axis=0)  # (2, H, W, D)
        # Map dice score to category
        if self.transform:
            data_dict = self.transform(data_dict)

        image_tensor = data_dict['image']
        label_tensor = torch.tensor(label_idx).long()


        # print('Image tensor shape : ', image_tensor.shape)
        # print('Label tensor shape : ', label_tensor.shape)

        return {
            'input': image_tensor,  # shape (C_total, D, H, W)
            'label': label_tensor,  # scalar tensor
        }



def supervised_contrastive_loss(embeddings, labels, temperature):
    """
       embeddings: (batch_size, embedding_dim)
       labels: (batch_size,)
       """
    device = embeddings.device
    batch_size = embeddings.shape[0]
    labels = labels.contiguous().view(-1, 1)  # (B,1)

    mask = torch.eq(labels, labels.T).float().to(device)  # (B, B)

    embeddings = F.normalize(embeddings, dim=1)

    similarity_matrix = torch.matmul(embeddings, embeddings.T)  # (B, B)
    logits = similarity_matrix / temperature

    # For numerical stability
    logits_max, _ = torch.max(logits, dim=1, keepdim=True)
    logits = logits - logits_max.detach()

    # Mask self-contrast cases
    mask_self = torch.eye(batch_size, dtype=torch.bool).to(device)
    mask = mask * (~mask_self).float()

    # Compute log_prob
    exp_logits = torch.exp(logits) * (~mask_self).float()
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

    mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-12)

    loss = -mean_log_prob_pos
    loss = loss.mean()

    return loss

def train(model, train_dataset,plot_dir, optimizer, scheduler, num_epochs, rank, world_size, device,):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')
    patience_counter = 0

    from torch.utils.data.distributed import DistributedSampler
    if world_size > 1:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        shuffle = False  # sampler handles shuffling
    else:
        train_sampler = None
        shuffle = True



    train_loader = DataLoader(train_dataset, batch_size=12, sampler=train_sampler, num_workers=4,
                              pin_memory=True,collate_fn=pad_list_data_collate)

    #train_loader = DataLoader(train_dataset, batch_size=24, shuffle=True, pin_memory=True, num_workers=4, collate_fn=pad_list_data_collate)

    train_losses = []  # <-- add here, before the epoch loop

    for epoch in range(num_epochs):
        model.train()
        print(f"Epoch {epoch+1}/{num_epochs}")
        running_loss, correct, total = 0.0, 0, 0

        scaler = GradScaler()

        for batch in train_loader:
            inputs = batch['input'].to(device)
            labels = batch['label'].to(device)
            #print("Input shape:", inputs.shape)

            optimizer.zero_grad()
            with autocast():
                embeddings = model(inputs)

                loss = supervised_contrastive_loss(embeddings, labels, temperature = 0.1)

                labels_cpu = labels.detach().cpu()


            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * inputs.size(0)
            total += labels.size(0)

            if epoch % 10 == 0:
                #create UMAP
                pass

        epoch_train_loss = running_loss / total
        print(f"Train Loss: {epoch_train_loss:.4f}")

        train_losses.append(epoch_train_loss)

        del inputs,embeddings,labels
        torch.cuda.empty_cache()


    return model, train_losses

def plot_UMAP(train, y_train, neighbours, m, name, image_dir):
    print(f'feature shape {train.shape}')

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=neighbours,
        min_dist=0.1,
        metric=m,
        random_state=42
    )
    train_umap = reducer.fit_transform(train)  # (N, 2)
    # Apply UMAP transform to validation data
    # val_umap = reducer.transform(val)

    # Map labels back to names (optional)
    idx_to_tumor = {v: k for k, v in tumor_to_idx.items()}

    # all_subtypes= np.concatenate([y_train, y_val])
    unique_subtypes = sorted(set(y_train))

    # labels = np.array(['train'] * len(train_umap) + ['val'] * len(val_umap))
    markers = {'train': 'o', 'val': 's'}

    cmap = plt.cm.tab20
    color_lookup = {lab: cmap(i % 20) for i, lab in enumerate(unique_subtypes)}
    # 7. scatter plot
    plt.figure(figsize=(8, 6))
    # for marker_type in ['train', 'val']:
    for subtype in unique_subtypes:
        idx = [i for i, lab in enumerate(y_train) if lab == subtype]
        if not idx: continue
        plt.scatter(
            train_umap[idx, 0], train_umap[idx, 1],
            s=25,
            c=[color_lookup[subtype]] * len(idx),
            label=f"{idx_to_tumor[subtype]} ",
            alpha=0.8,
        )

    plt.xlabel("UMAP‑1")
    plt.ylabel("UMAP‑2")
    plt.title("ROI Feature Map Clusters by Subtype and Set")
    plt.legend(fontsize=8, loc='best', markerscale=1)
    plt.tight_layout()
    image_loc = os.path.join(image_dir, name)
    plt.savefig(image_loc, dpi=300)
    plt.show()

def intra_class_distance(X_train, y_train):

    #Intra-Class distance
    intra_class_dists_maha = {}
    intra_class_dists_euc = {}
    std_maha = {}
    std_euc = {}

    for subtype in np.unique(y_train):
        features = X_train[y_train == subtype]
        mean_vec = features.mean(axis=0)

        # Mahalanobis setup
        cov = np.cov(features.T)
        inv_covmat = np.linalg.pinv(cov)

        mahalanobis = [distance.mahalanobis(f, mean_vec, inv_covmat) for f in features]
        euclidean = np.linalg.norm(features - mean_vec, axis=1)

        intra_class_dists_maha[subtype] = np.mean(mahalanobis)
        intra_class_dists_euc[subtype] = np.mean(euclidean)

        std_maha[subtype] = np.std(mahalanobis)
        std_euc[subtype] = np.std(euclidean)

    return intra_class_dists_maha, intra_class_dists_euc, std_maha, std_euc

def plot_intra_class_distances(intra_class_dists_maha, intra_class_dists_euc, std_maha, std_euc,plot_dir, tag):
    subtypes = list(intra_class_dists_maha.keys())
    maha_values = [intra_class_dists_maha[sub] for sub in subtypes]
    euc_values = [intra_class_dists_euc[sub] for sub in subtypes]
    maha_err = [std_maha[sub] for sub in subtypes]
    euc_err = [std_euc[sub] for sub in subtypes]

    x = np.arange(len(subtypes))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width/2, maha_values, width, yerr=maha_err, label='Mahalanobis', color='steelblue', capsize=5)
    ax.bar(x + width/2, euc_values, width, yerr=euc_err, label='Euclidean', color='orange', capsize=5)


    ax.set_ylabel("Average Intra-Class Distance")
    ax.set_xlabel("Subtype")
    ax.set_title("Intra-Class Distance per Tumor Subtype (with Std)")
    ax.set_xticks(x)

    ax.set_xticklabels(subtypes, rotation=45)
    ax.legend()
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f'{tag}_intra_class_distances.png'))
    plt.close()

def compute_mmd(x, y, gamma=None):
    """
    Compute the Maximum Mean Discrepancy (MMD) between two samples x and y using RBF kernel.
    Args:
        x: numpy array, shape (n_samples_x, n_features)
        y: numpy array, shape (n_samples_y, n_features)
        gamma: float or None, kernel parameter for RBF. If None, 1/n_features is used.
    Returns:
        mmd: float, squared MMD value between distributions of x and y
    """
    if gamma is None:
        gamma = 1.0 / x.shape[1]

    Kxx = rbf_kernel(x, x, gamma=gamma)
    Kyy = rbf_kernel(y, y, gamma=gamma)
    Kxy = rbf_kernel(x, y, gamma=gamma)

    mmd = Kxx.mean() + Kyy.mean() - 2 * Kxy.mean()
    return mmd

def inter_class_distance(X_train, y_train, plot_dir, tag):
    """
      Compute the Maximum Mean Discrepancy (MMD) between two samples x and y using RBF kernel.
      Args:
          x: numpy array, shape (n_samples_x, n_features)
          y: numpy array, shape (n_samples_y, n_features)
          gamma: float or None, kernel parameter for RBF. If None, 1/n_features is used.
      Returns:
          mmd: float, squared MMD value between distributions of x and y
      """
    sorted_tumors = sorted(tumor_to_idx.items(), key=lambda x: x[1])
    unique_subtypes = [tumor for tumor, _ in sorted_tumors]
    idx_to_tumor = {v: k for k, v in tumor_to_idx.items()}

    mmd_matrix = np.zeros((len(unique_subtypes), len(unique_subtypes)))

    # for i, subtype_i in enumerate(unique_subtypes):
    #     xi = X_train[y_train == subtype_i]
    #     for j, subtype_j in enumerate(unique_subtypes):
    #         xj = X_train[y_train == subtype_j]
    #         mmd_matrix[i, j] = compute_mmd(xi, xj)
    #
    for i, (_, idx_i) in enumerate(sorted_tumors):
        xi = X_train[y_train == idx_i]
        for j, (_, idx_j) in enumerate(sorted_tumors):
            xj = X_train[y_train == idx_j]
            mmd_matrix[i, j] = compute_mmd(xi, xj)

    # Create pretty labels with names and indices
    pretty_labels = [f"{tumor} ({tumor_to_idx[tumor]})" for tumor in unique_subtypes]#

    # Plot heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(mmd_matrix, xticklabels=pretty_labels, yticklabels=pretty_labels,
                cmap="viridis", annot=True, fmt=".3f")
    plt.title("MMD Distance Matrix Between Tumor Subtypes")
    plt.xlabel("Subtype")
    plt.ylabel("Subtype")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f'{tag}_MMD_distance.png'))
    plt.close()

    return mmd_matrix


def main(preprocessed_dir, plot_dir, fold_paths, world_size, rank, device):
    model = TumorClassifier(model_depth=18, in_channels=1, num_classes=5)
    model.to(device)

    # Combine training folds datasets
    train_fold_ids = [f"fold_{i}" for i in range(5)]
    train_datasets = []
    for train_fold in train_fold_ids:
        ds = QADataset(
            fold=train_fold,
            preprocessed_dir=preprocessed_dir,
            fold_paths=fold_paths,
            transform=train_transforms

        )
        train_datasets.append(ds)
    train_dataset = ConcatDataset(train_datasets)
    del train_datasets

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5,min_lr=1e-6)

    best_model, train_losses = train(
        model=model,
        train_dataset=train_dataset,
        plot_dir=plot_dir,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=50,
        rank=rank,
        world_size=world_size,
        device=device
    )


def extract_features(train_dir, fold_paths, device, plot_dir, trained = False):
    model = TumorClassifier(model_depth=18, in_channels=1, num_classes=5)
    #model.load_state_dict(torch.load("best_model_fold_0.pth", map_location=device))
    model.to(device)
    model.eval()
    # Combine training folds datasets
    train_fold_ids = [f"fold_{i}" for i in range(5)]
    train_datasets = []
    for train_fold in train_fold_ids:
        ds = QADataset(
            fold=train_fold,
            preprocessed_dir=train_dir,
            fold_paths=fold_paths,
        )
        train_datasets.append(ds)
    train_dataset = ConcatDataset(train_datasets)
    del train_datasets


    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=False, num_workers=4)


    all_features_train = []
    all_labels_train = []


    with torch.no_grad():
        for batch in train_loader:
            inputs = batch['input'].to(device)  # (B, C, D, H, W)
            labels = batch['label'].cpu().numpy()  # class indices

            features = model.extract_encoder_features(inputs).cpu().numpy()  # (B, 512)
            all_features_train.append(features)
            all_labels_train.extend(labels)

    # Combine into arrays
    X_train = np.concatenate(all_features_train, axis=0)
    y_train = np.array(all_labels_train)

    from sklearn.preprocessing import StandardScaler
    X_scaled = StandardScaler().fit_transform(X_train)

    if trained == True:
        tag = 'AFTER'
    else:
        tag = 'BEFORE'

    plot_UMAP(X_train, y_train, neighbours=5, m='cosine', name=f'{tag}_UMAP_cosine_5n_fold0.png', image_dir=plot_dir)
    plot_UMAP(X_train,y_train,neighbours=10, m='cosine', name=f'{tag}_UMAP_cosine_10n_fold0.png', image_dir =plot_dir)
    plot_UMAP(X_train, y_train, neighbours=15, m='cosine', name=f'{tag}_UMAP_cosine_15n_fold0.png', image_dir=plot_dir)


    # plot_UMAP(X_scaled, y_train, neighbours=5, m='manhattan', name='UMAP_manh_5n_fold0.png', image_dir=plot_dir)
    # plot_UMAP(X_scaled,y_train,neighbours=10, m='manhattan', name='UMAP_manh_10n_fold0.png', image_dir =plot_dir)
    # #plot_UMAP(X_scaled, y_train, neighbours=15, m='manhattan', name='UMAP_manh_15n_fold0.png', image_dir=plot_dir)

    maha, euc, std_maha, std_euc = intra_class_distance(X_scaled, y_train)
    plot_intra_class_distances(maha,euc, std_maha, std_euc,plot_dir, tag)
    mmd_matrix = inter_class_distance(X_scaled, y_train, plot_dir, tag)


def extract_latent_features(train_dir, fold_paths, device, plot_dir, trained = False):
    model = TumorClassifier(model_depth=18, in_channels=1, num_classes=5)
    #model.load_state_dict(torch.load("best_model_fold_0.pth", map_location=device))
    model.to(device)
    model.eval()
    # Combine training folds datasets
    train_fold_ids = [f"fold_{i}" for i in range(5)]
    train_datasets = []
    for train_fold in train_fold_ids:
        ds = QADataset(
            fold=train_fold,
            preprocessed_dir=train_dir,
            fold_paths=fold_paths,
        )
        train_datasets.append(ds)
    train_dataset = ConcatDataset(train_datasets)
    del train_datasets


    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=False, num_workers=4)


    all_features_train = []
    all_labels_train = []


    with torch.no_grad():
        for batch in train_loader:
            inputs = batch['input'].to(device)  # (B, C, D, H, W)
            labels = batch['label'].cpu().numpy()  # class indices

            features = model.forward(inputs).cpu().numpy()  # (B, 512)
            all_features_train.append(features)
            all_labels_train.extend(labels)

    # Combine into arrays
    X_train = np.concatenate(all_features_train, axis=0)
    y_train = np.array(all_labels_train)

    from sklearn.preprocessing import StandardScaler
    X_scaled = StandardScaler().fit_transform(X_train)

    if trained == True:
        tag = 'AFTER'
    else:
        tag = 'BEFORE'

    plot_UMAP(X_train, y_train, neighbours=5, m='cosine', name=f'{tag}Latent_UMAP_cosine_5n_fold0.png', image_dir=plot_dir)
    plot_UMAP(X_train,y_train,neighbours=10, m='cosine', name=f'{tag}_Latent_UMAP_cosine_10n_fold0.png', image_dir =plot_dir)
    plot_UMAP(X_train, y_train, neighbours=15, m='cosine', name=f'{tag}_Latent_UMAP_cosine_15n_fold0.png', image_dir=plot_dir)


    # plot_UMAP(X_scaled, y_train, neighbours=5, m='manhattan', name='UMAP_manh_5n_fold0.png', image_dir=plot_dir)
    # plot_UMAP(X_scaled,y_train,neighbours=10, m='manhattan', name='UMAP_manh_10n_fold0.png', image_dir =plot_dir)
    # #plot_UMAP(X_scaled, y_train, neighbours=15, m='manhattan', name='UMAP_manh_15n_fold0.png', image_dir=plot_dir)

    maha, euc, std_maha, std_euc = intra_class_distance(X_scaled, y_train)
    plot_intra_class_distances(maha,euc, std_maha, std_euc,plot_dir, tag)
    mmd_matrix = inter_class_distance(X_scaled, y_train, plot_dir, tag)

import torch.distributed as dist
def setup_device_and_dist():
    if "LOCAL_RANK" in os.environ:
        # torchrun launch
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl')
        world_size = dist.get_world_size()
        print(f"Running distributed on GPU {local_rank} / {world_size}")
        return local_rank, world_size
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        torch.cuda.set_device(device)
        print(f"Running on device: {device}")
        return 0, 1  # local_rank, world_size



if __name__ == '__main__':
    fold_paths = {
        'fold_0': '/gpfs/home6/palfken/masters_thesis/fold_0',
        'fold_1': '/gpfs/home6/palfken/masters_thesis/fold_1',
        'fold_2': '/gpfs/home6/palfken/masters_thesis/fold_2',
        'fold_3': '/gpfs/home6/palfken/masters_thesis/fold_3',
        'fold_4': '/gpfs/home6/palfken/masters_thesis/fold_4',
    }
    preprocessed= sys.argv[1]
    plot_dir = sys.argv[2]



    local_rank, world_size = setup_device_and_dist()
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
    if local_rank == 0:
        extract_features(preprocessed, fold_paths, device=device, plot_dir=plot_dir, trained = False)
        extract_latent_features(preprocessed, fold_paths, device=device, plot_dir=plot_dir, trained = False)
    main(preprocessed, plot_dir, fold_paths,world_size, local_rank,device = device)

    if local_rank == 0:
        extract_features(preprocessed, fold_paths, device=device, plot_dir=plot_dir, trained = True)
        extract_latent_features(preprocessed, fold_paths, device=device, plot_dir=plot_dir, trained=True)


