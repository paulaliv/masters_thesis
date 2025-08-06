Index: dice_regressor_ordinal_mask.py
IDEA additional info:
Subsystem: com.intellij.openapi.diff.impl.patch.CharsetEP
<+>UTF-8
===================================================================
diff --git a/dice_regressor_ordinal_mask.py b/dice_regressor_ordinal_mask.py
--- a/dice_regressor_ordinal_mask.py	
+++ b/dice_regressor_ordinal_mask.py	(date 1754481050864)
@@ -1,4 +1,6 @@
 from collections import defaultdict
+
+from sympy.physics.units import temperature
 from torch.cpu.amp import autocast
 from sklearn.metrics import classification_report
 import collections
@@ -15,7 +17,7 @@
 import time
 import sys
 import json
-from sklearn.metrics import mean_absolute_error
+from sklearn.metrics import accuracy_score, roc_auc_score
 from torch.optim.lr_scheduler import LambdaLR
 from sklearn.metrics import confusion_matrix
 import seaborn as sns
@@ -48,7 +50,7 @@
         keys=["image", "uncertainty"],  # apply same affine to both
         prob=1.0,
         rotate_range=(np.pi/12, np.pi/12, np.pi/12),
-        translate_range=(5, 5, 5),  # in voxels
+        translate_range=(3, 3, 3),  # in voxels
         scale_range=(0.1, 0.1, 0.1),
         spatial_size=(48, 256, 256),
         mode=('trilinear', 'nearest')  # bilinear for image, nearest for uncertainty (categorical or regression)
@@ -177,6 +179,88 @@
 #         x = self.encoder_unc(uncertainty)
 #         return x.view(x.size(0), -1)
 
+import torch
+import torch.nn as nn
+import torch.nn.functional as F
+
+class DistanceAwareCORNLoss(nn.Module):
+    def __init__(self, eps=1e-6, distance_power=0.5):
+        super().__init__()
+        self.eps = eps
+        self.distance_power = distance_power  # use sqrt by default
+
+    def forward(self, logits, labels):
+        """
+        logits: [B, K-1] - logits for ordinal thresholds
+        labels: [B] - integer labels in {0, ..., K-1}
+        """
+        B, K_minus_1 = logits.shape
+
+        # Binary label matrix: y_bin[b, k] = 1 if label[b] > k
+        y_bin = torch.zeros_like(logits, dtype=torch.long)
+        for k in range(K_minus_1):
+            y_bin[:, k] = (labels > k).long()
+
+        # Reshape logits for binary cross-entropy
+        logits_stacked = torch.stack([-logits, logits], dim=2)  # [B, K-1, 2]
+        logits_reshaped = logits_stacked.view(-1, 2)            # [B*(K-1), 2]
+        targets_reshaped = y_bin.view(-1)                       # [B*(K-1)]
+
+        # Compute distance-based weights (e.g., sqrt(|label - k|))
+        label_expanded = labels.unsqueeze(1).expand(-1, K_minus_1)  # [B, K-1]
+        ks = torch.arange(K_minus_1, device=labels.device).unsqueeze(0)  # [1, K-1]
+        distances = torch.abs(label_expanded - ks).float()  # [B, K-1]
+        weights = distances ** self.distance_power          # [B, K-1]
+
+        # Normalize per-sample weights (optional but stabilizing)
+        #weights = weights / (weights.sum(dim=1, keepdim=True) + self.eps)
+
+
+        # Flatten for use in loss
+        weights_flat = weights.view(-1)  # [B*(K-1)]
+
+        # Compute weighted cross-entropy loss
+        loss = F.cross_entropy(logits_reshaped, targets_reshaped, weight=None, reduction='none')
+        loss = (loss * weights_flat).mean()
+
+        return loss
+
+class CORNLoss(nn.Module):
+    """
+    CORN Loss for ordinal regression.
+    """
+    def __init__(self):
+        super(CORNLoss, self).__init__()
+
+    def forward(self, logits, labels):
+        """
+        logits: Tensor of shape (B, K-1), where K is the number of ordinal classes
+        labels: Tensor of shape (B,) with values in {0, ..., K-1}
+        """
+        B, K_minus_1 = logits.shape
+
+        # temperature = 0.5
+        #
+        # logits = logits/temperature
+
+        # Create binary targets: 1 if label > threshold
+        y_bin = torch.zeros_like(logits, dtype=torch.long)
+        for k in range(K_minus_1):
+            y_bin[:, k] = (labels > k).long()
+
+        # Compute softmax over two classes (not raw binary classification)
+        # Each logit becomes a 2-class classification: [P(class <= k), P(class > k)]
+        logits_stacked = torch.stack([-logits, logits], dim=2)  # shape: [B, K-1, 2]
+
+
+
+
+        logits_reshaped = logits_stacked.view(-1, 2)  # [B*(K-1), 2]
+        targets_reshaped = y_bin.view(-1)  # [B*(K-1)]
+
+        loss = F.cross_entropy(logits_reshaped, targets_reshaped, reduction='mean')
+        return loss
+
 class Light3DEncoder(nn.Module):
     def __init__(self):
         super().__init__()
@@ -197,11 +281,24 @@
             nn.AdaptiveAvgPool3d((1, 1, 1)),  # outputs [B, 64, 1, 1, 1]
         )
 
+        # self.encoder = nn.Sequential(
+        #     nn.Conv3d(1, 16, 3, padding=1), nn.BatchNorm3d(16), nn.ReLU(),
+        #     nn.MaxPool3d(2),
+        #     nn.Conv3d(16, 32, 3, padding=1), nn.BatchNorm3d(32), nn.ReLU(),
+        #     nn.MaxPool3d(2),
+        #     nn.Conv3d(32, 64, 3, padding=1), nn.BatchNorm3d(64), nn.ReLU(),
+        #     nn.MaxPool3d(2),  # <- NEW BLOCK
+        #     nn.Conv3d(64, 128, 3, padding=1), nn.BatchNorm3d(128), nn.ReLU(),
+        #     nn.AdaptiveAvgPool3d(1),
+        # )
+        # Output would now be [B, 128]
+
     def forward(self, x):
         x = self.encoder(x)
         return x.view(x.size(0), -1)  # Flatten to [B, 64]
 
 
+
 class QAModel(nn.Module):
     def __init__(self,num_thresholds):
         super().__init__()
@@ -215,6 +312,7 @@
             nn.Flatten(),
             nn.Linear(128, 64),
             nn.ReLU(),
+            nn.Dropout(0.2),
             nn.Linear(64, num_thresholds)  # Output = predicted Dice class
         )
         #self.biases = nn.Parameter(torch.zeros(num_thresholds))
@@ -246,6 +344,17 @@
     return np.digitize(dice_adjusted, bin_edges, right=True)  # right=True = (a <= x)
 
 
+def compute_class_distribution(labels):
+    """
+    labels: list or 1D numpy array or tensor of ordinal class labels (e.g., from 0 to 3)
+    """
+    counter = Counter(labels)
+    total = sum(counter.values())
+    distribution = {k: v / total for k, v in sorted(counter.items())}
+    return distribution
+
+
+
 class QADataset(Dataset):
     def __init__(self, case_ids, data_dir, df, uncertainty_metric,transform=None, want_features = False):
         """
@@ -289,31 +398,28 @@
         image = np.load(os.path.join(self.data_dir, f'{case_id}_pred.npy'))
         image = torch.from_numpy(image).float()
 
+        if image.ndim == 3:
+            image = image.unsqueeze(0)  # Add channel dim
+
+        assert image.ndim == 4 and image.shape[0] == 1, f"Expected shape (1, H, W, D), but got {image.shape}"
 
         uncertainty = np.load(os.path.join(self.data_dir, f'{case_id}_{self.uncertainty_metric}.npy'))
 
         if uncertainty.sum() == 0:
             print(f'{case_id} has empty map!')
 
-        # Convert to torch and ensure shape [1, D, H, W]
-        if image.ndim == 3:
-            image = torch.from_numpy(image).float().unsqueeze(0)
-        else:
-            image = torch.from_numpy(image).float()
-            assert image.ndim == 4, f"Expected image to have 4 dims (C, D, H, W), got {image.shape}"
-
-        if uncertainty.ndim == 3:
-            uncertainty_tensor = torch.from_numpy(uncertainty).float().unsqueeze(0)
-        else:
-            uncertainty_tensor = torch.from_numpy(uncertainty).float()
-            assert uncertainty_tensor.ndim == 4, f"Expected uncertainty to have 4 dims (C, D, H, W), got {uncertainty_tensor.shape}"
-
-
         # Map dice score to category
 
         label = bin_dice_score(dice_score)
 
 
+        #image_tensor = torch.from_numpy(image).float()
+
+        # print(f'Image shape {image.shape}')
+        uncertainty_tensor = torch.from_numpy(uncertainty).float()
+
+        uncertainty_tensor = uncertainty_tensor.unsqueeze(0)  # Add channel dim
+
         label_tensor = torch.tensor(label).long()
 
         if self.transform:
@@ -366,6 +472,12 @@
 #try: use argmax on cumulative logits
 #or logit based decoding
 
+# logit based decofing: (logits > 0).sum(dim=1)If all 3 logits > 0 → class 3
+# If first 2 > 0, last ≤ 0 → class 2
+# If only 1st > 0 → class 1
+# If all ≤ 0 → class 0
+#
+# ✅ This works well but relies heavily on raw logit thresholds at zero, which may be unstable or biased.
 def decode_predictions(logits):
     # print("Logits mean:", logits.mean().item())
     # print("Logits min/max:", logits.min().item(), logits.max().item())
@@ -373,7 +485,44 @@
 
     return (logits > 0).sum(dim=1)
 
+@torch.no_grad()
+def corn_predict(logits):
+    #logits shape: [B, num_thresholds]
+    probs = torch.stack([-logits, logits], dim=2)  # shape: [B, num_thresholds, 2]
+    pred = probs.softmax(dim=2).argmax(dim=2)  # [B, num_thresholds], values in {0,1}
+    return pred.sum(dim=1)  # sum of positive threshold decisions = predicted class
+
+def evaluate_per_threshold(logits, labels):
+    """
+    logits: [N, K-1] - model outputs
+    labels: [N] - ground truth labels (ordinal, from 0 to K-1)
+    """
+    N, K_minus_1 = logits.shape
+
+    # Create binary targets per threshold
+    y_bin = np.zeros_like(logits, dtype=np.int32)
+    for k in range(K_minus_1):
+        y_bin[:, k] = (labels > k).astype(int)
+
+    metrics = {}
+    for k in range(K_minus_1):
+        prob = torch.sigmoid(logits[:, k]).cpu().numpy()
+        true_bin = y_bin[:, k]
+
+        pred_bin = (prob > 0.5).astype(int)
 
+        acc = accuracy_score(true_bin, pred_bin)
+        try:
+            auc = roc_auc_score(true_bin, prob)
+        except:
+            auc = None  # In case only one class is present
+
+        metrics[f"Threshold {k}"] = {
+            "Accuracy": acc,
+            "AUROC": auc
+        }
+
+    return metrics
 def coral_loss_manual(logits, levels, smoothing = 0.2, entropy_weight = 0.01):
     """
     logits: [B, num_classes - 1]
@@ -436,26 +585,28 @@
     # Initialize your QA model and optimizer
     print('Initiating Model')
     model = QAModel(num_thresholds=3).to(device)
-    optimizer = optim.Adam(model.parameters(), lr=1e-3)
+    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
 
 
-   # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',factor=0.5, patience=5, verbose=True)
+    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',factor=0.5, patience=5, verbose=True,min_lr=1e-6)
     # Step 1: Warmup
-    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=3)
-
-    # Step 2: Cosine Annealing after warmup
-    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=47)  # 45 = total_epochs - warmup_epochs
+    warmup_epochs = 5
+    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
+    #
+    # # Step 2: Cosine Annealing after warmup
+    scheduler = CosineAnnealingLR(optimizer, T_max=45)  # 45 = total_epochs - warmup_epochs
 
     # Combine them
-    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[5])
+    #scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, scheduler], milestones=[5])
     #scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
 
     #criterion = nn.BCEWithLogitsLoss()
     criterion = coral_loss_manual
+    #criterion = CORNLoss()
 
     #Early stopping variables
     best_val_loss = float('inf')
-    patience = 15
+    patience = 10
     patience_counter = 0
 
     #Initiate Scaler
@@ -473,6 +624,13 @@
 
     val_preds_list, val_labels_list, val_subtypes_list = [], [], []
 
+    all_labels = []
+    for _,_,labels,_ in train_loader:
+        all_labels.extend(labels.cpu().numpy().tolist())
+
+    # dist = compute_class_distribution(all_labels)
+    # print("Global class distribution:", dist)
+
 
     class_names = ["Fail (0-0.1)", "Poor (0.1-0.5)", "Moderate(0.5-0.7)", " Good (>0.7)"]
 
@@ -488,11 +646,11 @@
 
             optimizer.zero_grad()
             with autocast(device_type='cuda'):
-                print(f'Label shape {label.shape}')
+                # print(f'Label shape {label.shape}')
                 preds = model(image, uncertainty)  # shape: [B, 3]
-                print(f'model Output Shape : {preds.shape}')
+                # print(f'model Output Shape : {preds.shape}')
                 targets = encode_ordinal_targets(label).to(preds.device)
-                print(f'Tagets shape: {targets.shape}')
+                #print(f'Tagets shape: {targets.shape}')
                 loss = criterion(preds, targets)
 
 
@@ -505,6 +663,7 @@
 
             with torch.no_grad():
                 decoded_preds = decode_predictions(preds)
+                #decoded_preds = corn_predict(preds)
 
                 correct += (decoded_preds == label).sum().item()
 
@@ -518,6 +677,13 @@
         # Validation step
         model.eval()
         val_running_loss, val_correct, val_total = 0.0, 0, 0
+
+        empty_mask_count = 0
+        total_masks = 0
+
+        all_logits = []
+        all_labels = []
+
         val_preds_list.clear()
         val_labels_list.clear()
         val_subtypes_list.clear()
@@ -527,17 +693,26 @@
 
         with torch.no_grad():
             for image, uncertainty, label, subtype in val_loader:
+
                 image, uncertainty, label = image.to(device), uncertainty.to(device), label.to(device)
 
+                # Count empty masks in the batch
+                batch_empty = (image.sum(dim=[1, 2, 3, 4]) == 0).sum().item()  # assumes mask shape is (B, C, D, H, W)
+                empty_mask_count += batch_empty
+                total_masks += image.size(0)
 
                 preds = model(image, uncertainty)
                 targets = encode_ordinal_targets(label).to(preds.device)
 
+
                 loss = criterion(preds, targets)
                 val_running_loss += loss.item() * image.size(0)
 
+
+
                 with torch.no_grad():
                     decoded_preds = decode_predictions(preds)
+                    #decoded_preds = corn_predict(preds)
                     val_correct += (decoded_preds == label).sum().item()
 
 
@@ -554,6 +729,7 @@
 
                 val_subtypes_list.extend(subtype_list)
 
+        print(f"{empty_mask_count} out of {total_masks} masks in validation were completely empty.")
 
         epoch_val_loss = val_running_loss / val_total
         epoch_val_acc = val_correct / val_total
@@ -561,10 +737,16 @@
         for param_group in optimizer.param_groups:
             print(f"Current LR: {param_group['lr']}")
 
-
-        print(f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}")
         val_losses.append(epoch_val_loss)
-        # avg_val_loss = sum(val_losses) / len(val_losses)
+
+        # all_logits = torch.cat(all_logits, dim=0)
+        # all_labels = torch.cat(all_labels, dim=0).numpy()
+        #
+        # threshold_metrics = evaluate_per_threshold(all_logits, all_labels)
+        #
+        # for t, metric in threshold_metrics.items():
+        #     print(f"{t} => Accuracy: {metric['Accuracy']:.3f}, AUROC: {metric['AUROC']}")
+        # # avg_val_loss = sum(val_losses) / len(val_losses)
 
         val_preds_np = np.array(val_preds_list)
         val_labels_np = np.array(val_labels_list)
@@ -591,14 +773,32 @@
             zero_division=0
         )
 
-        scheduler.step()
+
         for class_name in class_names:
             f1_history[class_name].append(report_dict[class_name]["f1-score"])
 
 
         #print(f"Epoch {epoch}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}")
         # After each validation epoch:
+
+        # Step the appropriate scheduler
+        if epoch < warmup_epochs:
+            warmup_scheduler.step()
+            print(f"[Warmup] LR: {optimizer.param_groups[0]['lr']:.6f}")
+        else:
+            scheduler.step(epoch_val_loss)
+            print(f"[ReduceLROnPlateau] LR: {optimizer.param_groups[0]['lr']:.6f}")
+
+            #
+        # # Early stopping check
+        if epoch_val_loss < best_val_loss:
+            print(f'YAY, new best Val loss: {epoch_val_loss}!')
+            best_val_loss = epoch_val_loss
+        else:
+            print(f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}")
+
         if kappa_quadratic > best_kappa:
+            print(f'New Best Kappa: {kappa_quadratic}!')
             best_kappa = kappa_quadratic
             patience_counter = 0
 
@@ -610,10 +810,7 @@
             best_kappa_report = report
             best_kappa_epoch = epoch
 
-        #
-        # # Early stopping check
-        # if epoch_val_loss < best_val_loss:
-        #     print(f'Yay, new best : {epoch_val_loss}!')
+
         #     best_val_loss = epoch_val_loss
         #     patience_counter = 0
         #     # Save best model weights
@@ -677,8 +874,7 @@
         )
 
         end = time.time()
-        # print(f'Best report for {metric}:')
-        # print(best_report)
+
         print(f"Total training time: {(end - start) / 60:.2f} minutes")
 
         # Convert prediction outputs to numpy arrays for plotting
