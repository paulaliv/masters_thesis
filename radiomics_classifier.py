import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from torch.backends.mkl import verbose
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_validate
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV


from sklearn.feature_selection import SelectKBest, mutual_info_classif,  f_classif

import matplotlib.pyplot as plt

# # Assign colors by prefix
# prefix_colors = {
#     'original': '#1f77b4',
#     'confidence': '#ff7f0e',
#     'entropy': '#2ca02c',
#     'epkl': '#d62728',
#     'mutual': '#9467bd'
# }
prefix_colors = {
    'original': '#4E79A7',    # steel blue – calm and professional
    'confidence': '#F28E2B',  # muted orange – stands out but not too bright
    'entropy': '#59A14F',     # medium green – pleasant and readable
    'epkl': '#E15759',        # soft red – draws attention
    'mutual': '#B07AA1'       # muted purple – complements other colors
}
# Clean up feature names for plotting (keep prefixes)
def clean_feature_name_keep_prefix(name):
    # Replace underscores with spaces and capitalize words
    name_parts = name.split('_')
    name_clean = ' '.join([part.capitalize() for part in name_parts])
    return name_clean

def remove_highly_correlated(X, threshold=0.95):
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X)
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    print(f"Dropping {len(to_drop)} highly correlated features")
    return X.drop(columns=to_drop)


def get_color(feature_name):
    for prefix in prefix_colors.keys():
        if feature_name.startswith(prefix):
            return prefix_colors[prefix]
    return '#7f7f7f'  # default gray

def feature_importance(X):
    # Mutual Information scores
    mi_scores = mutual_info_classif(X, y, random_state=42)
    mi_df = pd.DataFrame({"feature": X.columns, "mi_score": mi_scores})

    mi_df['feature_clean'] = mi_df['feature'].apply(clean_feature_name_keep_prefix)

    # Sort features by MI descending
    sorted_features = mi_df.sort_values('mi_score', ascending=False)['feature'].values

    selected_features = []
    threshold_corr = 0.9  # max allowed correlation between selected features

    for feat in sorted_features:
        if len(selected_features) == 0:
            selected_features.append(feat)
            continue
        # Compute correlation with already selected features
        corr_with_selected = np.max([abs(X[feat].corr(X[s])) for s in selected_features])
        if corr_with_selected < threshold_corr:
            selected_features.append(feat)

    print("Selected features after redundancy reduction:", selected_features)

    mi_df = mi_df[mi_df['feature'].isin(selected_features)]

    mi_df['color'] = mi_df['feature'].apply(get_color)

    # Top 20 features
    top_20 = mi_df.sort_values("mi_score", ascending=False).head(20)
    max_val = top_20['mi_score'].max()

    # Plot horizontal bar chart
    plt.figure(figsize=(10, 8))
    bars = plt.barh(top_20['feature_clean'], top_20['mi_score'], color=top_20['color'])
    plt.gca().invert_yaxis()  # largest on top

    plt.xlabel("Mutual Information Score")
    plt.title("Top 20 Features", fontsize=16)

    # # Add MI score labels
    # for bar in bars:
    #     width = bar.get_width()
    #     plt.text(width + 0.01, bar.get_y() + bar.get_height() / 2, f'{width:.2f}', va='center')

    plt.tight_layout()
    plt.savefig("/gpfs/home6/palfken/feature_imp_rad_professional.png", dpi=300)
    plt.show()



def get_models():
    models = {
        "XGBoost": XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42
        ),
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
        "LightGBM": LGBMClassifier(n_estimators=200, learning_rate=0.05, random_state=42,verbose=-1),
        "CatBoost": CatBoostClassifier(verbose=0, random_state=42),
        "SVM": SVC(kernel="rbf", probability=True, C=1.0, gamma="scale", random_state=42),
        "LogisticRegression": LogisticRegression(penalty='l2', C=1.0, solver='lbfgs', max_iter=1000, random_state=42),
        "MLP": MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)


    }


    return models

def evaluate_model(name, model, X, y, label_names, k_value, patience = 10, val_size=0.1):

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    accs, f1s, aucs = [], [], []
    all_y_true, all_y_pred = [], []
    X = remove_highly_correlated(X)
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        # Ensure X supports .iloc (convert back to DataFrame if needed)
        if isinstance(X, np.ndarray):
            X_train_full, X_test = X[train_idx], X[test_idx]
        else:
            X_train_full, X_test = X.iloc[train_idx], X.iloc[test_idx]

        if isinstance(y, pd.Series):
            y_train_full, y_test = y.iloc[train_idx], y.iloc[test_idx]
        else:
            y_train_full, y_test = y[train_idx], y[test_idx]
        #y_train, y_test = y[train_idx], y[test_idx]

        # Create a small validation split from training fold for early stopping
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full, y_train_full, test_size=val_size,
            stratify=y_train_full, random_state=42
        )
        #Preprocessing per fold
        pipeline = Pipeline([
            ('var_thresh', VarianceThreshold(threshold=0.05)),
            ('select_best', SelectKBest(mutual_info_classif, k=min(k_value, X.shape[1]))),
            ('scaler', StandardScaler()),
            ('clf', model)
        ])


        # Fit pipeline with early stopping if the model supports eval_set
        if hasattr(model, "fit") and "eval_set" in model.fit.__code__.co_varnames:
            pipeline.named_steps['clf'].fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=patience,
                verbose=False
            )
        else:
            pipeline.fit(X_train, y_train)


        #pipeline.fit(X_train, y_train)
        # low_var_cols = X_train.var()[X_train.var() < 1e-8]
        # print(f"Low variance features: {len(low_var_cols)}")

        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)

        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)

        accs.append(accuracy_score(y_test, y_pred))
        f1s.append(f1_score(y_test, y_pred, average='weighted'))
        aucs.append(roc_auc_score(y_test, y_proba, multi_class='ovr'))

    # Print final averaged classification report
    #print(f"Model: {name}")
    print(f"Classification report for {name} (K={k_value}):")
    print(classification_report(all_y_true, all_y_pred, target_names=label_names))
    print("-" * 40)

    return {
        "accuracy": np.mean(accs),
        "f1_score": np.mean(f1s),
        "roc_auc_ovr": np.mean(aucs)
    }






# Load your features and labels
csv_file = pd.read_csv("/gpfs/home6/palfken/final_features.csv", index_col=0)
csv_file.rename(columns={'tumor_class_x':'tumor_class'}, inplace=True)
csv_file.drop(columns='tumor_class_y', inplace=True)

#csv_file.to_csv("/gpfs/home6/palfken/final_features.csv")

X = csv_file.drop(columns=['case_id', 'tumor_class'])
non_numeric_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
print(X[non_numeric_cols].head(50))
print(non_numeric_cols)

# Print columns that will be dropped
print("Dropping the following non-numeric columns from X:", non_numeric_cols)

# Drop them
X = X.drop(columns=non_numeric_cols)
print(len(X.columns))

nan_cols = X.columns[X.isna().any()].tolist()
print("Dropping columns with NaN values:", nan_cols)
X = X.fillna(0)
#X = X.drop(columns=nan_cols)



y = csv_file['tumor_class']
# Encode labels if they're not numeric
if y.dtype == 'object':
    le = LabelEncoder()
    y = le.fit_transform(y)
    label_names = le.classes_
else:
    label_names = np.unique(y)

#feature_importance(X)

#
#
# preprocessing = Pipeline([
#     ('var_thresh', VarianceThreshold(threshold=0.01)),
#     ('scaler', StandardScaler())
# ])

# Apply preprocessing
#X_processed = preprocessing.fit_transform(X)


param_grid = {
    'XGBoost': {
        'clf__n_estimators': [100, 200, 500],  # already in your code
        'clf__learning_rate': [0.01, 0.05, 0.1],
        'clf__max_depth': [3, 5, 7],
        'clf__subsample': [0.7, 0.85, 1.0],
        'clf__colsample_bytree': [0.7, 0.85, 1.0],
        'clf__gamma': [0, 0.1, 0.3],
        'select_best__k': [200, 250, 300]


    },
    'LightGBM': {
    'clf__n_estimators': [100, 200, 500],
    'clf__learning_rate': [0.01, 0.05, 0.1],
    'clf__max_depth': [3, 5, 7, -1],   # -1 = no limit
    'clf__num_leaves': [31, 50, 70],
    'clf__subsample': [0.7, 0.85, 1.0],
    'clf__colsample_bytree': [0.7, 0.85, 1.0],
    'select_best__k': [200, 250, 300]
    },
    'CatBoost':{
        'clf__iterations': [200, 500, 1000],
        'clf__learning_rate': [0.01, 0.05, 0.1],
        'clf__depth': [3, 5, 7],
        'clf__l2_leaf_reg': [1, 3, 5, 7],
        'select_best__k': [200, 250, 300]

    }
}


models = get_models()

best_scores = {}
for name, model in models.items():
    if name not in param_grid:  # Skip models without param grids
        continue
    print(f"Running GridSearchCV for {name}...")

    # Wrap classifier in a pipeline with SelectKBest
    pipe = Pipeline([
        ('select_best', SelectKBest(score_func=f_classif)),
        ('clf', model)
    ])

    # Setup GridSearchCV
    grid = GridSearchCV(
        estimator=model,
        param_grid=param_grid[name],
        scoring='f1_weighted',  # or 'accuracy', 'roc_auc_ovr', etc.
        cv=5,
        n_jobs=-1,
        verbose=2
    )

    # Fit the grid search
    grid.fit(X, y)

    print(f"Best parameters for {name}: {grid.best_params_}")
    print(f"Best F1 score for {name}: {grid.best_score_:.4f}")
    best_scores[name] = grid.best_score_

    # best_score = 0
    # best_k = None
    # print(f"Model: {name}")
    # for k_value in k_values:
    #     print(f"=== Testing with K={k_value} ===")
    #     results = evaluate_model(name, model, X, y, label_names, k_value)
    #     for key, value in results.items():
    #         print(f"  {key}: {value:.4f}")
    #         if key == 'f1_score':
    #             if value > best_score:
    #                 best_score = value
    #                 best_k = k_value
    #                 best_scores[name] = value
    #     print("-" * 30)
    # print(f'Best Score for {name}: {best_score}')
    # print(f'Best K for {name}: {best_k}')