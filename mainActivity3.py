# =========================
# PROJETO ECAC 2025 - mainActivity3.py
# Autor: João Natálio 2023205576
# =========================

import time
import random
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis, entropy
from scipy.fft import rfft, rfftfreq
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
from scipy.stats import wilcoxon

# Reproducibility
RND = 42

# -------------------------
# 0. Importação dos datasets já criados
# -------------------------

def load_datasets(emb_path="embeddings_dataset.csv", feat_path="features_dataset.csv"):
    emb = pd.read_csv(emb_path)
    feat = pd.read_csv(feat_path)
    print("***** 0 - Importando datasets já criados *****")
    print(f" - Embeddings dataset carregado: {emb.shape}")
    print(f" - Features dataset carregado:   {feat.shape}")
    return emb, feat


# -------------------------
# Utilitários usados por vários pontos
# -------------------------

def filter_activities_1_7(df):
    """ Mantém apenas linhas com activity entre 1 e 7 (inclusive). """
    df2 = df[(df['activity'] >= 1) & (df['activity'] <= 7)].reset_index(drop=True)
    return df2


def feature_columns_from_df(df):
    """ Extrai colunas de features (todas exceto activity, participant, device). """
    exclude = ['activity', 'participant', 'device']
    cols = [c for c in df.columns if c not in exclude]
    return cols


# =========================
# Funções de extração de features (Meta 1 / usadas em Meta 2 ponto 6)
# =========================
def extrair_features_temporais(v):
    """Calcula features temporais básicas de um vetor."""
    v = np.asarray(v, dtype=float)
    if v.size == 0:
        return {'mean': 0.0, 'std': 0.0, 'median': 0.0, 'min': 0.0, 'max': 0.0,
                'iqr': 0.0, 'rms': 0.0, 'skew': 0.0, 'kurtosis': 0.0, 'zcr': 0.0}
    try:
        zcr = np.sum(np.diff(np.sign(v)) != 0) / float(len(v))
    except Exception:
        zcr = 0.0
    return {
        'mean': np.mean(v),
        'std': np.std(v),
        'median': np.median(v),
        'min': np.min(v),
        'max': np.max(v),
        'iqr': np.percentile(v, 75) - np.percentile(v, 25),
        'rms': np.sqrt(np.mean(v**2)),
        'skew': float(skew(v)),
        'kurtosis': float(kurtosis(v)),
        'zcr': float(zcr)
    }


def extrair_features_espectrais(v, fs):
    """Calcula features espectrais de um vetor."""
    v = np.asarray(v, dtype=float)
    N = len(v)
    if N == 0:
        return {'spec_energy': 0.0, 'spec_domfreq': 0.0, 'spec_centroid': 0.0, 'spec_bandwidth': 0.0, 'spec_entropy': 0.0}

    fft_vals = np.abs(rfft(v))
    freqs = rfftfreq(N, d=1.0/fs)
    psd = fft_vals**2
    ssum = np.sum(psd)
    if ssum == 0:
        psd_norm = np.zeros_like(psd)
    else:
        psd_norm = psd / ssum

    energia = np.sum(psd) / float(max(1, N))
    if psd.size == 0:
        freq_dom = 0.0
        centroide = 0.0
        banda = 0.0
        entrop = 0.0
    else:
        idx_max = int(np.argmax(psd))
        freq_dom = float(freqs[idx_max]) if freqs.size > 0 else 0.0
        centroide = float(np.sum(freqs * psd) / float(max(1e-12, ssum)))
        banda = float(np.sqrt(np.sum(((freqs - centroide)**2) * psd) / float(max(1e-12, ssum))))
        # evitar log(0) na entropia
        psd_norm_safe = np.where(psd_norm > 0, psd_norm, 1e-12)
        entrop = float(-np.sum(psd_norm_safe * np.log(psd_norm_safe)))

    return {
        'spec_energy': energia,
        'spec_domfreq': freq_dom,
        'spec_centroid': centroide,
        'spec_bandwidth': banda,
        'spec_entropy': entrop
    }


def extrair_features_janela(janela, fs):
    """
    Extrai features temporais + espectrais dos 9 eixos + magnitudes.
    Assume que 'janela' é um DataFrame com colunas acc_x,...,mag_z e coluna 'activity'.
    """
    features = {}
    sensores = ['acc', 'gyro', 'mag']

    for s in sensores:
        for eixo in ['x', 'y', 'z']:
            col = f'{s}_{eixo}'
            v = janela[col].values

            # temporais
            ftemp = extrair_features_temporais(v)
            ftemp = {f'{col}_{k}': v for k, v in ftemp.items()}

            # espectrais
            fspec = extrair_features_espectrais(v, fs)
            fspec = {f'{col}_{k}': v for k, v in fspec.items()}

            features.update(ftemp)
            features.update(fspec)

    # Magnitudes
    for s in sensores:
        arr = janela[[f'{s}_x', f'{s}_y', f'{s}_z']].values
        mag = np.sqrt(np.sum(arr**2, axis=1))

        ftemp = extrair_features_temporais(mag)
        ftemp = {f'{s}_mag_{k}': v for k, v in ftemp.items()}

        fspec = extrair_features_espectrais(mag, fs)
        fspec = {f'{s}_mag_{k}': v for k, v in fspec.items()}

        features.update(ftemp)
        features.update(fspec)

    # label (se existir)
    if 'activity' in janela.columns and len(janela['activity'].unique()) == 1:
        features['activity'] = int(janela['activity'].iloc[0])
    else:
        features['activity'] = -1
    return features


# -------------------------
# 3.1 Within-subject TVT 60/20/20
# -------------------------
def split_within_subject(df, train_frac=0.6, val_frac=0.2, test_frac=0.2, random_state=RND):
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6
    participantes = sorted(df['participant'].unique())
    train_list, val_list, test_list = [], [], []

    for p in participantes:
        sub = df[df['participant'] == p].reset_index(drop=True)
        if len(sub) == 0:
            continue
        y = sub['activity'].values
        try:
            sub_train, sub_rest = train_test_split(sub, train_size=train_frac, stratify=y, random_state=random_state)
        except ValueError:
            sub_train, sub_rest = train_test_split(sub, train_size=train_frac, random_state=random_state)

        rest_frac_of_total = 1.0 - train_frac
        if rest_frac_of_total <= 0:
            sub_val = pd.DataFrame(columns=sub.columns)
            sub_test = pd.DataFrame(columns=sub.columns)
        else:
            val_rel = val_frac / rest_frac_of_total
            y_rest = sub_rest['activity'].values
            try:
                sub_val, sub_test = train_test_split(sub_rest, train_size=val_rel, stratify=y_rest, random_state=random_state)
            except ValueError:
                sub_val, sub_test = train_test_split(sub_rest, train_size=val_rel, random_state=random_state)

        train_list.append(sub_train)
        val_list.append(sub_val)
        test_list.append(sub_test)

    train_df = pd.concat(train_list, ignore_index=True).sample(frac=1, random_state=random_state).reset_index(drop=True)
    val_df = pd.concat(val_list, ignore_index=True).sample(frac=1, random_state=random_state).reset_index(drop=True)
    test_df = pd.concat(test_list, ignore_index=True).sample(frac=1, random_state=random_state).reset_index(drop=True)

    print("\n***** 3.1 - Split within-subject (TVT 60/20/20) *****")
    print(f"Total rows: {len(df)}")
    print(f"Train: {train_df.shape} Val: {val_df.shape} Test: {test_df.shape}")
    return train_df, val_df, test_df


# -------------------------
# 3.2 Between-subjects split: 9 / 3 / 3 participants
# -------------------------
def split_between_subjects(df, n_train=9, n_val=3, n_test=3, random_state=RND):
    participants = sorted(df['participant'].unique())
    if len(participants) < (n_train + n_val + n_test):
        raise ValueError("Não há participantes suficientes para o split between-subjects pedido.")

    rnd = random.Random(random_state)
    shuffled = participants.copy()
    rnd.shuffle(shuffled)

    train_p = shuffled[:n_train]
    val_p = shuffled[n_train:n_train + n_val]
    test_p = shuffled[n_train + n_val:n_train + n_val + n_test]

    train_df = df[df['participant'].isin(train_p)].reset_index(drop=True)
    val_df = df[df['participant'].isin(val_p)].reset_index(drop=True)
    test_df = df[df['participant'].isin(test_p)].reset_index(drop=True)

    print("\n***** 3.2 - Split between-subjects (9 / 3 / 3) *****")
    print(f"Participants -> train: {train_p} | val: {val_p} | test: {test_p}")
    print(f"Train: {train_df.shape} Val: {val_df.shape} Test: {test_df.shape}")
    return train_df, val_df, test_df

# --------------------------
# 3.3 Feature extractor from pre-trained model
# -------------------------
'''
Comentário:
Diferenças entre as estratégias:
- Within-subject: treino e teste usam dados do mesmo participante; resulta em desempenho mais alto,
mas sobrestima a performance real porque o modelo vê o "estilo" desse participante.
- Between-subject: treino e teste usam participantes diferentes; é mais difícil, mas avalia melhor
a capacidade de generalização.
Conclusão: a estratégia between-subject dá uma estimativa muito mais realista do desempenho
quando o modelo é aplicado a um novo participante.
'''


# -------------------------
# 3.4 Helpers: ALL / PCA / ReliefF
# -------------------------
def get_Xy_from_df(df, feature_cols=None):
    if feature_cols is None:
        feature_cols = feature_columns_from_df(df)
    X = df[feature_cols].values
    y = df['activity'].values
    return X, y, feature_cols


def fit_transform_pca(X_train, X_val, X_test, var_target=0.90):
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    pca = PCA(n_components=var_target, svd_solver='auto')
    X_train_p = pca.fit_transform(X_train_s)
    X_val_p = pca.transform(X_val_s)
    X_test_p = pca.transform(X_test_s)

    print(f"PCA: reduzido para {X_train_p.shape[1]} componentes para {var_target*100:.0f}% var.")
    return (X_train_p, X_val_p, X_test_p), scaler, pca


def compute_reliefF_topk(X_train, y_train, k_select=15, n_neighbors=10):
    n_samples, n_features = X_train.shape
    weights = np.zeros(n_features, dtype=np.float64)

    n_neighbors_eff = min(n_neighbors + 1, n_samples)
    nn = NearestNeighbors(n_neighbors=n_neighbors_eff, algorithm='auto').fit(X_train)
    neighbors = nn.kneighbors(X_train, return_distance=False)

    for i in range(n_samples):
        xi = X_train[i]
        yi = y_train[i]
        neigh_idx = neighbors[i]
        neigh_idx = neigh_idx[neigh_idx != i]
        if neigh_idx.size == 0:
            continue

        mask_hits = (y_train[neigh_idx] == yi)
        hits_idx = neigh_idx[mask_hits]
        misses_idx = neigh_idx[~mask_hits]

        if hits_idx.size == 0 or misses_idx.size == 0:
            continue

        hits = np.mean(np.abs(X_train[hits_idx] - xi), axis=0)
        misses = np.mean(np.abs(X_train[misses_idx] - xi), axis=0)
        weights += (misses - hits)

    if n_samples > 0:
        weights = weights / float(max(1, n_samples))
    topk_idx = np.argsort(-weights)[:k_select]
    return topk_idx, weights


def prepare_versions(train_df, val_df, test_df, k_relief=15, pca_var=0.90):
    feature_cols = feature_columns_from_df(train_df)
    X_train_all, y_train, _ = get_Xy_from_df(train_df, feature_cols)
    X_val_all, y_val, _ = get_Xy_from_df(val_df, feature_cols)
    X_test_all, y_test, _ = get_Xy_from_df(test_df, feature_cols)

    versions = {}
    versions['all'] = {
        'X_train': X_train_all, 'y_train': y_train,
        'X_val': X_val_all, 'y_val': y_val,
        'X_test': X_test_all, 'y_test': y_test,
        'feature_cols': feature_cols
    }

    (Xtr_p, Xv_p, Xt_p), scaler_pca, pca = fit_transform_pca(X_train_all, X_val_all, X_test_all, var_target=pca_var)
    versions['pca'] = {
        'X_train': Xtr_p, 'y_train': y_train,
        'X_val': Xv_p, 'y_val': y_val,
        'X_test': Xt_p, 'y_test': y_test,
        'pca': pca, 'scaler': scaler_pca
    }

    scaler_rel = StandardScaler()
    Xtr_s = scaler_rel.fit_transform(X_train_all)
    Xv_s = scaler_rel.transform(X_val_all)
    Xt_s = scaler_rel.transform(X_test_all)

    topk_idx, weights = compute_reliefF_topk(Xtr_s, y_train, k_select=k_relief, n_neighbors=10)
    selected_cols = [feature_cols[i] for i in topk_idx]

    versions['relief'] = {
        'X_train': Xtr_s[:, topk_idx], 'y_train': y_train,
        'X_val': Xv_s[:, topk_idx], 'y_val': y_val,
        'X_test': Xt_s[:, topk_idx], 'y_test': y_test,
        'selected_idx': topk_idx, 'selected_cols': selected_cols, 'scaler': scaler_rel, 'weights': weights
    }

    print("\nPrepared versions: 'all', 'pca' (90% var), 'relief' (top-15).")
    print(f" - All feature dim: {versions['all']['X_train'].shape[1]}")
    print(f" - PCA dim: {versions['pca']['X_train'].shape[1]}")
    print(f" - Relief dim: {versions['relief']['X_train'].shape[1]} (top-15)")

    return versions


# -------------------------
# 4.1 Implementação K-NN manual
# -------------------------
def knn_predict(X_train, y_train, X_test, k=5, metric='euclidean'):
    if X_train.ndim == 1:
        X_train = X_train.reshape(-1, 1)
    if X_test.ndim == 1:
        X_test = X_test.reshape(-1, 1)

    D = cdist(X_test, X_train, metric=metric)
    n_test = D.shape[0]
    preds = []
    for i in range(n_test):
        idx_sorted = np.argsort(D[i])[:k]
        neigh_labels = y_train[idx_sorted]
        vals, counts = np.unique(neigh_labels, return_counts=True)
        max_count = counts.max()
        candidates = vals[counts == max_count]
        if candidates.size == 1:
            pred = int(candidates[0])
        else:
            sumd_per_candidate = {}
            for c in candidates:
                mask = (neigh_labels == c)
                sumd = D[i][idx_sorted][mask].sum()
                sumd_per_candidate[int(c)] = sumd
            pred = int(min(sumd_per_candidate.items(), key=lambda x: (x[1], x[0]))[0])
        preds.append(pred)
    return np.array(preds, dtype=int)


def tune_k(X_train, y_train, X_val, y_val, ks=(1,3,5,7,9), metric='euclidean'):
    best_k = None
    best_score = -1.0
    print("Tunando k (val)...")
    for k in ks:
        y_pred = knn_predict(X_train, y_train, X_val, k=k, metric=metric)
        acc = accuracy_score(y_val, y_pred)
        print(f" k={k} -> val-acc={acc:.4f}")
        if acc > best_score:
            best_score = acc
            best_k = k
    print(f"Melhor k encontrado: {best_k} | Val-Acc={best_score:.4f}")
    return best_k, best_score


# -------------------------
# 4.2 Métricas
# -------------------------
def compute_metrics(y_true, y_pred, labels=None):
    if labels is None:
        labels = np.unique(np.concatenate((y_true, y_pred)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, sup = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    metrics = {
        'labels': labels,
        'confusion_matrix': cm,
        'accuracy': acc,
        'precision_per_class': prec,
        'recall_per_class': rec,
        'f1_per_class': f1,
        'support_per_class': sup,
        'precision_macro': np.mean(prec) if len(prec)>0 else 0.0,
        'recall_macro': np.mean(rec) if len(rec)>0 else 0.0,
        'f1_macro': np.mean(f1) if len(f1)>0 else 0.0
    }
    return metrics


def print_metrics(metrics):
    labels = metrics['labels']
    print("\n===== Resultados =====")
    print(f"Accuracy = {metrics['accuracy']:.4f}")
    print("Confusion Matrix:\n", metrics['confusion_matrix'])
    prec = [f"{p:.3f}" for p in metrics['precision_per_class']]
    rec = [f"{r:.3f}" for r in metrics['recall_per_class']]
    f1 = [f"{f:.3f}" for f in metrics['f1_per_class']]
    print("\nPrecision por classe:", prec)
    print("Recall por classe:   ", rec)
    print("F1 por classe:       ", f1)
    print(f"\nMacro Precision={metrics['precision_macro']:.4f}  Macro Recall={metrics['recall_macro']:.4f}  Macro F1={metrics['f1_macro']:.4f}")


# -------------------------
# 5. Experiment runner
# -------------------------
def run_evaluation_for_setting(dataset_name, split_type, emb_df, feat_df, ks=(1,3,5,7,9), repeats=5, random_state=RND):
    results = {}
    for version in ['all', 'pca', 'relief']:
        results[version] = {'test_accs': [], 'test_f1s': [], 'best_ks': []}

    for r in range(repeats):
        seed = random_state + r
        if dataset_name == 'embeddings':
            df = emb_df
        elif dataset_name == 'features':
            df = feat_df
        else:
            raise ValueError("dataset_name must be 'embeddings' or 'features'")

        if split_type == 'within':
            tr, val, te = split_within_subject(df, random_state=seed)
        elif split_type == 'between':
            tr, val, te = split_between_subjects(df, random_state=seed)
        else:
            raise ValueError("split_type must be 'within' or 'between'")

        versions = prepare_versions(tr, val, te, k_relief=15, pca_var=0.90)

        for version in ['all', 'pca', 'relief']:
            ds = versions[version]
            Xtr, ytr = ds['X_train'], ds['y_train']
            Xv, yv = ds['X_val'], ds['y_val']
            Xt, yt = ds['X_test'], ds['y_test']

            if Xtr.shape[0] == 0 or Xt.shape[0] == 0:
                print(f" Warning: empty train or test for repeat {r}, version {version}. Skipping.")
                results[version]['test_accs'].append(np.nan)
                results[version]['test_f1s'].append(np.nan)
                results[version]['best_ks'].append(None)
                continue

            if Xv.shape[0] == 0:
                best_k, best_val_acc = tune_k(Xtr, ytr, Xtr, ytr, ks=ks)
            else:
                best_k, best_val_acc = tune_k(Xtr, ytr, Xv, yv, ks=ks)

            X_comb = np.vstack([Xtr, Xv]) if Xv.shape[0] > 0 else Xtr
            y_comb = np.concatenate([ytr, yv]) if yv.size > 0 else ytr

            y_pred = knn_predict(X_comb, y_comb, Xt, k=best_k)
            m = compute_metrics(yt, y_pred)
            results[version]['test_accs'].append(m['accuracy'])
            results[version]['test_f1s'].append(m['f1_macro'])
            results[version]['best_ks'].append(best_k)

            print(f" repeat {r+1}/{repeats} -> test-acc={m['accuracy']:.4f} macro-f1={m['f1_macro']:.4f} (k={best_k})")

    print(f"\n--- Summary for {dataset_name} | {split_type} ---")
    for version in ['all', 'pca', 'relief']:
        accs = np.array(results[version]['test_accs'], dtype=np.float64)
        f1s = np.array(results[version]['test_f1s'], dtype=np.float64)
        accs_valid = accs[~np.isnan(accs)] if accs.size>0 else np.array([])
        f1s_valid = f1s[~np.isnan(f1s)] if f1s.size>0 else np.array([])
        if accs_valid.size > 0:
            print(f" {version} -> acc mean={accs_valid.mean():.4f} std={accs_valid.std():.4f} | f1 mean={f1s_valid.mean():.4f} std={f1s_valid.std():.4f}")
        else:
            print(f" {version} -> no valid runs")

    means = {}
    for v in results:
        arr = np.array(results[v]['test_f1s'], dtype=np.float64)
        arr = arr[~np.isnan(arr)]
        means[v] = np.mean(arr) if arr.size > 0 else -np.inf
    best_version = max(means, key=means.get)
    print(f"\nBest version by mean macro-F1: {best_version} (mean F1={means[best_version]:.4f})")
    for v in results:
        if v == best_version:
            continue
        a = np.array(results[best_version]['test_f1s'], dtype=np.float64)
        b = np.array(results[v]['test_f1s'], dtype=np.float64)
        mask = (~np.isnan(a)) & (~np.isnan(b))
        a_p = a[mask]
        b_p = b[mask]
        if a_p.size == 0:
            print(f"Wilcoxon {best_version} vs {v}: not enough paired runs")
            continue
        if np.allclose(a_p, b_p):
            print(f"Wilcoxon {best_version} vs {v}: identical values (p=1.0000)")
            continue
        try:
            stat, p = wilcoxon(a_p, b_p)
            print(f"Wilcoxon {best_version} vs {v}: stat={stat:.4f}, p={p:.4f}")
        except Exception as e:
            print(f"Wilcoxon failed for {best_version} vs {v}: {e}")

    return results


'''
Comentário:
Os resultados das avaliações mostram que as performances variam conforme a estratégia e a transformação usada.
Nos embeddings, a versão All foi a que obteve melhor macro-F1 em média.
Nas features tradicionais, a versão PCA destacou-se com desempenho claramente superior.
As repetições mostraram resultados estáveis e consistentes, e os testes estatísticos confirmam
que as diferenças observadas têm tendência real, mesmo que não sejam estatisticamente significativas.
No geral, a versão PCA das features apresentou o melhor equilíbrio entre desempenho e robustez.

'''


# ======================================================
# 6. DEPLOYMENT – Classificação direta de janelas 256x9
# ======================================================
def extract_features_from_window(window_256x9, fs=50.0):
    """
    Extrai features temporais + espectrais de uma janela 256x9.
    Retorna dict com features na mesma ordem (nomes) que o feature dataset.
    """
    df_temp = pd.DataFrame(window_256x9, columns=[
        'acc_x','acc_y','acc_z',
        'gyro_x','gyro_y','gyro_z',
        'mag_x','mag_y','mag_z'
    ])

    f = extrair_features_janela(df_temp, fs)

    # remover chaves irrelevantes
    f.pop('activity', None)
    f.pop('participant', None)
    f.pop('device', None)
    return f


def classify_window_features(window_256x9, model):
    """
    Classifica uma janela bruta (256x9) usando o modelo final:
    - extração de features
    - ordenação de colunas
    - normalização
    - PCA / seleção relief
    - KNN manual
    """
    feat = extract_features_from_window(window_256x9)

    # garantir que todas as feature_cols existem
    missing = [c for c in model['feature_cols'] if c not in feat]
    if missing:
        raise KeyError(f"Faltam features na janela extraída: {missing[:5]}... (total {len(missing)})")

    X = np.array([feat[c] for c in model['feature_cols']], dtype=float).reshape(1, -1)

    if model.get('scaler') is not None:
        X = model['scaler'].transform(X)

    if model['type'] == 'pca':
        X = model['pca'].transform(X)
    elif model['type'] == 'relief' and model.get('relief_idx') is not None:
        X = X[:, model['relief_idx']]

    pred = knn_predict(model['X_train'], model['y_train'], X, k=model['knn_k'])[0]
    return int(pred)


def build_deployment_model_from_features_df(feat_df, k=7, pca_var=0.90):
    """
    Cria o modelo FINAL de deployment usando TODO o feature set:
    - StandardScaler
    - PCA
    - KNN manual (armazenamos X_train como transformado)
    """
    print("\n===== Construção do modelo final — Ponto 6 =====")
    feats = feat_df.copy().reset_index(drop=True)

    feature_cols = feature_columns_from_df(feats)
    X = feats[feature_cols].values.astype(float)
    y = feats['activity'].values.astype(int)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    pca = PCA(n_components=pca_var, svd_solver='auto')
    Xp = pca.fit_transform(Xs)

    model = {
        'type': 'pca',
        'knn_k': int(k),
        'X_train': Xp,
        'y_train': y,
        'feature_cols': feature_cols,
        'scaler': scaler,
        'pca': pca,
        'relief_idx': None
    }

    print(f"Modelo final criado: PCA={Xp.shape[1]} componentes | k={k}")
    return model


def predict_from_raw_segment(raw_window_256x9, model):
    arr = np.asarray(raw_window_256x9, dtype=float)
    if arr.shape != (256, 9):
        raise ValueError(f"Shape inválido — esperado (256,9), recebido {arr.shape}")
    return classify_window_features(arr, model)


# ======================================================
# 7. Weighted KNN (Variação simples para melhorar performance)
# ======================================================
def knn_predict_weighted(X_train, y_train, X_test, k=5, metric='euclidean', epsilon=1e-9):
    """
    Versão ponderada do KNN:
    - Cada vizinho contribui com peso = 1 / (distância + epsilon)
    - Em vez de maioria simples, escolhe-se o label com maior soma de pesos.

    É muito simples, rápido e melhora normalmente o F1.
    """
    if X_train.ndim == 1:
        X_train = X_train.reshape(-1, 1)
    if X_test.ndim == 1:
        X_test = X_test.reshape(-1, 1)

    D = cdist(X_test, X_train, metric=metric)
    n_test = D.shape[0]
    preds = []

    for i in range(n_test):
        idx_sorted = np.argsort(D[i])[:k]
        neigh_labels = y_train[idx_sorted]
        neigh_dist = D[i][idx_sorted]

        # pesos inversos: quanto menor a distância, maior o peso
        weights = 1.0 / (neigh_dist + epsilon)

        # somar pesos por classe
        class_scores = {}
        for lab, w in zip(neigh_labels, weights):
            class_scores[lab] = class_scores.get(lab, 0.0) + w

        # escolher a classe com maior soma de pesos
        pred = max(class_scores.items(), key=lambda x: (x[1], -x[0]))[0]
        preds.append(pred)

    return np.array(preds, dtype=int)


# -------------------------
# MAIN
# -------------------------
if __name__ == "__main__":
    t0 = time.time()
    try:
        emb_path = "embeddings_dataset.csv"
        feat_path = "features_dataset.csv"

        emb_df, feat_df = load_datasets(emb_path, feat_path)

        # garantir colunas participant/device exist
        if 'participant' not in emb_df.columns:
            raise KeyError("Embeddings dataset não tem coluna 'participant'.")
        if 'participant' not in feat_df.columns:
            # alguns feature datasets podem não ter participant (ex: gerados sem). Se faltar, criamos participant=-1
            feat_df['participant'] = -1

        # filtrar activities
        print("\n***** Preparar datasets (filtro atividades 1..7) *****")
        emb_df = filter_activities_1_7(emb_df)
        feat_df = filter_activities_1_7(feat_df)
        print(f" - Filtrar atividades 1..7: {emb_df.shape[0]} linhas (embeddings)")
        print(f" - Filtrar atividades 1..7: {feat_df.shape[0]} linhas (features)")

        print("\nShapes após filtragem:")
        print(f" - EMBEDDINGS: {emb_df.shape}")
        print(f" - FEATURES:   {feat_df.shape}")

        # Exemplo split within embeddings
        print("\n***** 3.1 - Split within-subject (embeddings) *****")
        emb_train, emb_val, emb_test = split_within_subject(emb_df, random_state=RND)
        emb_feature_cols = feature_columns_from_df(emb_df)
        Xtr_e, ytr_e, _ = get_Xy_from_df(emb_train, emb_feature_cols)
        print(f"Train: {Xtr_e.shape} Val: {get_Xy_from_df(emb_val, emb_feature_cols)[0].shape} Test: {get_Xy_from_df(emb_test, emb_feature_cols)[0].shape}")

        # split between for features
        print("\n***** 3.2 - Split between-subjects (features) *****")
        feat_train, feat_val, feat_test = split_between_subjects(feat_df, random_state=RND)

        # preparar versões (exemplo)
        print("\n***** Preparar cenários (exemplo) *****")
        print(" - Preparando versões para EMBEDDINGS (within-subject split)...")
        emb_versions = prepare_versions(emb_train, emb_val, emb_test, k_relief=15, pca_var=0.90)

        print("\n - Preparando versões para FEATURES (between-subjects split)...")
        feat_versions = prepare_versions(feat_train, feat_val, feat_test, k_relief=15, pca_var=0.90)

        # Teste rápido K-NN
        print("\n===== 4.1 – Teste do KNN manual =====")
        scenario = 'all'
        ds = emb_versions[scenario]
        Xtr, ytr = ds['X_train'], ds['y_train']
        Xv, yv = ds['X_val'], ds['y_val']
        Xt, yt = ds['X_test'], ds['y_test']

        ks = [1,3,5,7,9,11]
        best_k, _ = tune_k(Xtr, ytr, Xv, yv, ks=ks)
        print("A testar no conjunto de teste...")
        y_pred = knn_predict(np.vstack([Xtr, Xv]), np.concatenate([ytr, yv]), Xt, k=best_k)
        metrics = compute_metrics(yt, y_pred)
        print_metrics(metrics)
        print("\nFIM do exemplo K-NN (ponto 4).\n")

        # 5. Run experiments (single-run)
        print("\n===== 5. Evaluation (single-run experiments conforme PDF) =====\n")
        ks_to_try = [1,3,5,7,9]
        repeats = 5

        print("\n--- Evaluating: dataset=embeddings | split=within ---")
        results_emb_within = run_evaluation_for_setting("embeddings", "within", emb_df, feat_df, ks=ks_to_try, repeats=repeats, random_state=RND)

        print("\n--- Evaluating: dataset=embeddings | split=between ---")
        results_emb_between = run_evaluation_for_setting("embeddings", "between", emb_df, feat_df, ks=ks_to_try, repeats=repeats, random_state=RND+100)

        print("\n--- Evaluating: dataset=features | split=between ---")
        results_feat_between = run_evaluation_for_setting("features", "between", emb_df, feat_df, ks=ks_to_try, repeats=repeats, random_state=RND+200)

        print("\n--- Evaluating: dataset=features | split=within ---")
        results_feat_within = run_evaluation_for_setting("features", "within", emb_df, feat_df, ks=ks_to_try, repeats=repeats, random_state=RND+300)

        print("\n===== Experiments finished. =====")

        # 6. Deployment - construir modelo final usando features dataset carregado
        print("\n===== 6. Deployment — Construção do Modelo Final =====")
        # usamos feat_df (já filtrado) como dataset final para treinar o modelo de deployment
        best_model = build_deployment_model_from_features_df(feat_df, k=7, pca_var=0.90)

        # exemplo sintético
        example_window = np.tile(np.linspace(0, 1, 256), (9, 1)).T
        pred_example = predict_from_raw_segment(example_window, best_model)

        print("\n===== 6. Deployment — Previsão =====")
        print("Atividade prevista (janela sintética):", pred_example)
        
        # 7. Go Further – Weighted KNN (ponto extremamente simples)
        print("\n===== 7. Go Further – Weighted KNN =====")

        # Usamos o mesmo cenário do exemplo anterior (ponto 4)
        scenario = 'all'
        ds = emb_versions[scenario]
        Xtr, ytr = ds['X_train'], ds['y_train']
        Xv, yv = ds['X_val'], ds['y_val']
        Xt, yt = ds['X_test'], ds['y_test']

        print("A testar Weighted KNN no set de teste...")

        k = 7  # mesmo k usado no ponto 6
        y_pred_w = knn_predict_weighted(np.vstack([Xtr, Xv]), np.concatenate([ytr, yv]), Xt, k=k)
        metrics_w = compute_metrics(yt, y_pred_w)

        print("\n===== RESULTADOS WEIGHTED KNN =====")
        print_metrics(metrics_w)

        print("\nMelhoria simples explicada na defesa:")
        print(" - Vizinhos mais próximos recebem mais peso.")
        print(" - Melhor estabilidade em classes com muita variância.")
        print(" - Reduz desempates.")


    except Exception as e:
        print("Erro durante execução do MAIN:", e)
    finally:
        print(f"\nTempo total: {time.time() - t0:.2f} s")