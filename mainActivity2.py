# =========================
# PROJETO ECAC 2025 - Meta 2 (Part A)
# Autor: João Natálio 2023205576
# =========================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
import torch
from embeddings_extractor import load_model, resample_to_30hz_5s, acc_segmentation


# =========================
# 0. Importação dos dados
# =========================
def importData():
    cols = ['device', 'acc_x', 'acc_y', 'acc_z',
            'gyro_x', 'gyro_y', 'gyro_z',
            'mag_x', 'mag_y', 'mag_z',
            'timestamp', 'activity']

    dev1_list, dev2_list, dev3_list, dev4_list, dev5_list = [], [], [], [], []

    for i in range(15):  # participantes 0..14
        for d in range(1, 6):
            filename = f'part{i}dev{d}.csv'
            try:
                df = pd.read_csv(filename, names=cols)
            except FileNotFoundError:
                print(f"Aviso: Ficheiro {filename} não encontrado. Continuando...")
                continue

            df['participant'] = i
            df = df[df['activity'] <= 7]

            if d == 1:
                dev1_list.append(df)
            elif d == 2:
                dev2_list.append(df)
            elif d == 3:
                dev3_list.append(df)
            elif d == 4:
                dev4_list.append(df)
            elif d == 5:
                dev5_list.append(df)

    dev1 = pd.concat(dev1_list, ignore_index=True)
    dev2 = pd.concat(dev2_list, ignore_index=True)
    dev3 = pd.concat(dev3_list, ignore_index=True)
    dev4 = pd.concat(dev4_list, ignore_index=True)
    dev5 = pd.concat(dev5_list, ignore_index=True)

    print("Dados importados com sucesso:")
    print(f" - Dispositivo 1: {len(dev1)} amostras")
    print(f" - Dispositivo 2: {len(dev2)} amostras")
    print(f" - Dispositivo 3: {len(dev3)} amostras")
    print(f" - Dispositivo 4: {len(dev4)} amostras")
    print(f" - Dispositivo 5: {len(dev5)} amostras")

    return dev1, dev2, dev3, dev4, dev5


# =========================
# 1.1. Analisar equilíbrio
# =========================
def analisarEquilibrio(dev, nome_dev="Desconhecido"):
    print(f"Distribuição de atividades - {nome_dev}")
    contagem = dev['activity'].value_counts().sort_index()
    for atividade, n_amostras in contagem.items():
        print(f"Atividade {atividade}: {n_amostras} amostras")
        
        

# =========================
# 1.2. SMOTE
# =========================
def gerarSMOTE(dev, atividade, k_novos):
    features = [c for c in dev.columns if c not in ['device', 'timestamp', 'activity', 'participant']]
    X = dev[features]
    y = dev['activity']
    y_bin = (y == atividade).astype(int)
    n_atual = y_bin.sum()
    n_desejado = n_atual + k_novos

    smote = SMOTE(sampling_strategy={1: n_desejado}, random_state=42)
    X_res, y_res = smote.fit_resample(X, y_bin)

    novas = X_res.iloc[len(X):].copy()
    novas['activity'] = atividade
    novas['device'] = dev['device'].iloc[0]
    novas['timestamp'] = np.nan
    novas['participant'] = dev['participant'].iloc[0]

    print(f"Foram geradas {len(novas)} novas amostras para a atividade {atividade}.")
    dev_final = pd.concat([dev, novas], ignore_index=True)
    return dev_final, novas


# =========================
# 1.3 Visualizar SMOTE
# =========================
def obter_participante(dev1, dev2, dev3, dev4, dev5, participant_id):
    df_all = pd.concat([dev1, dev2, dev3, dev4, dev5], ignore_index=True)
    return df_all[df_all['participant'] == participant_id].reset_index(drop=True)


def visualizarSMOTE(dev_original, novas_amostras, atividade):
    features = [c for c in dev_original.columns if c not in ['device', 'timestamp', 'activity', 'participant']]
    if len(features) < 2:
        print("Não há features suficientes para visualização.")
        return
    f1, f2 = features[0], features[1]
    plt.figure(figsize=(8, 6))
    plt.scatter(dev_original[f1], dev_original[f2],
                c=dev_original['activity'], cmap='tab10', alpha=0.6)
    plt.scatter(novas_amostras[f1], novas_amostras[f2],
                color='red', marker='x', s=80)
    plt.title(f"Atividade {atividade} - Novas amostras geradas")
    plt.tight_layout()
    plt.show()


# =========================
# 2.1 Geração de EMBEDDINGS
# =========================
def gerar_embeddings_dataset(dev, device_id=2, save_csv=True):
    """
    Gera embeddings usando:
        - acc_segmentation
        - resample_to_30hz_5s
        - load_model
    """

    # converter para array ordenado
    cols_order = [
        'device', 'acc_x', 'acc_y', 'acc_z',
        'gyro_x', 'gyro_y', 'gyro_z',
        'mag_x', 'mag_y', 'mag_z',
        'timestamp', 'activity'
    ]

    data_np = dev[cols_order].values.astype(np.float32)

    # segmentação com STARTS
    print(" - Segmentando dados com acc_segmentation() ...")
    segments, acts, starts = acc_segmentation(data_np)

    print(f"   > Segmentos válidos encontrados: {len(segments)}")
    if len(segments) == 0:
        raise RuntimeError("Nenhum segmento encontrado!")

    # carregar modelo
    print(" - Carregando modelo HARNet5 ...")
    encoder = load_model()
    encoder.eval()

    embeddings = []
    participants = []
    devices = []

    print(" - Extraindo embeddings ...")
    for i, (seg, act) in enumerate(zip(segments, acts)):

        acc_resampled, _ = resample_to_30hz_5s(seg, fs_in_hz=51.5)

        tens = torch.tensor(acc_resampled, dtype=torch.float32).permute(1, 0).unsqueeze(0)

        with torch.no_grad():
            emb = encoder(tens)

        emb_np = emb.cpu().numpy().squeeze()
        embeddings.append(emb_np)

        # METADATA REAL
        participants.append(int(dev["participant"].iloc[starts[i]]))
        devices.append(int(dev["device"].iloc[starts[i]]))

    # construir DataFrame final
    emb_dim = len(embeddings[0])
    col_names = [f"emb_{i}" for i in range(emb_dim)]

    df_emb = pd.DataFrame(embeddings, columns=col_names)
    df_emb["activity"] = acts
    df_emb["participant"] = participants
    df_emb["device"] = devices

    print(f"\n - EMBEDDINGS DATASET criado com shape: {df_emb.shape}")

    if save_csv:
        df_emb.to_csv("embeddings_dataset.csv", index=False)
        print(" - Gravado como embeddings_dataset.csv")

    return df_emb


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("***** 0 - Importando dados *****")
    dev1, dev2, dev3, dev4, dev5 = importData()

    print("\n***** 1.1 - Analisando equilíbrio entre as atividades *****")
    analisarEquilibrio(dev1, "Dispositivo 1")
    analisarEquilibrio(dev2, "Dispositivo 2")
    analisarEquilibrio(dev3, "Dispositivo 3")
    analisarEquilibrio(dev4, "Dispositivo 4")
    analisarEquilibrio(dev5, "Dispositivo 5")

    # SMOTE exemplo
    print("\n***** 1.3 - Extraindo dados do participante 3 *****")
    part3 = obter_participante(dev1, dev2, dev3, dev4, dev5, 3)

    print("\n***** 1.3 - Gerando 3 novas amostras *****")
    part3_smote, novas_amostras = gerarSMOTE(part3, atividade=4, k_novos=3)

    # Geração de embeddings do dispositivo 2
    print("\n***** 2.1 - Criando o EMBEDDINGS DATASET (Dispositivo 2) *****")
    df_embeddings = gerar_embeddings_dataset(dev2, device_id=2, save_csv=True)