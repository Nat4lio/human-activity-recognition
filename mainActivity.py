# =========================
# PROJETO ECAC 2025 - Meta 1
# Autor: João Natálio 2023205576
# =========================


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from scipy.stats import kstest, kruskal, zscore
from scipy.fft import rfft, rfftfreq
from scipy.stats import skew, kurtosis, entropy
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors


# =========================
# 2. Importação dos dados
# =========================
def importData():
    """
    Lê todos os CSV de part0–part14 e adiciona:
        - device
        - participant
    Devolve dev1..dev5 com ambas as colunas incluídas.
    """
    
    cols = ['device', 'acc_x', 'acc_y', 'acc_z',
            'gyro_x', 'gyro_y', 'gyro_z',
            'mag_x', 'mag_y', 'mag_z',
            'timestamp', 'activity']

    dev1_list, dev2_list, dev3_list, dev4_list, dev5_list = [], [], [], [], []

    for i in range(15):            # participants
        for d in range(1, 5 + 1):  # devices
            filename = f'part{i}dev{d}.csv'
            df = pd.read_csv(filename, names=cols)

            df["participant"] = i
            df["device"] = d

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

    print("Dados importados com sucesso.")
    print(f" - Dev2: {len(dev2)} amostras (para feature set final)")

    return dev1, dev2, dev3, dev4, dev5


# =========================
# 3. Criação dos vetores e Cálculo das magnitudes
# =========================
def criarVetores(dev):
    """
    Recebe o DataFrame de um dispositivo e devolve os vetores de
    aceleração, giroscópio e magnetómetro (cada um com 3 eixos).
    """
    acc = dev[['acc_x', 'acc_y', 'acc_z']].values
    gyro = dev[['gyro_x', 'gyro_y', 'gyro_z']].values
    mag = dev[['mag_x', 'mag_y', 'mag_z']].values
    return acc, gyro, mag


def calcularMagnitudes(acc, gyro, mag):
    """
    Calcula os módulos (normas Euclidianas) dos vetores de
    aceleração, giroscópio e magnetómetro.
    """
    acc_mag = np.sqrt(np.sum(acc**2, axis=1))
    gyro_mag = np.sqrt(np.sum(gyro**2, axis=1))
    mag_mag = np.sqrt(np.sum(mag**2, axis=1))
    return acc_mag, gyro_mag, mag_mag


# =========================
# 3.1. Boxplots por atividade (todos os dispositivos)
# =========================
def boxplotsPorDispositivo(dev, acc_mag, gyro_mag, mag_mag, dev_id):
    """
    Gera 3 boxplots (aceleração, giroscópio, magnetómetro) por atividade
    para um determinado dispositivo.
    """
    atividades = dev['activity'].values
    atividades_unicas = np.unique(atividades)

    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    sensores = [('Aceleração', acc_mag), ('Giroscópio', gyro_mag), ('Magnetómetro', mag_mag)]

    for i, (titulo, valores) in enumerate(sensores):
        axs[i].boxplot([valores[atividades == a] for a in atividades_unicas],
                    tick_labels=atividades_unicas)

        axs[i].set_title(f'{titulo} – Dispositivo {dev_id}')
        axs[i].set_xlabel('Atividade')
        axs[i].set_ylabel('Magnitude')
        axs[i].grid(True, linestyle='--', alpha=0.5)

    plt.suptitle(f'Boxplots por Atividade – Dispositivo {dev_id}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    plt.close('all')


# =========================
# 3.2 – 3.4. Análise e deteção de outliers (Tukey e Z-Score)
# =========================
def calcular_densidade_outliers(dev, acc_mag, gyro_mag, mag_mag, k=1.5):
    atividades = dev['activity']
    resultados = []

    for a in np.unique(atividades):
        idx = atividades == a
        def densidade(v):
            q1, q3 = np.percentile(v[idx], [25, 75])
            iqr = q3 - q1
            lower, upper = q1 - k*iqr, q3 + k*iqr
            return np.sum((v[idx] < lower) | (v[idx] > upper)) / len(v[idx]) * 100

        resultados.append({
            'Atividade': int(a),
            'Outliers Aceleração (%)': densidade(acc_mag),
            'Outliers Giroscópio (%)': densidade(gyro_mag),
            'Outliers Magnetómetro (%)': densidade(mag_mag)
        })

    df_outliers = pd.DataFrame(resultados)
    return df_outliers

"""
COMENTÁRIO: 
De forma global, observa-se que as densidades de outliers são relativamente baixas, situando-se maioritariamente abaixo dos 15 %.
Isto indica que a maior parte das amostras está dentro dos limites esperados para o comportamento normal dos sensores, o que 
confirma uma boa qualidade dos dados brutos e ausência de ruído excessivo.

Apenas algumas atividades de transição ou de mudança súbita de movimento apresentam percentagens mais elevadas (acima de 15 %), 
o que é coerente com a natureza dinâmica dessas ações.
"""


def detectarOutliersZscore(dados, k=3.0):
    """
    Deteta outliers num vetor de dados com base no método do Z-score.
    Valores cujo |z| > k são considerados outliers.
    """
    dados = np.array(dados, dtype=float)
    media = np.mean(dados)
    desvio = np.std(dados)
    if desvio == 0:
        return np.zeros_like(dados, dtype=bool)
    z = (dados - media) / desvio
    outliers = np.abs(z) > k
    return outliers


def plotOutliersPorAtividade(atividades, valores, outliers, titulo):
    """
    Mostra os valores por atividade, destacando os outliers a vermelho.
    Cada ponto representa uma amostra da atividade correspondente.
    """
    plt.figure(figsize=(12, 5))
    plt.scatter(atividades[~outliers], valores[~outliers],
                s=6, alpha=0.6, label='Normal', color='blue')
    plt.scatter(atividades[outliers], valores[outliers],
                s=8, alpha=0.8, label='Outlier', color='red')
    plt.title(titulo)
    plt.xlabel('Atividade')
    plt.ylabel('Magnitude')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.show()


# ============================================================
# 3.5. Comparação entre os métodos de deteção de outliers
# ============================================================
"""
COMENTÁRIO:
O método de Tukey (1.5·IQR) detetou percentagens de outliers mais elevadas,
sobretudo nas atividades com maior movimento ou transição,
onde há maior variação de aceleração e ruído nos sensores.

Já o método do Z-score foi mais conservador:
- k = 3.0 → 2.15 % de outliers
- k = 3.5 → 1.29 % de outliers
- k = 4.0 → 0.79 % de outliers

Ou seja, à medida que o limiar k aumenta, apenas os picos mais extremos
são considerados anómalos. Assim, o Tukey é mais sensível a dispersões gerais,
enquanto o Z-score identifica apenas os valores realmente fora do padrão.
Ambos os métodos mostram coerência: as atividades com mais movimento
são naturalmente as mais “ruidosas”.
"""


# ============================================================
# 3.6. Implementação do algoritmo k-means
# ============================================================
def aplicarKMeans(dados, n_clusters=3):
    """
    Aplica o algoritmo k-means a um conjunto de dados.
    Retorna os rótulos atribuídos a cada amostra e as coordenadas dos centroides.
    """
    modelo = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    modelo.fit(dados)
    labels = modelo.labels_
    centroids = modelo.cluster_centers_
    return labels, centroids


# ============================================================
# 3.7. Deteção de outliers multivariáveis (K-means)
# ============================================================
def detetarOutliersKMeans(dados, n_clusters=3, fator_limite=2.0):
    """
    Aplica K-means e deteta outliers com base nas distâncias aos centroides.
    Pontos cuja distância > (média + fator_limite * desvio) são considerados outliers.
    """
    modelo = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    modelo.fit(dados)
    labels = modelo.labels_
    centroids = modelo.cluster_centers_

    # Distância de cada ponto ao respetivo centroide
    _, distancias = pairwise_distances_argmin_min(dados, centroids)

    # Limiar para deteção de outliers
    limiar = np.mean(distancias) + fator_limite * np.std(distancias)
    outliers = distancias > limiar

    return labels, centroids, distancias, outliers, limiar


def plotOutliers3D(dados, labels, outliers, centroids, titulo):
    """
    Mostra em 3D os clusters e destaca os outliers a vermelho.
    """
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Pontos normais
    ax.scatter(dados[~outliers, 0], dados[~outliers, 1], dados[~outliers, 2],
                c=labels[~outliers], cmap='viridis', s=2, alpha=0.5)

    # Outliers
    ax.scatter(dados[outliers, 0], dados[outliers, 1], dados[outliers, 2],
                c='r', s=10, label='Outliers')

    # Centroides
    ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2],
                c='black', s=100, marker='X', label='Centroides')

    ax.set_title(titulo)
    ax.set_xlabel('Acc X')
    ax.set_ylabel('Acc Y')
    ax.set_zlabel('Acc Z')
    ax.legend()
    plt.tight_layout()
    plt.show()


# ============================================================
# 3.7.1. Extra – Análise da distância média por cluster (K-means)
# ============================================================
def analisarClustersKMeans(dados, labels, distancias, centroids):
    """
    Calcula e imprime estatísticas sobre as distâncias dentro de cada cluster.
    Mostra a média, desvio padrão e amplitude das distâncias.
    """
    clusters = np.unique(labels)
    print("\n===== 3.7.1 Análise detalhada dos clusters (K-means) =====")

    for c in clusters:
        mask = labels == c
        media = np.mean(distancias[mask])
        desvio = np.std(distancias[mask])
        min_d = np.min(distancias[mask])
        max_d = np.max(distancias[mask])
        print(f"Cluster {c}: {np.sum(mask)} amostras | "
                f"Média dist={media:.3f} | Desvio={desvio:.3f} | "
                f"Min={min_d:.3f} | Max={max_d:.3f}")

    # Histograma comparativo das distâncias por cluster
    plt.figure(figsize=(10, 5))
    for c in clusters:
        plt.hist(distancias[labels == c], bins=50, alpha=0.6, label=f'Cluster {c}')
    plt.title('Distribuição das distâncias aos centroides (K-means)')
    plt.xlabel('Distância ao centroide')
    plt.ylabel('Frequência')
    plt.legend()
    plt.tight_layout()
    plt.show()


"""
COMENTÁRIO:
O K-means identificou 3 grupos distintos de padrões de aceleração, o que indica
a existência de diferentes regimes de movimento no dispositivo 2 (pulso direito).

Os centroides apresentam direções e magnitudes bem separadas, sugerindo que
cada cluster representa um tipo de movimento — repouso, movimento
moderado e movimento mais intenso.

O limiar de deteção foi 5.46 e cerca de 4.17 % das amostras foram classificadas
como outliers, valor coerente com os resultados univariáveis anteriores.
Estes outliers correspondem a movimentos bruscos ou picos anómalos nos três eixos.

Em resumo:
- O K-means permite detetar outliers considerando simultaneamente as 3 dimensões;
- A percentagem de outliers é consistente com o método de Tukey e Z-score (~2–5%);
- A maioria dos outliers está associada ao cluster mais disperso, com maior distância média
aos centroides — típico de movimentos rápidos ou descontrolados.
"""


# ============================================================
# 4.1. Significância estatística das variáveis por atividade
# ============================================================
def analisarSignificancia(dev, acc_mag, gyro_mag, mag_mag, dev_id=2):
    """
    Avalia a significância estatística das magnitudes (acc, gyro, mag)
    entre as diferentes atividades do dispositivo especificado.
    Passos:
        1. Teste de normalidade (Kolmogorov-Smirnov)
        2. Teste de Kruskal-Wallis (não paramétrico)
    """

    print(f"\n===== 4.1 Significância Estatística (Dispositivo {dev_id}) =====")

    atividades = dev['activity'].values
    atividades_unicas = np.unique(atividades)

    # --- 1. Teste de normalidade (Kolmogorov–Smirnov)
    print("Teste de normalidade (Kolmogorov–Smirnov):")
    for nome, dados in [('acc_mag', acc_mag), ('gyro_mag', gyro_mag), ('mag_mag', mag_mag)]:
        # Normalizamos antes de aplicar o KS test
        dados_norm = zscore(dados)
        ks_stat, p = kstest(dados_norm, 'norm')
        resultado = "Normal" if p >= 0.05 else "Não normal"
        print(f"{nome:<10} → p = {p:.4f} → {resultado}")
    print()

    # --- 2. Teste de Kruskal–Wallis (diferenças entre atividades)
    print("Teste Kruskal-Wallis entre atividades:")
    for nome, dados in [('acc_mag', acc_mag), ('gyro_mag', gyro_mag), ('mag_mag', mag_mag)]:
        grupos = [dados[atividades == a] for a in atividades_unicas if len(dados[atividades == a]) > 0]
        h_stat, p = kruskal(*grupos)
        signif = "p<0.05" if p < 0.05 else "n.s."
        print(f"{nome:<10} → H={h_stat:.2f}, p={p:.4e} → {signif}")


"""
COMENTÁRIO:
As três variáveis analisadas (magnitude da aceleração, giroscópio e magnetómetro)
apresentam distribuições não normais (teste Kolmogorov–Smirnov → p < 0.05),
pelo que é adequado recorrer a um teste não paramétrico.

O teste de Kruskal–Wallis revelou diferenças estatisticamente significativas
entre as atividades (p < 0.05) para todas as variáveis, confirmando que
os valores médios variam de forma consistente com o tipo de movimento.

Conclusão: as magnitudes dos sensores são características discriminantes
relevantes para a classificação das atividades humanas, pois refletem
diferenças reais e consistentes entre os padrões de movimento.
"""


# ==========================================================
# 4.2 - EXTRAÇÃO DE FEATURES TEMPORAIS E ESPECTRAIS
# ==========================================================
def estimar_freq_amostragem(timestamps):
    """
    Estima a frequência de amostragem (Hz) de forma robusta,
    adaptando automaticamente a unidade de timestamp (s, ms ou µs).
    """
    diffs = np.diff(timestamps)
    diffs = diffs[diffs > 0]
    if len(diffs) == 0:
        return 50.0  # fallback padrão

    mean_diff = np.mean(diffs)

    # Se o intervalo médio for muito grande, ajustar unidade
    if mean_diff > 1.0:        # timestamps provavelmente em segundos
        fs = 1.0 / mean_diff
    elif mean_diff > 1e-3:     # timestamps provavelmente em milissegundos
        fs = 1000.0 / (mean_diff * 1000.0)
    elif mean_diff > 1e-6:     # timestamps em microsegundos
        fs = 1e6 / (mean_diff * 1e6)
    else:
        fs = 50.0  # fallback (exemplo típico: 50 Hz)
    
    # Garantir limite mínimo
    if fs < 1.0:
        fs = 50.0  # fallback seguro para sinais de movimento
    return fs


def criar_janelas(df, fs, janela_s=5.0, overlap=0.5):
    """
    Divide o DataFrame em janelas de duração 'janela_s' (em segundos)
    com overlap definido (0.5 = 50%).
    Devolve lista de DataFrames (uma por janela).
    """
    N = int(fs * janela_s)
    passo = int(N * (1 - overlap))
    janelas = []
    
    if passo == 0:
        raise ValueError(f"Passo calculado = 0 (fs={fs}, N={N}) — verifique timestamps!")

    for start in range(0, len(df) - N + 1, passo):
        end = start + N
        j = df.iloc[start:end]
        # Só mantemos se tiver uma única atividade
        if len(j['activity'].unique()) == 1:
            janelas.append(j)
    return janelas


def extrair_features_temporais(v):
    """Calcula features temporais básicas de um vetor."""
    return {
        'mean': np.mean(v),
        'std': np.std(v),
        'median': np.median(v),
        'min': np.min(v),
        'max': np.max(v),
        'iqr': np.percentile(v, 75) - np.percentile(v, 25),
        'rms': np.sqrt(np.mean(v**2)),
        'skew': skew(v),
        'kurtosis': kurtosis(v),
        'zcr': np.sum(np.diff(np.sign(v)) != 0) / len(v)
    }


def extrair_features_espectrais(v, fs):
    """Calcula features espectrais de um vetor."""
    N = len(v)
    # FFT real
    fft_vals = np.abs(rfft(v))
    freqs = rfftfreq(N, d=1/fs)
    psd = fft_vals**2
    psd_norm = psd / np.sum(psd)

    # Energia total
    energia = np.sum(psd) / N
    # Frequência dominante
    freq_dom = freqs[np.argmax(psd)]
    # Centroide espectral
    centroide = np.sum(freqs * psd) / np.sum(psd)
    # Largura de banda
    banda = np.sqrt(np.sum(((freqs - centroide)**2) * psd) / np.sum(psd))
    # Entropia espectral
    entrop = entropy(psd_norm)

    return {
        'spec_energy': energia,
        'spec_domfreq': freq_dom,
        'spec_centroid': centroide,
        'spec_bandwidth': banda,
        'spec_entropy': entrop
    }


def extrair_features_janela(janela, fs):
    """
    Extrai todas as features (temporais + espectrais)
    para uma janela de dados contendo as colunas:
    acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, mag_x, mag_y, mag_z.
    """
    features = {}
    sensores = ['acc', 'gyro', 'mag']
    for s in sensores:
        for eixo in ['x', 'y', 'z']:
            col = f'{s}_{eixo}'
            v = janela[col].values
            ftemp = extrair_features_temporais(v)
            ftemp = {f'{col}_{k}': v for k, v in ftemp.items()}
            fspec = extrair_features_espectrais(v, fs)
            fspec = {f'{col}_{k}': v for k, v in fspec.items()}
            features.update(ftemp)
            features.update(fspec)

    # Magnitudes também (norma dos 3 eixos)
    for s in sensores:
        mag = np.sqrt(np.sum(janela[[f'{s}_x', f'{s}_y', f'{s}_z']].values**2, axis=1))
        ftemp = extrair_features_temporais(mag)
        ftemp = {f'{s}_mag_{k}': v for k, v in ftemp.items()}
        fspec = extrair_features_espectrais(mag, fs)
        fspec = {f'{s}_mag_{k}': v for k, v in fspec.items()}
        features.update(ftemp)
        features.update(fspec)

    # Label (atividade da janela)
    features['activity'] = int(janela['activity'].iloc[0])
    return features


def gerar_feature_set(dev):
    """
    Cria o feature set completo APENAS para dev2.
    Inclui campo 'participant' para consistência com embeddings.
    """
    print("A gerar feature set do dispositivo 2...")

    fs = estimar_freq_amostragem(dev['timestamp'].values)
    print(f"Frequência estimada: {fs:.2f} Hz")

    janelas = criar_janelas(dev, fs, janela_s=5.0, overlap=0.5)
    print(f"Número de janelas válidas: {len(janelas)}")

    feature_list = []

    for j in janelas:
        f = extrair_features_janela(j, fs)
        f["participant"] = int(j["participant"].iloc[0])   # <-- ADICIONADO
        f["device"] = 2                                     # <-- ADICIONADO
        feature_list.append(f)

    df_features = pd.DataFrame(feature_list)
    print(f"Feature set gerado com shape: {df_features.shape}")

    return df_features


# ==========================================================
# 4.3 - PCA (Análise de Componentes Principais)
# ==========================================================
def aplicarPCA(df_features, var_target=0.75):
    """
    Aplica PCA ao feature set:
        1. Normaliza as features (Z-score)
        2. Calcula as componentes principais
        3. Determina o nº mínimo de componentes que explicam 'var_target' (ex: 75%)
        4. Retorna o modelo PCA, os dados projetados e estatísticas
    """
    print("\n===== 4.3 Análise PCA =====")

    # Separar features (X) e labels (y)
    X = df_features.drop(columns=['activity']).values
    y = df_features['activity'].values

    # Normalização (Z-score)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Aplicar PCA
    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)

    # Variância explicada acumulada
    var_exp = np.cumsum(pca.explained_variance_ratio_)
    n_components = np.argmax(var_exp >= var_target) + 1

    print(f"Número mínimo de componentes para {var_target*100:.0f}% da variância: {n_components}")
    print(f"Variância total explicada pelas {n_components} primeiras componentes: {var_exp[n_components-1]*100:.2f}%")

    # Gráfico Scree Plot
    plt.figure(figsize=(8, 5))
    plt.plot(np.arange(1, len(var_exp)+1), var_exp*100, marker='o')
    plt.axhline(y=var_target*100, color='r', linestyle='--', label=f'{var_target*100:.0f}% alvo')
    plt.axvline(x=n_components, color='g', linestyle='--', label=f'{n_components} componentes')
    plt.title('Variância Acumulada - PCA')
    plt.xlabel('Número de Componentes Principais')
    plt.ylabel('Variância Explicada (%)')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return pca, X_pca, y, scaler


# ==========================================================
# 4.4 / 4.4.1 – Interpretação e projeção no espaço reduzido (PCA)
# ==========================================================
def analisarImportanciaComponentes(pca, n_componentes=15):
    """
    Mostra a importância (variância explicada) de cada componente principal.
    """
    variancias = pca.explained_variance_ratio_[:n_componentes] * 100
    print("\n===== 4.4 Importância das Componentes Principais =====")
    for i, v in enumerate(variancias, 1):
        print(f"Componente {i:02d} → {v:.2f}% da variância")

    plt.figure(figsize=(8,5))
    plt.bar(range(1, n_componentes+1), variancias, color='steelblue')
    plt.title('Importância das Componentes Principais (Top 15)')
    plt.xlabel('Componente Principal')
    plt.ylabel('Variância Explicada (%)')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()


def projetarExemploPCA(X_pca, y, n_componentes=3):
    """
    Mostra a projeção das janelas no espaço PCA reduzido (2D ou 3D).
    """
    from mpl_toolkits.mplot3d import Axes3D
    if n_componentes == 2:
        plt.figure(figsize=(8,6))
        scatter = plt.scatter(X_pca[:,0], X_pca[:,1], c=y, cmap='tab20', s=10, alpha=0.6)
        plt.title('Projeção PCA – 2 Primeiras Componentes')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.colorbar(scatter, label='Atividade')
        plt.grid(True, linestyle='--', alpha=0.4)
        plt.tight_layout()
        plt.show()
    elif n_componentes >= 3:
        fig = plt.figure(figsize=(9,7))
        ax = fig.add_subplot(111, projection='3d')
        p = ax.scatter(X_pca[:,0], X_pca[:,1], X_pca[:,2], c=y, cmap='tab20', s=10, alpha=0.6)
        ax.set_title('Projeção PCA – 3 Primeiras Componentes')
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')
        fig.colorbar(p, label='Atividade')
        plt.tight_layout()
        plt.show()


"""
COMENTÁRIO:
A reconstrução no espaço reduzido via PCA demonstrou ser uma estratégia poderosa para compressão e análise de
sinais multivariados, preservando grande parte da estrutura estatística dos dados e facilitando a visualização
e o processamento.
Contudo, trata-se de uma técnica linear e global, não ideal para fenómenos altamente não lineares ou variáveis 
fortemente dependentes do contexto físico.
Assim, o PCA é excelente para pré-processamento e redução de ruído, mas deve ser complementado com métodos não 
lineares ou supervisionados quando o objetivo é classificação de atividades complexas.
"""


# ==========================================================
# 4.5 – Seleção de Features pelo Fisher Score
# ==========================================================
def calcular_fisher_score(X, y):
    """
    Calcula o Fisher Score para cada feature em X, considerando as classes em y.
    Retorna um DataFrame ordenado por score decrescente.
    """
    scores = []
    n_features = X.shape[1]
    classes = np.unique(y)
    media_global = np.mean(X, axis=0)

    for i in range(n_features):
        num, den = 0, 0
        for c in classes:
            Xc = X[y == c, i]
            n_c = len(Xc)
            if n_c == 0:
                continue
            media_c = np.mean(Xc)
            var_c = np.var(Xc)
            num += n_c * (media_c - media_global[i])**2
            den += n_c * var_c
        score = num / den if den != 0 else 0
        scores.append(score)

    df_scores = pd.DataFrame({
        'Feature': [f'f{i+1}' for i in range(n_features)],
        'Fisher_Score': scores
    }).sort_values('Fisher_Score', ascending=False).reset_index(drop=True)

    return df_scores


# ==========================================================
# 4.5 – Seleção de Features pelo ReliefF
# ==========================================================
def calcular_reliefF(X, y, k=10):
    """
    Implementação simplificada do algoritmo ReliefF.
    Retorna um DataFrame com o peso de cada feature.
    """
    n_samples, n_features = X.shape
    weights = np.zeros(n_features)
    nn = NearestNeighbors(n_neighbors=k+1)
    nn.fit(X)
    neighbors = nn.kneighbors(X, return_distance=False)

    for i in range(n_samples):
        xi, yi = X[i], y[i]
        hits, misses = [], []
        for n_idx in neighbors[i][1:]:
            if y[n_idx] == yi:
                hits.append(X[n_idx])
            else:
                misses.append(X[n_idx])
        if not hits or not misses:
            continue
        hits = np.mean(np.abs(hits - xi), axis=0)
        misses = np.mean(np.abs(misses - xi), axis=0)
        weights += misses - hits

    weights /= n_samples
    df_relief = pd.DataFrame({
        'Feature': [f'f{i+1}' for i in range(n_features)],
        'ReliefF_Score': weights
    }).sort_values('ReliefF_Score', ascending=False).reset_index(drop=True)
    return df_relief


# ============================================================
# 4.6.2 – Exemplo de visualização das features mais importantes
# ============================================================
"""
Comentário — Comparação entre Fisher Score e ReliefF
Nesta etapa, foram comparadas duas abordagens de seleção de features:

- Fisher Score:
    Método estatístico supervisionado que mede a razão entre a variância interclasse e intraclasse.
    Quanto maior o valor, maior a capacidade da feature em distinguir entre diferentes atividades.
    É simples, rápido e eficaz para dados linearmente separáveis, mas ignora correlações e relações não lineares.

- ReliefF:
    Método baseado em vizinhança (kNN) que avalia o quanto cada feature contribui para diferenciar
    amostras próximas de classes distintas. Considera interações e dependências entre variáveis,
    sendo mais robusto em contextos complexos e não lineares, embora mais exigente computacionalmente.

A comparação mostra que as features de maior relevância variam entre os métodos:
    - Fisher destacou variáveis relacionadas com energia, dispersão e variabilidade dos sensores.
    - ReliefF identificou outras de maior peso contextual, refletindo a influência de relações não lineares
    entre medições.

Conclusão:
Enquanto o Fisher Score privilegia a separabilidade estatística direta, o ReliefF capta padrões
mais subtis e contextuais. A combinação de ambos resulta numa seleção de features mais equilibrada,
unindo simplicidade, interpretabilidade e robustez.
"""


# ============================================================
# Programa principal
# ============================================================
if __name__ == "__main__":
# ------------------------------------------------------------
# 2. Importar dados
# ------------------------------------------------------------
    dev1, dev2, dev3, dev4, dev5 = importData()
    

# ------------------------------------------------------------
# 3. Análise exploratória dos dados
# ------------------------------------------------------------
    acc1, gyro1, mag1 = criarVetores(dev1)
    acc2, gyro2, mag2 = criarVetores(dev2)
    acc3, gyro3, mag3 = criarVetores(dev3)
    acc4, gyro4, mag4 = criarVetores(dev4)
    acc5, gyro5, mag5 = criarVetores(dev5)

    acc1_mag, gyro1_mag, mag1_mag = calcularMagnitudes(acc1, gyro1, mag1)
    acc2_mag, gyro2_mag, mag2_mag = calcularMagnitudes(acc2, gyro2, mag2)
    acc3_mag, gyro3_mag, mag3_mag = calcularMagnitudes(acc3, gyro3, mag3)
    acc4_mag, gyro4_mag, mag4_mag = calcularMagnitudes(acc4, gyro4, mag4)
    acc5_mag, gyro5_mag, mag5_mag = calcularMagnitudes(acc5, gyro5, mag5)


# ------------------------------------------------------------
# 3.1. Boxplots por atividade (aceleração, giroscópio e magnetómetro)
# ------------------------------------------------------------
    print("\n===== 3.1 Boxplots por atividade =====")

    boxplotsPorDispositivo(dev1, acc1_mag, gyro1_mag, mag1_mag, dev_id=1)
    plt.close('all')
    boxplotsPorDispositivo(dev2, acc2_mag, gyro2_mag, mag2_mag, dev_id=2)
    plt.close('all')
    boxplotsPorDispositivo(dev3, acc3_mag, gyro3_mag, mag3_mag, dev_id=3)
    plt.close('all')
    boxplotsPorDispositivo(dev4, acc4_mag, gyro4_mag, mag4_mag, dev_id=4)
    plt.close('all')
    boxplotsPorDispositivo(dev5, acc5_mag, gyro5_mag, mag5_mag, dev_id=5)
    plt.close('all')


# ------------------------------------------------------------
# 3.2. Densidade de outliers (Tukey 1.5·IQR) - Dispositivo 2 (Pulso Direito)
# ------------------------------------------------------------
    print("\n===== 3.2 Densidade de Outliers (Tukey 1.5·IQR) - Dispositivo 2 =====")
    # usa explicitamente as variáveis do dispositivo 2
    df_densidade_dev2 = calcular_densidade_outliers(dev2, acc2_mag, gyro2_mag, mag2_mag)
    print(df_densidade_dev2.to_string(index=False))


# ------------------------------------------------------------
# 3.3–3.4. Deteção e visualização de outliers via Z-Score (por atividade)
# ------------------------------------------------------------
    print("\n===== 3.3–3.4 Deteção e Visualização de Outliers (Z-Score) – Dispositivo 2 =====")

    valores = acc2_mag
    atividades = dev2['activity'].values  # usar atividade em vez de timestamp

    for k in [3.0, 3.5, 4.0]:
        outliers = detectarOutliersZscore(valores, k)
        percent = np.sum(outliers) / len(valores) * 100
        print(f"k = {k} → {np.sum(outliers)} outliers ({percent:.2f}%)")
        plotOutliersPorAtividade(
            atividades, valores, outliers,
            f'3.4 - Z-score (k={k}) – Aceleração por Atividade (Dispositivo 2)'
        )
        plt.close('all')


# ------------------------------------------------------------
# 3.6 - Teste básico do K-means (Dispositivo 2)
# ------------------------------------------------------------
    print("\n===== 3.6 Teste do algoritmo K-means – Dispositivo 2 =====")

# Usar os 3 eixos da aceleração do dispositivo 2
    dados = acc2  # matriz Nx3 (acc_x, acc_y, acc_z)

# Aplicar K-means com 3 clusters
    modelo = KMeans(n_clusters=3, n_init=10, random_state=42)
    modelo.fit(dados)

    labels = modelo.labels_
    centroids = modelo.cluster_centers_

# Mostrar resultados
    print(f"Número de clusters: {len(np.unique(labels))}")
    print("Centroides calculados:")
    print(centroids)

    
# ------------------------------------------------------------
# 3.7 - Deteção de outliers multivariáveis (K-means)
# ------------------------------------------------------------
    print("\n===== 3.7 Deteção de outliers multivariáveis (K-means) – Dispositivo 2 =====")

# Aplicar o método
    labels, centroids, distancias, outliers, limiar = detetarOutliersKMeans(acc2, n_clusters=3, fator_limite=2.0)

    print(f"Total de amostras: {len(acc2)}")
    print(f"Número de outliers detetados: {np.sum(outliers)} ({np.mean(outliers)*100:.2f}%)")
    print(f"Limiar de deteção: {limiar:.4f}")

# Mostrar visualização 3D
    plotOutliers3D(acc2, labels, outliers, centroids,
                '3.7 - Deteção de Outliers Multivariáveis (K-means – Dispositivo 2)')
    plt.close('all')


# ------------------------------------------------------------
# 3.7.1 - Análise detalhada dos clusters
# ------------------------------------------------------------
    analisarClustersKMeans(acc2, labels, distancias, centroids)
    plt.close('all')


# ------------------------------------------------------------
# 4.1 – Significância estatística das variáveis (Dispositivo 2)
# ------------------------------------------------------------
    analisarSignificancia(dev2, acc2_mag, gyro2_mag, mag2_mag, dev_id=2)
    
    
# ------------------------------------------------------------
# 4.2 – Extra: Geração do feature set completo (Dispositivo 2)
# ------------------------------------------------------------
    df_features_dev2 = gerar_feature_set(dev2)


# ------------------------------------------------------------
# 4.3 - Análise PCA (Dispositivo 2)
# ------------------------------------------------------------
    pca, X_pca, y_pca, scaler_pca = aplicarPCA(df_features_dev2, var_target=0.75)
    
    
# ------------------------------------------------------------
# 4.4 / 4.4.1 – Mostrar importância e projeção
# ------------------------------------------------------------
    analisarImportanciaComponentes(pca, n_componentes=15)
    #projetarExemploPCA(X_pca, y_pca, n_componentes=3)
    
    
# ------------------------------------------------------------
# 4.5 - Aplicar Fisher Score ao dataset de features
# ------------------------------------------------------------
    X = df_features_dev2.drop(columns=['activity']).values
    y = df_features_dev2['activity'].values

    df_fisher = calcular_fisher_score(X, y)

    print("\n===== 4.5.1 Top 10 Features pelo Fisher Score =====")
    print(df_fisher.head(10))

    plt.figure(figsize=(10,5))
    plt.bar(df_fisher['Feature'][:10], df_fisher['Fisher_Score'][:10])
    plt.title('Top 10 Features – Fisher Score')
    plt.xlabel('Feature')
    plt.ylabel('Fisher Score')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------
# 4.6 – Comparação Fisher Score vs ReliefF
# ------------------------------------------------------------
    df_relief = calcular_reliefF(X, y, k=10)

    print("\n===== 4.6 Top 10 Features pelo ReliefF =====")
    print(df_relief.head(10))

# Combinar e comparar
    df_comparacao = pd.merge(
        df_fisher.head(10), df_relief.head(10),
        on='Feature', how='outer'
    ).fillna('-')

    print("\n===== Comparação Fisher Score vs ReliefF =====")
    print(df_comparacao)

# Visualização comparativa
    plt.figure(figsize=(10,5))
    plt.bar(df_fisher['Feature'][:10], df_fisher['Fisher_Score'][:10], alpha=0.7, label='Fisher Score')
    plt.bar(df_relief['Feature'][:10], df_relief['ReliefF_Score'][:10], alpha=0.7, label='ReliefF')
    plt.title('Comparação – Top 10 Features (Fisher vs ReliefF)')
    plt.xlabel('Feature')
    plt.ylabel('Score / Peso Relativo')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------
# 4.6.1 – Exemplo de extração das features mais importantes
# ------------------------------------------------------------
# Obter o índice das 10 melhores features do Fisher
    top10_idx = [int(f[1:]) - 1 for f in df_fisher['Feature'][:10]]  # converte 'f26' → 25

# Mapeia os índices para os nomes reais das colunas do DataFrame
    feature_names = df_features_dev2.drop(columns=['activity']).columns
    top10_realnames = [feature_names[i] for i in top10_idx]

# Escolher um instante (janela) qualquer, ex: 100
    instante = 100
    exemplo = df_features_dev2.loc[instante, top10_realnames + ['activity']]

    print(f"\n===== 4.6.1 Features selecionadas – Janela {instante} =====")
    print(exemplo)
    
    
# ============================================================
# GUARDAR FEATURE SET EM FICHEIRO
# ============================================================
# Guardar ficheiro final
df_features_dev2.to_csv("features_dataset.csv", index=False)

print("FEATURES DATASET criado e guardado como 'features_dataset.csv'")