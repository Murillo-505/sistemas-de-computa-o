import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import decimate

try:
    from moabb.datasets import BI2014a as DatasetP300
except ImportError:
    from moabb.datasets import BNCI2014_009 as DatasetP300

from moabb.paradigms import P300
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

print("1. Carregando dados e aplicando downsampling...")
dataset = DatasetP300()
paradigm = P300(fmin=1.0, fmax=20.0)
channels = ['Fp1', 'Fp2', 'F3', 'AFz', 'F4', 'T7', 'Cz', 'T8', 'P7', 'P3', 'Pz', 'P4', 'P8', 'O1', 'Oz', 'O2']

X, y, metadata = paradigm.get_data(dataset=dataset, subjects=[1])
X_down = decimate(X, q=16, axis=-1)
channels_np = np.array(channels)

# Mapear classes ('Target' -> 1, 'NonTarget' -> 0)
y_bin = np.where(y == 'Target', 1, 0) if y.dtype.kind in {'U', 'S', 'O'} else y

# Separar índices de treino (primeiros 70%) e teste (últimos 30%) mantendo ordem temporal
split_idx = int(len(y_bin) * 0.7)
train_idx = np.arange(split_idx)
test_idx = np.arange(split_idx, len(y_bin))

subconjuntos = {
    "9 Canais (Ótimo)": [c for c in ['Cz', 'P3', 'Pz', 'P4', 'P7', 'P8', 'O1', 'Oz', 'O2'] if c in channels],
    "4 Canais (Capati)": [c for c in ['Pz', 'Oz', 'O1', 'O2'] if c in channels],
    "2 Canais (Mínimo)": [c for c in ['Pz', 'Oz'] if c in channels]
}

repeticoes = [1, 2, 4, 6, 8, 12]
resultados = []

print("2. Simulando ensemble averaging...")

for nome_subset, canais_subset in subconjuntos.items():
    indices = [np.where(channels_np == c)[0][0] for c in canais_subset]
    X_sub = X_down[:, indices, :]
    X_flat = X_sub.reshape(X_sub.shape[0], -1)

    # Treina o classificador no trial individual (single-trial)
    clf = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(solver='svd'))
    clf.fit(X_flat[train_idx], y_bin[train_idx])

    # Avaliação no conjunto de teste com médias de blocos
    X_test = X_flat[test_idx]
    y_test = y_bin[test_idx]

    # Separar trials alvo e não-alvo no conjunto de teste
    idx_target = np.where(y_test == 1)[0]
    idx_nontarget = np.where(y_test == 0)[0]

    for n_rep in repeticoes:
        aucs_rep = []
        # Faz simulações de sorteios aleatórios de médias (Bootstrap de repetições)
        for _ in range(30):
            # Amostra aleatória de n_rep trials para formar a média
            n_samples = min(len(idx_target) // n_rep, len(idx_nontarget) // n_rep)
            if n_samples < 5:
                continue

            target_blocks = np.random.choice(idx_target, (n_samples, n_rep), replace=False)
            nontarget_blocks = np.random.choice(idx_nontarget, (n_samples, n_rep), replace=False)

            # Calcula a média (ensemble average)
            X_avg_target = np.mean(X_test[target_blocks], axis=1)
            X_avg_nontarget = np.mean(X_test[nontarget_blocks], axis=1)

            X_eval = np.vstack([X_avg_target, X_avg_nontarget])
            y_eval = np.array([1] * n_samples + [0] * n_samples)

            # Predição de probabilidade
            probs = clf.predict_proba(X_eval)[:, 1]
            aucs_rep.append(roc_auc_score(y_eval, probs))

        if aucs_rep:
            resultados.append({
                "Configuração": nome_subset,
                "Repetições (Flashes)": n_rep,
                "ROC AUC": np.mean(aucs_rep)
            })

df_rep = pd.DataFrame(resultados)

# 3. Plotar Curva de Latência vs Performance
plt.figure(figsize=(9, 5.5))
sns.lineplot(
    data=df_rep,
    x="Repetições (Flashes)",
    y="ROC AUC",
    hue="Configuração",
    marker="o",
    linewidth=2.5,
    markersize=8
)

plt.axhline(0.9, color='gray', linestyle=':', label='Meta de Alta Confiabilidade (0.90)')
plt.title("Curva de Compensação: Número de Repetições vs. Redução de Canais")
plt.xlabel("Número de Repetições de Estímulo (Ensemble Averaging)")
plt.ylabel("Área sob a Curva ROC (AUC)")
plt.ylim(0.7, 1.02)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig("curva_repeticoes.png", dpi=300)
plt.close()

print("\nConcluído com sucesso! Gráfico gerado: curva_repeticoes.png")