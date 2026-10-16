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
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

print("1. Carregando dados...")
dataset = DatasetP300()
paradigm = P300(fmin=1.0, fmax=20.0)

# Canais EEG reais do dataset
channels = ['Fp1', 'Fp2', 'F3', 'AFz', 'F4', 'T7', 'Cz', 'T8', 'P7', 'P3', 'Pz', 'P4', 'P8', 'O1', 'Oz', 'O2']

X, y, metadata = paradigm.get_data(dataset=dataset, subjects=[1])
print(f"Shape original de X: {X.shape}")

# 2. Downsampling temporal (Decimação)
# Reduz de ~512 Hz para ~32 Hz (fator 16). O sinal P300 opera abaixo de 10 Hz, então não há perda de informação.
print("Aplicando downsampling temporal...")
X_down = decimate(X, q=16, axis=-1)
print(f"Novo shape de X (otimizado): {X_down.shape}")

# 3. Subconjuntos de canais
channels_np = np.array(channels)

c_parieto_occ = [c for c in ['Cz', 'P3', 'Pz', 'P4', 'P7', 'P8', 'O1', 'Oz', 'O2'] if c in channels]
c_capati = [c for c in ['Pz', 'Oz', 'O1', 'O2', 'P7', 'P8'] if c in channels][:4]
c_minimo = [c for c in ['Pz', 'Oz'] if c in channels]

subconjuntos = {
    f"Todos ({len(channels)} canais)": channels,
    f"Parieto-Occipital ({len(c_parieto_occ)} canais)": c_parieto_occ,
    f"Estilo Capati ({len(c_capati)} canais)": c_capati,
    f"Mínimo ({len(c_minimo)} canais)": c_minimo
}

# 4. Avaliação com LDA rápido
resultados = []
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for nome_subset, canais_subset in subconjuntos.items():
    indices = [np.where(channels_np == c)[0][0] for c in canais_subset]
    
    X_sub = X_down[:, indices, :]
    n_trials = X_sub.shape[0]
    X_flat = X_sub.reshape(n_trials, -1)  # Agora tem apenas canais * 33 features
    
    # Solver 'svd' padrão do scikit-learn sem shrinkage pesado é quase instantâneo
    clf = make_pipeline(
        StandardScaler(),
        LinearDiscriminantAnalysis(solver='svd')
    )
    
    scores = cross_val_score(clf, X_flat, y, cv=cv, scoring='roc_auc')
    
    for score in scores:
        resultados.append({
            "Configuração": nome_subset,
            "Qtd Canais": len(canais_subset),
            "ROC AUC": score
        })
    print(f"[{nome_subset}] Concluído - AUC Média: {np.mean(scores):.3f}")

# 5. Salvar gráfico
df_res = pd.DataFrame(resultados)

plt.figure(figsize=(9, 5))
sns.barplot(data=df_res, x="Configuração", y="ROC AUC", palette="Blues_d", capsize=0.1)
plt.axhline(0.5, color='red', linestyle='--', label='Acaso (0.50)')
plt.title("Impacto da Redução de Canais na Detecção de P300")
plt.ylabel("Área sob a Curva ROC (AUC)")
plt.ylim(0.4, 1.0)
plt.legend()
plt.tight_layout()
plt.savefig("degradacao_canais.png", dpi=300)
plt.close()

print("\nProcesso finalizado com sucesso! Imagem salva: degradacao_canais.png")