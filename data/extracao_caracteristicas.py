from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import welch
import mne

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results" / "data"

epochs_limpas = mne.read_epochs(RESULTS_DIR / "bi2014a_s01_limpo-epo.fif", verbose=False)
fs = epochs_limpas.info['sfreq']

# Recorta apenas a janela pós-estímulo (0 a 650 ms) para extração
epochs_uteis = epochs_limpas.copy().crop(tmin=0.0, tmax=0.65)
X = epochs_uteis.get_data()  # (n_trials, n_canais, n_amostras)
y = epochs_uteis.events[:, -1]
tempos_ms = epochs_uteis.times * 1000
canais = epochs_uteis.ch_names

# Máscara da janela clássica do componente P300 (250 a 450 ms)
mask_p300 = (tempos_ms >= 250) & (tempos_ms <= 450)
X_p300 = X[:, :, mask_p300]

# 1. Características no Domínio do Tempo (focadas na morfologia do P300)
feat_pico_max = np.max(X_p300, axis=-1)                            # Amplitude máxima na janela P300
feat_media_p300 = np.mean(X_p300, axis=-1)                         # Voltagem média sustentada (250-450 ms)
feat_area_p300 = np.trapezoid(X_p300, axis=-1)                     # Energia/área sob a curva P300
feat_energia_total = np.mean(X ** 2, axis=-1)                      # Potência média na época pós-estímulo

# 2. Características no Domínio da Frequência (Potência na banda Delta/Theta do P300: 1-8 Hz)
freqs, psd = welch(X, fs=fs, nperseg=X.shape[-1], axis=-1)
mask_banda_p300 = (freqs >= 1.0) & (freqs <= 8.0)
feat_potencia_espectral = np.sum(psd[:, :, mask_banda_p300], axis=-1)

# Organiza no tensor (n_trials, n_canais, 5 features) e achata para (n_trials, n_canais * 5)
tensor_features = np.stack(
    [feat_pico_max, feat_media_p300, feat_area_p300, feat_energia_total, feat_potencia_espectral],
    axis=-1
)

# Salva matrizes 2D por topologia de hardware do estudo
topologias = {
    "9_canais": ['Cz', 'P3', 'Pz', 'P4', 'P7', 'P8', 'O1', 'Oz', 'O2'],
    "4_canais_capati": ['Pz', 'Oz', 'O1', 'O2'],
    "2_canais_minimo": ['Pz', 'Oz']
}

for nome, lista_ch in topologias.items():
    indices = [canais.index(c) for c in lista_ch]
    matriz_2d = tensor_features[:, indices, :].reshape(len(X), -1)
    np.save(RESULTS_DIR / f"matriz_features_{nome}.npy", matriz_2d)
    print(f"Topologia '{nome}': shape {matriz_2d.shape} salva em results/data/")

# Salva também o tensor 3D e os rótulos para a Parte 5 (Visualização e Seleção)
np.save(RESULTS_DIR / "tensor_features_3d.npy", tensor_features)
np.save(RESULTS_DIR / "labels_y.npy", y)

# Resumo interpretável em CSV (canais centrais Pz e Oz)
idx_pz, idx_oz = canais.index('Pz'), canais.index('Oz')
df_resumo = pd.DataFrame({
    "Classe": np.where(y == 2, "Target", "NonTarget"),
    "Pz_Pico_uV": feat_pico_max[:, idx_pz] * 1e6,
    "Pz_Media_uV": feat_media_p300[:, idx_pz] * 1e6,
    "Pz_Potencia_1_8Hz": feat_potencia_espectral[:, idx_pz],
    "Oz_Pico_uV": feat_pico_max[:, idx_oz] * 1e6,
    "Oz_Media_uV": feat_media_p300[:, idx_oz] * 1e6,
})
df_resumo.to_csv(RESULTS_DIR / "resumo_features_p300.csv", index=False)
print("\nResumo médio por classe:")
print(df_resumo.groupby("Classe").mean().round(3))