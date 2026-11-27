from pathlib import Path
import matplotlib.pyplot as plt
import mne

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results" / "data"

epochs_brutas = mne.read_epochs(RESULTS_DIR / "bi2014a_s01_bruto-epo.fif", verbose=False)

# Pipeline de limpeza fundamentado na literatura de P300:
# 1. Correção de linha de base (-100 a 0 ms) -> Arvaneh et al. (2019)
# 2. Filtro passa-banda 1.0 a 20.0 Hz -> Amiri et al. (2013)
# 3. Subamostragem (decimação) para 32 Hz -> Amiri et al. (2013)
epochs_limpas = epochs_brutas.copy()
epochs_limpas.apply_baseline((-0.1, 0.0), verbose=False)
epochs_limpas.filter(l_freq=1.0, h_freq=20.0, fir_design='firwin', verbose=False)
epochs_limpas.resample(sfreq=32.0, verbose=False)

# Comparativo visual de um trial individual (antes vs depois) no canal Pz
trial_idx = 5
ch_pz = epochs_brutas.ch_names.index('Pz')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
ax1.plot(epochs_brutas.times * 1000, epochs_brutas['Target'][trial_idx].get_data()[0, ch_pz] * 1e6, color='gray')
ax1.set_title("Trial Bruto (512 Hz, sem filtro/baseline)")
ax1.set_xlabel("Tempo (ms)")
ax1.set_ylabel("Amplitude (µV)")
ax1.grid(True, linestyle='--', alpha=0.5)

ax2.plot(epochs_limpas.times * 1000, epochs_limpas['Target'][trial_idx].get_data()[0, ch_pz] * 1e6, color='#d62728', marker='o', markersize=3)
ax2.set_title("Trial Pré-processado (1-20 Hz, baseline, 32 Hz)")
ax2.set_xlabel("Tempo (ms)")
ax2.set_ylabel("Amplitude (µV)")
ax2.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig(RESULTS_DIR / "comparativo_preprocessamento.png", dpi=300)
plt.close()

caminho_limpo = RESULTS_DIR / "bi2014a_s01_limpo-epo.fif"
epochs_limpas.save(caminho_limpo, overwrite=True)
print(f"Dados limpos salvos em: {caminho_limpo} | Shape: {epochs_limpas.get_data().shape}")