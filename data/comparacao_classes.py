from pathlib import Path
import matplotlib.pyplot as plt
import mne

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results" / "data"

epochs_limpas = mne.read_epochs(RESULTS_DIR / "bi2014a_s01_limpo-epo.fif", verbose=False)

# Calcula a resposta evocada média (ERP) de cada classe
erp_target = epochs_limpas['Target'].average()
erp_nontarget = epochs_limpas['NonTarget'].average()

tempos_ms = erp_target.times * 1000
canais = erp_target.ch_names

fig, axes = plt.subplots(3, 3, figsize=(14, 10), sharex=True, sharey=True)
axes = axes.flatten()

for idx, canal in enumerate(canais):
    ax = axes[idx]
    onda_t = erp_target.get_data()[idx] * 1e6
    onda_nt = erp_nontarget.get_data()[idx] * 1e6

    ax.plot(tempos_ms, onda_t, color='#d62728', linewidth=2, label='Target (Alvo)')
    ax.plot(tempos_ms, onda_nt, color='#1f77b4', linestyle='--', linewidth=1.5, label='NonTarget')
    ax.axvline(0, color='black', linestyle=':', alpha=0.6)
    ax.axvspan(250, 450, color='gold', alpha=0.18, label='Janela P300')
    ax.axhline(0, color='gray', linewidth=0.7, alpha=0.5)
    ax.set_title(f"Eletrodo {canal}")
    ax.grid(True, linestyle='--', alpha=0.4)

    if idx % 3 == 0:
        ax.set_ylabel("Amplitude (µV)")
    if idx >= 6:
        ax.set_xlabel("Tempo pós-estímulo (ms)")
    if idx == 0:
        ax.legend(loc='upper right', fontsize=8)

plt.suptitle("Comparação das Classes (Target vs NonTarget) nos 9 Canais Pré-processados", fontsize=13)
plt.tight_layout()
caminho_img = RESULTS_DIR / "comparacao_classes_9canais.png"
plt.savefig(caminho_img, dpi=300)
plt.show()
print(f"Gráfico comparativo salvo em: {caminho_img}")