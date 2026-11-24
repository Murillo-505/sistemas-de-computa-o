from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne
from scipy.signal import decimate

try:
    from moabb.datasets import BI2014a as DatasetP300
except ImportError:
    from moabb.datasets import BNCI2014_009 as DatasetP300

# Localiza automaticamente a pasta raiz (TESDC) e a pasta de resultados (results/data)
BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results" / "data"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

print("="*60)
print("INICIANDO ETAPA DE LIMPEZA E PRÉ-PROCESSAMENTO (SUJEITO 1)")
print("="*60)

# 1. Carregar a primeira sessão/corrida bruta do Sujeito 1
dataset = DatasetP300()
raw_dict = dataset.get_data(subjects=[1])
sessao = list(raw_dict[1].keys())[0]
corrida = list(raw_dict[1][sessao].keys())[0]
raw = raw_dict[1][sessao][corrida].copy()

print(f"• Sinal Bruto Original:")
print(f"  - Taxa original: {raw.info['sfreq']} Hz")
print(f"  - Total de canais: {len(raw.ch_names)}")

# ----------------------------------------------------------------------
# PASSO 1: FILTRAGEM TEMPORAL PASSA-BANDA (1.0 Hz - 20.0 Hz)
# ----------------------------------------------------------------------
print("\n[Passo 1] Aplicando filtro passa-banda (1 a 20 Hz)...")
raw_filtrado = raw.copy().filter(l_freq=1.0, h_freq=20.0, fir_design='firwin', verbose=False)

# ----------------------------------------------------------------------
# PASSO 2: PODA ESPACIAL DE CANAIS (Eliminação de canais com ruído ocular)
# ----------------------------------------------------------------------
canais_interesse = ['Cz', 'P3', 'Pz', 'P4', 'P7', 'P8', 'O1', 'Oz', 'O2']
canais_existentes = [ch for ch in canais_interesse if ch in raw_filtrado.ch_names]
print(f"\n[Passo 2] Isolando canais parieto-occipitais ({len(canais_existentes)} canais): {canais_existentes}")
raw_selecionado = raw_filtrado.copy().pick(canais_existentes)

# ----------------------------------------------------------------------
# PASSO 3: SEGMENTAÇÃO (EPOCHING) COM CORREÇÃO DE LINHA DE BASE
# ----------------------------------------------------------------------
print("\n[Passo 3] Segmentando em épocas (-100 ms a 700 ms pós-estímulo)...")
eventos, id_eventos = mne.events_from_annotations(raw, verbose=False)

tmin, tmax = -0.1, 0.7
baseline = (-0.1, 0.0)

epochs = mne.Epochs(
    raw_selecionado,
    eventos,
    event_id=id_eventos,
    tmin=tmin,
    tmax=tmax,
    baseline=baseline,
    preload=True,
    verbose=False
)

dados_epochs = epochs.get_data(copy=True)
print(f"  - Formato após segmentação: {dados_epochs.shape}")

# ----------------------------------------------------------------------
# PASSO 4: DECIMAÇÃO TEMPORAL (DOWNSAMPLING PARA ~32 Hz)
# ----------------------------------------------------------------------
print("\n[Passo 4] Aplicando decimação temporal (fator 16)...")
dados_limpos = decimate(dados_epochs, q=16, axis=-1)
print(f"  - Formato final dos dados limpos: {dados_limpos.shape}")
print(f"    * {dados_limpos.shape[0]} trials")
print(f"    * {dados_limpos.shape[1]} canais limpos")
print(f"    * {dados_limpos.shape[2]} pontos temporais por trial")

# ----------------------------------------------------------------------
# COMPARAÇÃO VISUAL: ANTES vs. DEPOIS DA LIMPEZA
# ----------------------------------------------------------------------
idx_pz = canais_existentes.index('Pz')
tempo_bruto = epochs.times
tempo_decimado = np.linspace(-100, 700, dados_limpos.shape[-1])

plt.figure(figsize=(11, 5))

# Sinal com ruído de alta frequência
plt.subplot(1, 2, 1)
plt.plot(tempo_bruto * 1000, dados_epochs[0, idx_pz, :] * 1e6, color='gray', alpha=0.8)
plt.title("Sinal Segmentado com Resolução Total (512 Hz)", fontsize=11)
plt.xlabel("Tempo (ms)")
plt.ylabel("Amplitude ($\mu$V)")
plt.grid(True, linestyle='--', alpha=0.5)

# Sinal limpo e decimado
plt.subplot(1, 2, 2)
plt.plot(tempo_decimado, dados_limpos[0, idx_pz, :] * 1e6, color='#1f77b4', marker='o', markersize=3)
plt.title("Sinal Limpo e Decimado (~32 Hz)", fontsize=11)
plt.xlabel("Tempo (ms)")
plt.ylabel("Amplitude ($\mu$V)")
plt.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
caminho_img = RESULTS_DIR / "comparativo_limpeza_dados.png"
plt.savefig(caminho_img, dpi=300)
plt.close()

# Salvar os dados limpos em formato numpy dentro de results/data
caminho_x = RESULTS_DIR / "dados_limpos_X.npy"
caminho_y = RESULTS_DIR / "rotulos_y.npy"
np.save(caminho_x, dados_limpos)
np.save(caminho_y, epochs.events[:, -1])

print("\nProcessamento concluído com sucesso!")
print(f"Arquivos gerados em '{RESULTS_DIR}':")
print(f"• {caminho_x.name} (Matriz de dados limpa)")
print(f"• {caminho_y.name} (Rótulos dos eventos)")
print(f"• {caminho_img.name} (Gráfico antes/depois)")