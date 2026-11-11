import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne
from scipy.signal import decimate

try:
    from moabb.datasets import BI2014a as DatasetP300
except ImportError:
    from moabb.datasets import BNCI2014_009 as DatasetP300

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
# Usamos filtro IIR/FIR padrão para remover DC offset e ruído de rede
raw_filtrado = raw.copy().filter(l_freq=1.0, h_freq=20.0, fir_design='firwin', verbose=False)

# ----------------------------------------------------------------------
# PASSO 2: PODA ESPACIAL DE CANAIS (Eliminação de canais com ruído ocular)
# ----------------------------------------------------------------------
# Mantemos apenas os canais de interesse parietal/occipital para o nosso projeto
canais_interesse = ['Cz', 'P3', 'Pz', 'P4', 'P7', 'P8', 'O1', 'Oz', 'O2']
canais_existentes = [ch for ch in canais_interesse if ch in raw_filtrado.ch_names]
print(f"\n[Passo 2] Isolando canais parieto-occipitais ({len(canais_existentes)} canais): {canais_existentes}")
raw_selecionado = raw_filtrado.copy().pick(canais_existentes)

# ----------------------------------------------------------------------
# PASSO 3: SEGMENTAÇÃO (EPOCHING) COM CORREÇÃO DE LINHA DE BASE
# ----------------------------------------------------------------------
print("\n[Passo 3] Segmentando em épocas (-100 ms a 700 ms pós-estímulo)...")
# Extrair eventos do canal de trigger
eventos, id_eventos = mne.events_from_annotations(raw, verbose=False)

# Mapeamento do BrainInvaders (1 = Não-Alvo, 2 = Alvo)
# Cortamos a janela de -0.1s a 0.7s e subtraímos o baseline (-0.1s a 0s)
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

dados_epochs = epochs.get_data(copy=True) # formato: (trials, canais, amostras)
print(f"  - Formato após segmentação: {dados_epochs.shape}")

# ----------------------------------------------------------------------
# PASSO 4: DECIMAÇÃO TEMPORAL (DOWNSAMPLING PARA ~32 Hz)
# ----------------------------------------------------------------------
print("\n[Passo 4] Aplicando decimação temporal (fator 16)...")
# Reduz o eixo do tempo (último eixo)
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
plt.savefig("comparativo_limpeza_dados.png", dpi=300)
plt.close()

# Salvar os dados limpos em formato numpy para uso imediato no projeto
np.save("dados_limpos_X.npy", dados_limpos)
np.save("rotulos_y.npy", epochs.events[:, -1])

print("\nProcessamento concluído com sucesso!")
print("Arquivos gerados:")
print("• dados_limpos_X.npy (Matriz de dados limpa)")
print("• rotulos_y.npy (Rótulos dos eventos)")
print("• comparativo_limpeza_dados.png (Gráfico antes/depois)")