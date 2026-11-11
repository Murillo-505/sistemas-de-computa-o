import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from moabb.datasets import BI2014a as DatasetP300
except ImportError:
    from moabb.datasets import BNCI2014_009 as DatasetP300

from moabb.paradigms import P300

print("="*65)
print("1. CARREGAMENTO E FORMATO ORIGINAL DOS DADOS")
print("="*65)

dataset = DatasetP300()
print(f"• Nome do Dataset: {dataset.code}")
print(f"• Total de sujeitos disponíveis na base: {len(dataset.subject_list)}")
print(f"• Paradigma biológico: P300 (Potenciais Evocados Visuais)")

# Inspecionando a estrutura bruta do Sujeito 1
raw_dict = dataset.get_data(subjects=[1])
print(f"\n• Estrutura de organização interna (Hierarquia):")
sujeito = 1
sessoes = list(raw_dict[sujeito].keys())
print(f"  - Sujeito avaliado: Sujeito {sujeito}")
print(f"  - Sessões encontradas: {sessoes}")

primeira_sessao = sessoes[0]
corridas = list(raw_dict[sujeito][primeira_sessao].keys())
print(f"  - Corridas (Runs/Partidas) na {primeira_sessao}: {len(corridas)} corridas ({corridas})")

# Pega a primeira corrida como amostra do sinal contínuo
raw_exemplo = raw_dict[sujeito][primeira_sessao][corridas[0]]
info = raw_exemplo.info

print(f"\n• Formato Físico do Sinal Contínuo (Raw):")
print(f"  - Taxa de Amostragem: {info['sfreq']} Hz")
print(f"  - Canais presentes: {raw_exemplo.ch_names}")
print(f"  - Formato da matriz contínua da corrida: {raw_exemplo.get_data().shape} (canais x pontos no tempo)")

print("\n" + "="*65)
print("2. ENTENDIMENTO DOS DADOS FOCADO NO NOSSO PROBLEMA (P300)")
print("="*65)

# Agora aplicamos o paradigma para fatiar o sinal nos momentos dos flashes
paradigm = P300(fmin=1.0, fmax=20.0)
X, y, metadata = paradigm.get_data(dataset=dataset, subjects=[1])

n_trials, n_canais, n_amostras = X.shape
tempo_ms = np.linspace(-100, 700, n_amostras)

print(f"• Formato dos dados após fatiamento em eventos (Epoching):")
print(f"  - Matriz X: {X.shape}")
print(f"    * {n_trials} eventos/piscadas gravadas")
print(f"    * {n_canais} canais de eletrodos")
print(f"    * {n_amostras} amostras de tempo por evento (Janela: -100 ms a +700 ms)")

print(f"\n• Distribuição dos Rótulos (Classes do problema):")
contagem = pd.Series(y).value_counts()
print(contagem.to_string())
print(f"  - Razão: {contagem['NonTarget'] / contagem['Target']:.1f} estímulos neutros para cada 1 alvo.")

# ----------------------------------------------------------------------
# 3. VISUALIZAÇÃO: Como o dado bruto se conecta ao problema
# ----------------------------------------------------------------------
plt.figure(figsize=(12, 5))

# Subplot 1: Balanço das classes
plt.subplot(1, 2, 1)
cores = ['#1f77b4', '#d62728']
plt.bar(contagem.index, contagem.values, color=cores, width=0.5)
plt.title("Desbalanceamento Natural do Paradigma Oddball", fontsize=11)
plt.ylabel("Quantidade de Estímulos (Trials)")
for i, v in enumerate(contagem.values):
    plt.text(i, v + 20, str(v), ha='center', fontweight='bold')

# Subplot 2: Comportamento temporal do eletrodo Pz (Onde o P300 vive)
plt.subplot(1, 2, 2)
idx_pz = raw_exemplo.ch_names.index('Pz') if 'Pz' in raw_exemplo.ch_names else 10

media_target = np.mean(X[y == 'Target', idx_pz, :], axis=0) * 1e6
media_nontarget = np.mean(X[y == 'NonTarget', idx_pz, :], axis=0) * 1e6

plt.plot(tempo_ms, media_target, label='Target (Alvo)', color='#d62728', linewidth=2)
plt.plot(tempo_ms, media_nontarget, label='NonTarget (Neutro)', color='#1f77b4', linestyle='--', linewidth=1.5)
plt.axvline(0, color='black', linestyle='-', alpha=0.5, label='Início do Flash (0 ms)')
plt.axvline(300, color='gray', linestyle=':', label='Pico esperado P300')
plt.axvspan(250, 450, color='yellow', alpha=0.15, label='Janela P300')
plt.title(f"Sinal Médio Extraído no Canal {raw_exemplo.ch_names[idx_pz]}", fontsize=11)
plt.xlabel("Tempo pós-estímulo (ms)")
plt.ylabel("Amplitude ($\mu$V)")
plt.legend(loc='upper right', fontsize=8)
plt.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig("entendimento_base_dados.png", dpi=300)
plt.show()

print("\nProcesso finalizado! Imagem salva: entendimento_base_dados.png")