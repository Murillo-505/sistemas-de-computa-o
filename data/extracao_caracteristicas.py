from pathlib import Path
import numpy as np
import pandas as pd

# Localiza automaticamente a pasta raiz (TESDC) e a pasta de resultados (results/data)
BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results" / "data"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 65)
print("PARTE 4: EXTRAÇÃO DE CARACTERÍSTICAS (BCI P300)")
print("=" * 65)

# 1. Carregar os dados já limpos na Parte 3
X_limpo = np.load(RESULTS_DIR / "dados_limpos_X.npy")
y = np.load(RESULTS_DIR / "rotulos_y.npy")

# Mapeamento dos 9 canais parieto-occipitais mantidos na limpeza
canais_limpos = ['Cz', 'P3', 'Pz', 'P4', 'P7', 'P8', 'O1', 'Oz', 'O2']
n_trials, n_canais, n_amostras = X_limpo.shape

# Eixo temporal correspondente às 33 amostras decimadas (-100 ms a +700 ms)
tempo_ms = np.linspace(-100, 700, n_amostras)

# Converter rótulos numéricos do MNE para nomes legíveis
classe_alvo = np.max(y)
y_labels = np.where(y == classe_alvo, "Target", "NonTarget")

print(f"• Dados carregados de: {RESULTS_DIR}")
print(f"• Shape carregado: {X_limpo.shape} (trials, canais, pontos no tempo)")

# ----------------------------------------------------------------------
# ABORDAGEM A: Vetores Temporais Achatados (Para Classificação / LDA)
# ----------------------------------------------------------------------
mask_pos_estimulo = (tempo_ms >= 0) & (tempo_ms <= 650)
X_janela = X_limpo[:, :, mask_pos_estimulo]
n_pontos_uteis = X_janela.shape[2]

topologias = {
    "9_canais": ['Cz', 'P3', 'Pz', 'P4', 'P7', 'P8', 'O1', 'Oz', 'O2'],
    "4_canais_capati": ['Pz', 'Oz', 'O1', 'O2'],
    "2_canais_minimo": ['Pz', 'Oz']
}

print(f"\n[A] Extração de Vetores Temporais (Janela 0–650 ms -> {n_pontos_uteis} pontos/canal):")

for nome_topo, lista_canais in topologias.items():
    idx_canais = [canais_limpos.index(c) for c in lista_canais]
    
    X_sub = X_janela[:, idx_canais, :]
    X_features_2d = X_sub.reshape(n_trials, -1)
    
    caminho_saida = RESULTS_DIR / f"features_{nome_topo}.npy"
    np.save(caminho_saida, X_features_2d)
    print(f"  - Topologia '{nome_topo}' ({len(lista_canais)} canais): "
          f"Matriz {X_features_2d.shape} -> Salva em 'results/data/{caminho_saida.name}'")

# ----------------------------------------------------------------------
# ABORDAGEM B: Características Morfológicas da Janela P300 (250–450 ms)
# ----------------------------------------------------------------------
mask_p300 = (tempo_ms >= 250) & (tempo_ms <= 450)
tempo_p300 = tempo_ms[mask_p300]

idx_pz = canais_limpos.index('Pz')
idx_oz = canais_limpos.index('Oz')

sinal_pz_uV = X_limpo[:, idx_pz, :][:, mask_p300] * 1e6
sinal_oz_uV = X_limpo[:, idx_oz, :][:, mask_p300] * 1e6

df_features_morf = pd.DataFrame({
    "Classe": y_labels,
    "Pz_Pico_Max_uV": np.max(sinal_pz_uV, axis=1),
    "Pz_Media_uV": np.mean(sinal_pz_uV, axis=1),
    "Pz_Latencia_Pico_ms": tempo_p300[np.argmax(sinal_pz_uV, axis=1)],
    "Pz_Area_Sinal": np.trapezoid(sinal_pz_uV, x=tempo_p300, axis=1),
    "Oz_Pico_Max_uV": np.max(sinal_oz_uV, axis=1),
    "Oz_Media_uV": np.mean(sinal_oz_uV, axis=1)
})

caminho_csv = RESULTS_DIR / "caracteristicas_morfologicas_p300.csv"
df_features_morf.to_csv(caminho_csv, index=False)

print("\n[B] Extração de Características Morfológicas (Janela 250–450 ms):")
print(f"  - Tabela salva em 'results/data/{caminho_csv.name}'")
print("\nResumo Médio por Classe (Validando separabilidade):")
print(df_features_morf.groupby("Classe")[["Pz_Pico_Max_uV", "Pz_Media_uV", "Pz_Latencia_Pico_ms", "Oz_Media_uV"]].mean().round(2))
print("=" * 65)