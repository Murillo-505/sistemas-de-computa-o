from pathlib import Path
import numpy as np
import pandas as pd
import mne

try:
    from moabb.datasets import BI2014a as BrainInvaders
except ImportError:
    from moabb.datasets import BNCI2014_009 as BrainInvaders

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results" / "data"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# 1. Carregamento do dataset BrainInvaders2014a (Sujeito 1)
dataset = BrainInvaders()
raw_dict = dataset.get_data(subjects=[1])
sessao = list(raw_dict[1].keys())[0]
corrida = list(raw_dict[1][sessao].keys())[0]
raw_completo = raw_dict[1][sessao][corrida].copy()

print(f"Dataset: {dataset.code} | Frequência: {raw_completo.info['sfreq']} Hz")
print(f"Canais originais ({len(raw_completo.ch_names)}): {raw_completo.ch_names}")

# 2. Poda espacial anatômica (Capati et al., 2016): mantendo região parieto-occipital
canais_alvo = ['Cz', 'P3', 'Pz', 'P4', 'P7', 'P8', 'O1', 'Oz', 'O2']
raw_selecionado = raw_completo.copy().pick(canais_alvo)

# 3. Extração de eventos do canal de trigger (STI 014)
eventos, _ = mne.events_from_annotations(raw_completo, verbose=False)
# No BI2014a: código 1 = NonTarget, código 2 = Target
id_classes = {'NonTarget': 1, 'Target': 2}

epochs_brutas = mne.Epochs(
    raw_selecionado,
    eventos,
    event_id=id_classes,
    tmin=-0.1,
    tmax=0.7,
    baseline=None,
    preload=True,
    verbose=False
)

contagem = pd.Series(epochs_brutas.events[:, -1]).map({1: 'NonTarget', 2: 'Target'}).value_counts()
print("\nDistribuição das classes (Paradigma Oddball):")
print(contagem.to_string())

caminho_fif = RESULTS_DIR / "bi2014a_s01_bruto-epo.fif"
epochs_brutas.save(caminho_fif, overwrite=True)
print(f"\nÉpocas brutas salvas em: {caminho_fif}")