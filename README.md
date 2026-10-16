# Avaliação de Robustez sob Redução de Canais em BCI P300

Estudo experimental sobre o impacto da subamostragem espacial de eletrodos em interfaces cérebro-computador (BCI) visuais baseadas em potenciais evocados P300, avaliando o trade-off entre contagem de sensores, latência e acurácia.

## Motivação e Hipótese
Aplicações clínicas de BCI tradicionalmente utilizam toucas de 16 a 64 canais. Investigamos empiricamente se dispositivos vestíveis de baixo custo sustentam decodificação robusta (AUC > 85%) e como o cálculo de médias temporais compensa a redução de sensores à custa da taxa de transferência de informação (ITR).

## Dataset
- **Base:** BrainInvaders 2014a (BI2014a), acessada via biblioteca MOABB.
- **Sinal:** EEG registrado a 512 Hz sob estimulação visual discreta.

## Pipeline
1. **Filtragem:** Passa-banda temporal de 1.0 a 20.0 Hz.
2. **Decimação:** *Downsampling* para ~32 Hz (fator 16) para redução de dimensionalidade.
3. **Classificação:** Normalização (*StandardScaler*) + Análise Discriminante Linear (*LDA/SVD*).
4. **Topologias avaliadas:** 16 canais (completo), 9 canais (parieto-occipital), 4 canais posteriores e 2 canais mínimos.

## Execução

```bash
pip install -r requirements.txt

# Executar benchmark de poda de canais
python src/benchmark_canais.py

# Executar simulação de repetições (ensemble averaging)
python src/benchmark_repeticoes.py