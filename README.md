# Portfolio Optimizer

Este repositório contém um script robusto em Python desenhado para realizar a **Otimização de Carteira de Investimentos** baseada na Moderna Teoria do Portfólio de Markowitz. O sistema busca automaticamente o histórico de preços dos ativos, calcula a Fronteira Eficiente, e encontra a alocação que maximiza o Índice de Sharpe.

No final do processo, um **relatório completo e formatado em PDF** é gerado com tabelas de pesos ótimos e visualizações detalhadas.

## 🚀 Funcionalidades

- **Captação Automática de Dados**: Integração com o Yahoo Finance (via `yfinance`) buscando todo o histórico de preços de fechamento dos ativos selecionados na B3 ou bolsas globais.
- **Tratamento de Dados e Limpeza**: Identifica automaticamente ativos sem dados suficientes (removendo as colunas com anomalias de formatação antes do Solver) e emite alertas para ativos com histórico inferior a 5 anos.
- **Motor Estatístico**: Calcula o log-retorno diário, médias anualizadas e a matriz de covariância.
- **Otimização Híbrida**:
  - *Monte Carlo*: 20.000 iterações aleatórias vetorizadas explorando o universo possível das alocações.
  - *Solver Matemático (SciPy)*: Algoritmo de minimização SLSQP para descobrir cirurgicamente o ponto exato da Fronteira Eficiente que maximiza o Índice de Sharpe.
- **Visualizações (Gráficos)**: Desempenho Histórico Normalizado (Base 100), Matriz de Correlação (Heatmap), Dispersão Risco x Retorno com a CML (Capital Market Line), e Gráfico de Pizza da alocação ideal.
- **Relatório PDF Automático**: Utiliza a biblioteca `reportlab` para agregar os dados matemáticos e os gráficos de maneira corporativa, comparando a alocação ótima gerada contra uma carteira equiponderada.

## 🛠️ Como Funciona e Como Parametrizar

O arquivo principal é o `portfolio_optimizer.py`.

No topo do arquivo, você encontrará as variáveis de configuração que podem ser modificadas livremente:
```python
TICKERS_YFINANCE = ['DIVO11.SA', 'IVVB11.SA', 'GOLD11.SA', 'BITH11.SA', 'B5P211.SA']
PESOS_MAXIMOS = {
    'DIVO11.SA': 0.40,
    # ...
}
RISK_FREE_RATE = 0.105  # Representando a Selic a 10.5%
```
- **TICKERS_YFINANCE**: Lista com o código dos ativos de interesse (ativos brasileiros costumam levar o sufixo `.SA`).
- **PESOS_MAXIMOS**: Restrição máxima (teto) de alocação que o algoritmo é permitido destinar a um ativo (ex: `0.40` significa que a carteira ótima não poderá ter mais do que 40% neste ativo, impedindo concentração excessiva de risco).
- **RISK_FREE_RATE**: Taxa Livre de Risco (Rf) usada para descontar o Índice de Sharpe e desenhar a *Capital Market Line*.

---

## 💻 Como rodar em outra máquina (Ambiente Virtual)

Este projeto utiliza um ambiente virtual Python (`venv`) para isolar suas dependências. Para configurá-lo e rodar em qualquer máquina ou servidor sem afetar pacotes globais, siga o passo a passo:

### 1. Pré-requisitos
Certifique-se de que a linguagem Python 3 esteja instalada na máquina destino.

### 2. Criar e Ativar o Ambiente Virtual

Abra o terminal na pasta raiz do repositório (onde estão os arquivos) e execute:
```bash
# Cria a pasta do ambiente virtual (chamada venv)
python3 -m venv venv

# Ativa o ambiente virtual (Linux / macOS)
source venv/bin/activate

# Ou, se você estiver operando pelo Windows:
# venv\Scripts\activate
```

### 3. Instalar as Dependências

Com o ambiente ativado (geralmente você verá o indicativo `(venv)` na frente do cursor no terminal), instale as bibliotecas fixas listadas no `requirements.txt`:
```bash
pip install -r requirements.txt
```

### 4. Executar o Script

Pronto! Agora é só rodar o otimizador:
```bash
python3 portfolio_optimizer.py
```
Ao finalizar a execução (geralmente dura apenas alguns segundos), o script criará ou sobrescreverá temporariamente os 4 arquivos `.png` de gráficos no diretório atual, além de entregar o documento PDF final: `relatorio_otimizacao_portfolio.pdf`.
