import pandas as pd
import numpy as np
import yfinance as yf
import scipy.optimize as sco
import matplotlib.pyplot as plt
import seaborn as sns
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
import datetime
import os
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 1. PARAMETRIZAÇÃO E INPUTS DE ENTRADA
# ==========================================
TICKERS_YFINANCE = ['DIVO11.SA', 'IVVB11.SA', 'GOLD11.SA', 'BITH11.SA', 'B5P211.SA']
PESOS_MAXIMOS = {
    'DIVO11.SA': 0.40,
    'IVVB11.SA': 0.40,
    'GOLD11.SA': 0.40,
    'BITH11.SA': 0.40,
    'B5P211.SA': 0.40
}

RISK_FREE_RATE = 0.105  # Selic reference 10.5%
NUM_PORTFOLIOS = 20000

# Paleta de Cores
DEEP_NAVY = colors.HexColor('#001F3F')
SLATE_GREY = colors.HexColor('#708090')
WARNING_RED = colors.HexColor('#FF4136')

# ==========================================
# 2. CAPTAÇÃO DE DADOS
# ==========================================
class DataFetcher:
    def __init__(self, tickers):
        self.tickers = tickers
        
    def fetch_data(self):
        print("Baixando dados do Yahoo Finance...")
        data = yf.download(self.tickers, period="max")['Close']
        if isinstance(data, pd.Series):
            data = data.to_frame()
        return data

# ==========================================
# 3. MOTOR ESTATÍSTICO
# ==========================================
class PortfolioStats:
    def __init__(self, data):
        self.data = data
        self.log_returns = None
        self.mean_returns = None
        self.cov_matrix = None
        self.start_dates = {}
        self.history_years = {}
        self.warnings_5y = {}
        
    def calculate_stats(self):
        print("Calculando estatísticas...")
        
        # Limpar ativos que vieram sem nenhum dado
        self.data = self.data.dropna(axis=1, how='all')
        
        self.log_returns = np.log(self.data / self.data.shift(1))
        
        # Média anualizada
        self.mean_returns = self.log_returns.mean() * 252
        
        # Matriz de covariância anualizada (pairwise na)
        self.cov_matrix = self.log_returns.cov() * 252
        
        # Tratar NaNs na covariância ou média que quebram a otimização
        bad_assets = set()
        for col in self.mean_returns.index:
            if pd.isna(self.mean_returns[col]):
                bad_assets.add(col)
        for col in self.cov_matrix.columns:
            if self.cov_matrix[col].isna().any():
                bad_assets.add(col)
                
        if bad_assets:
            print(f"Ativos ignorados por falta de dados suficientes: {bad_assets}")
            valid_assets = [c for c in self.data.columns if c not in bad_assets]
            self.data = self.data[valid_assets]
            self.log_returns = self.log_returns[valid_assets]
            self.mean_returns = self.mean_returns[valid_assets]
            self.cov_matrix = self.cov_matrix.loc[valid_assets, valid_assets]
        
        today = datetime.datetime.today()
        for ticker in self.data.columns:
            s = self.data[ticker].dropna()
            if len(s) > 0:
                start = s.index[0]
                self.start_dates[ticker] = start
                years = (today - start).days / 365.25
                self.history_years[ticker] = years
                self.warnings_5y[ticker] = years < 5
            else:
                self.start_dates[ticker] = pd.NaT
                self.history_years[ticker] = 0
                self.warnings_5y[ticker] = True

# ==========================================
# 4. OTIMIZAÇÃO (HÍBRIDA)
# ==========================================
class Optimizer:
    def __init__(self, tickers, mean_returns, cov_matrix, max_weights):
        self.tickers = tickers
        self.mean_returns = mean_returns
        self.cov_matrix = cov_matrix
        self.max_weights = max_weights
        self.num_assets = len(tickers)
        
    def calc_portfolio_perf(self, weights):
        ret = np.sum(self.mean_returns * weights)
        vol = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
        sharpe = (ret - RISK_FREE_RATE) / vol
        return ret, vol, sharpe

    def run_monte_carlo(self, num_simulations):
        print(f"Executando Monte Carlo ({num_simulations} iterações)...")
        results = np.zeros((3, num_simulations))
        weights_record = []
        
        bounds_arr = np.array([self.max_weights.get(t, 1.0) for t in self.tickers])
        
        # Abordagem vetorizada Dirichlet e rejeição para respeitar bounds
        batch_size = num_simulations * 20
        valid_weights = []
        while len(valid_weights) < num_simulations:
            w = np.random.dirichlet(np.ones(self.num_assets), size=batch_size)
            mask = (w <= bounds_arr).all(axis=1)
            valid = w[mask]
            valid_weights.extend(valid)
            
        valid_weights = np.array(valid_weights[:num_simulations])
        weights_record = valid_weights
        
        for i in range(num_simulations):
            ret, vol, sharpe = self.calc_portfolio_perf(valid_weights[i])
            results[0,i] = ret
            results[1,i] = vol
            results[2,i] = sharpe
            
        return results, weights_record

    def run_scipy_solver(self):
        print("Executando Solver Numérico...")
        def neg_sharpe(weights):
            return -self.calc_portfolio_perf(weights)[2]

        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, self.max_weights.get(t, 1.0)) for t in self.tickers)
        initial_guess = np.array([1.0/self.num_assets] * self.num_assets)
        
        opt_results = sco.minimize(neg_sharpe, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)
        return opt_results.x

    def get_efficient_frontier(self, results):
        print("Traçando Fronteira Eficiente...")
        target_returns = np.linspace(results[0].min(), results[0].max(), 50)
        frontier_vols = []
        
        bounds = tuple((0, self.max_weights.get(t, 1.0)) for t in self.tickers)
        
        for tr in target_returns:
            cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                    {'type': 'eq', 'fun': lambda x: np.sum(x * self.mean_returns) - tr})
            res = sco.minimize(lambda w: np.sqrt(np.dot(w.T, np.dot(self.cov_matrix, w))), 
                               np.array([1.0/self.num_assets]*self.num_assets), 
                               method='SLSQP', bounds=bounds, constraints=cons)
            frontier_vols.append(res.fun)
            
        return target_returns, frontier_vols

# ==========================================
# 5. VISUALIZAÇÕES
# ==========================================
class Visualizer:
    @staticmethod
    def plot_normalized_history(data):
        plt.figure(figsize=(10, 6))
        for col in data.columns:
            s = data[col].dropna()
            if len(s) > 0:
                norm = s / s.iloc[0] * 100
                plt.plot(norm.index, norm, label=col, alpha=0.8)
        plt.title('Desempenho Histórico Normalizado (Base 100)', color='#001F3F')
        plt.xlabel('Data')
        plt.ylabel('Preço Normalizado')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig('grafico1_historico.png', dpi=300)
        plt.close()

    @staticmethod
    def plot_correlation(log_returns):
        plt.figure(figsize=(10, 8))
        corr = log_returns.corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt=".2f", linewidths=.5)
        plt.title('Matriz de Correlação (Pearson)', color='#001F3F')
        plt.tight_layout()
        plt.savefig('grafico2_correlacao.png', dpi=300)
        plt.close()

    @staticmethod
    def plot_efficient_frontier(mc_results, opt_ret, opt_vol, frontier_vols, target_returns):
        plt.figure(figsize=(10, 6))
        plt.scatter(mc_results[1,:], mc_results[0,:], c=mc_results[2,:], cmap='viridis', marker='o', s=10, alpha=0.3)
        plt.colorbar(label='Índice de Sharpe')
        
        plt.plot(frontier_vols, target_returns, 'k--', linewidth=2, label='Fronteira Eficiente (Restrita)')
        plt.scatter(opt_vol, opt_ret, marker='*', color='red', s=500, label='Carteira Ótima (Max Sharpe)')
        
        # Desenhando a Capital Market Line (Linha de Tangência)
        cml_x = np.linspace(0, max(mc_results[1,:]) * 1.1, 100)
        sharpe_ratio = (opt_ret - RISK_FREE_RATE) / opt_vol
        cml_y = RISK_FREE_RATE + sharpe_ratio * cml_x
        plt.plot(cml_x, cml_y, color='blue', linestyle='-', linewidth=1.5, label=f'CML (Tangência) Rf={RISK_FREE_RATE*100:.1f}%')
        
        plt.title('Fronteira Eficiente: Risco x Retorno', color='#001F3F')
        plt.xlabel('Risco Anualizado (Volatilidade)')
        plt.ylabel('Retorno Esperado Anualizado')
        
        # Ajustando a janela (limites) para enquadrar a nuvem e evitar distorção pela CML
        min_vol, max_vol = min(mc_results[1,:]), max(mc_results[1,:])
        min_ret, max_ret = min(mc_results[0,:]), max(mc_results[0,:])
        
        plt.xlim(left=0, right=max_vol * 1.05)
        plt.ylim(bottom=min(RISK_FREE_RATE, min_ret) * 0.9, top=max_ret * 1.15)
        
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig('grafico3_fronteira.png', dpi=300)
        plt.close()

    @staticmethod
    def plot_pie_weights(tickers, weights):
        plt.figure(figsize=(8, 8))
        
        # Filtra pesos pequenos para visualização
        labels = [tickers[i] if weights[i] > 0.01 else "" for i in range(len(tickers))]
        colors_map = plt.cm.tab20(np.linspace(0, 1, len(tickers)))
        
        plt.pie(weights, labels=labels, colors=colors_map, autopct=lambda p: '{:.1f}%'.format(p) if p > 1 else '', 
                startangle=140, textprops={'fontsize': 10})
        plt.title('Alocação Ótima da Carteira', color='#001F3F')
        plt.tight_layout()
        plt.savefig('grafico4_pizza.png', dpi=300)
        plt.close()

# ==========================================
# 6. GERAÇÃO DE PDF
# ==========================================
class ReportGenerator:
    def __init__(self, filename='relatorio_otimizacao_portfolio.pdf'):
        self.filename = filename
        self.styles = getSampleStyleSheet()
        self.doc = SimpleDocTemplate(self.filename, pagesize=A4, rightMargin=30, leftMargin=30, topMargin=30, bottomMargin=30)
        self.elements = []
        
        self.title_style = ParagraphStyle(
            'TitleStyle', parent=self.styles['Heading1'], fontSize=20, textColor=DEEP_NAVY, alignment=1, spaceAfter=20
        )
        self.section_style = ParagraphStyle(
            'SectionStyle', parent=self.styles['Heading2'], fontSize=16, textColor=DEEP_NAVY, spaceBefore=15, spaceAfter=10
        )
        self.text_style = ParagraphStyle(
            'TextStyle', parent=self.styles['Normal'], fontSize=10, textColor=SLATE_GREY, spaceAfter=10, alignment=4
        )

    def add_title(self, text):
        self.elements.append(Paragraph(text, self.title_style))
        
    def add_section(self, text):
        self.elements.append(Paragraph(text, self.section_style))
        
    def add_text(self, text):
        self.elements.append(Paragraph(text, self.text_style))
        
    def add_image(self, path, width=400, height=250):
        if os.path.exists(path):
            img = Image(path, width=width, height=height)
            self.elements.append(img)
            self.elements.append(Spacer(1, 15))

    def add_table(self, data, colWidths=None):
        table = Table(data, colWidths=colWidths)
        style = TableStyle([
            ('BACKGROUND', (0,0), (-1,0), DEEP_NAVY),
            ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTSIZE', (0,0), (-1,0), 12),
            ('BOTTOMPADDING', (0,0), (-1,0), 12),
            ('BACKGROUND', (0,1), (-1,-1), colors.white),
            ('TEXTCOLOR', (0,1), (-1,-1), colors.black),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('GRID', (0,0), (-1,-1), 1, SLATE_GREY)
        ])
        table.setStyle(style)
        self.elements.append(table)
        self.elements.append(Spacer(1, 15))

    def generate(self, tickers, stats, opt_weights, eq_weights):
        print("Gerando Relatório PDF...")
        self.add_title("Relatório de Otimização de Portfólio")
        
        # Introdução
        self.add_text("Este relatório apresenta a análise quantitativa e a otimização de uma carteira de investimentos utilizando a Moderna Teoria do Portfólio de Markowitz. O objetivo primário é encontrar a alocação que maximiza o Índice de Sharpe, respeitando limites máximos de exposição por ativo.")
        
        # Etapa de Dados e Estatísticas
        self.add_section("1. Captação de Dados e Motor Estatístico")
        self.add_text("Os dados foram coletados considerando o histórico máximo disponível de cada ativo. A tabela abaixo resume as estatísticas anuais e inclui a regra de verificação de 5 anos.")
        
        # Tabela Estatísticas
        table_data = [["Ativo", "Retorno Anual", "Volatilidade", "Data Inicial", "Status de Janela"]]
        for t in tickers:
            ret = f"{stats.mean_returns[t]*100:.1f}%"
            vol = f"{np.sqrt(stats.cov_matrix.loc[t,t])*100:.1f}%"
            d_init = stats.start_dates[t].strftime('%Y-%m-%d') if not pd.isna(stats.start_dates[t]) else "N/A"
            
            if stats.warnings_5y[t]:
                status = Paragraph(f"<font color='red'>[AVISO: Histórico reduzido de {stats.history_years[t]:.1f} anos]</font>", self.styles['Normal'])
            else:
                status = f"{stats.history_years[t]:.1f} anos"
            
            table_data.append([t, ret, vol, d_init, status])
            
        self.add_table(table_data)
        
        self.add_image('grafico1_historico.png')
        self.add_image('grafico2_correlacao.png')
        
        # Etapa Otimização
        self.add_section("2. Abordagem Híbrida de Otimização")
        self.add_text("Foram realizadas iterações via Simulação de Monte Carlo para explorar o espaço amostral, seguidas de uma otimização com Solver Numérico (Scipy) para identificar matematicamente o ponto ótimo da Fronteira Eficiente (Máximo Sharpe).")
        
        self.add_image('grafico3_fronteira.png', width=450, height=300)
        self.add_image('grafico4_pizza.png', width=300, height=300)
        
        # Tabela Final
        self.add_section("3. Resultados Finais: Rebalanceamento")
        self.add_text("Comparação entre uma carteira equiponderada e a carteira ótima gerada pelo Solver numérico.")
        
        final_table = [["Ativo", "Peso Máximo Restrito", "Carteira Equiponderada", "Carteira Max Sharpe (Ótima)"]]
        for i, t in enumerate(tickers):
            max_w = f"{PESOS_MAXIMOS.get(t, 1.0)*100:.1f}%"
            eq_w = f"{eq_weights[i]*100:.1f}%"
            opt_w = f"{opt_weights[i]*100:.2f}%"
            final_table.append([t, max_w, eq_w, opt_w])
            
        self.add_table(final_table)
        
        self.doc.build(self.elements)
        print(f"Relatório gerado com sucesso: {self.filename}")

# ==========================================
# 7. EXECUÇÃO PRINCIPAL
# ==========================================
def main():
    tickers_to_fetch = [t for t in TICKERS_YFINANCE if t != 'TESOURO_IPCA']
    
    # 1. Fetch Data
    fetcher = DataFetcher(tickers_to_fetch)
    data = fetcher.fetch_data()
    
    # Check if data was fetched
    if data.empty:
        print("Erro: Nenhum dado foi retornado pelo Yahoo Finance.")
        return
        
    # 2. Stats
    stats = PortfolioStats(data)
    stats.calculate_stats()
    
    # Atualizar lista de tickers válidos após limpeza de NaNs no motor estatístico
    tickers_fetched = list(stats.data.columns)
    
    if not tickers_fetched:
        print("Erro: Nenhum ativo sobrou com dados válidos.")
        return
        
    # Filtrar dados visuais para não ter colunas vazias
    data = stats.data

    
    # 3. Optimize
    optimizer = Optimizer(tickers_fetched, stats.mean_returns, stats.cov_matrix, PESOS_MAXIMOS)
    mc_results, mc_weights = optimizer.run_monte_carlo(NUM_PORTFOLIOS)
    
    opt_weights = optimizer.run_scipy_solver()
    opt_ret, opt_vol, opt_sharpe = optimizer.calc_portfolio_perf(opt_weights)
    
    frontier_x, frontier_y = optimizer.get_efficient_frontier(mc_results)
    
    # 4. Visualize
    print("Gerando gráficos...")
    viz = Visualizer()
    viz.plot_normalized_history(data)
    viz.plot_correlation(stats.log_returns)
    viz.plot_efficient_frontier(mc_results, opt_ret, opt_vol, frontier_y, frontier_x)
    viz.plot_pie_weights(tickers_fetched, opt_weights)
    
    # 5. Report
    eq_weights = np.array([1.0/len(tickers_fetched)] * len(tickers_fetched))
    report = ReportGenerator()
    report.generate(tickers_fetched, stats, opt_weights, eq_weights)
    
if __name__ == "__main__":
    main()
