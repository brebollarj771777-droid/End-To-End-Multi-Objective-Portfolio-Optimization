"""
Multi-Objective Portfolio Optimization Pipeline

Creates a GRU/Dense-based Neural Network Model to obtain expected returns for portfolio's assets
using a exponential moving average, and uses NSGA-II genetic algorithm to optimize portfolio's
metrics such as Sharpe Ratio and Conditional Value At Risk.
"""

#Third Party Library Imports.

import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import sqlite3 as sq3
import pandas as pd
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Input, Dense, GRU, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from sqlalchemy import create_engine

#########################
#USER-DEFINED FUNCTIONS.#
#########################

##########################
#1. Data Window Creation.#
##########################

#Create data windows to improve tendency analysis, with an offset on the Target to avoid look-ahead bias.

def Crear_Ventanas(df,objetivo,ventana):
  X,Y = [],[]
  for i in range(ventana,len(df)):
    X.append(df.values[i-ventana:i])
    Y.append(objetivo.values[i - 1])
  return np.array(X),np.array(Y)

########################################
#2. Data Fetching From SQLite Database.#
########################################

# Select asset data ordered by ticker, processing each asset individually for further analysis.

def Obtencion_De_Datos(ticker):

  with sq3.connect("Activos Portafolio.db") as conexion:
    datos = pd.read_sql(f'''
     SELECT Fecha,
            Ticker,
            Rendimientos_Logaritmicos,
            Media_Movil20Dias,
            Media_Movil10Dias,
            Volatilidad_Movil20Dias,
            Volatilidad_Movil10Dias,
            Volumen_Normalizado_20,
            Volumen_Normalizado_10,
            ATR_Normalizado_20,
            ATR_Normalizado_10,
            RSI_Normalizado_20,
            RSI_Normalizado_10,
            ADX_Normalizado_20,
            ADX_Normalizado_10,
            row_number() OVER (ORDER BY Fecha ASC) AS Fila
     FROM Datos_Modelo WHERE Ticker = '{ticker}'
  ''', conexion)

  return datos

###############################
#3. Features Table Assembling.#
###############################

#Calculate 20-day Window Average Normalized ATR, choosing between 10 or 20-day window metrics based on the result.

def Obtencion_De_Features(datos):
  features = datos.copy()
  
  #Daily volatility threshold set on 1.5%
  umbral_vol_crit_ = 0.015
  
  #To calculate ATR, only future model-training data is used to avoid look-ahead bias.
  limite_entrenamiento = int(len(features) * 0.8)
  volatilidad_promedio = features["ATR_Normalizado_20"].iloc[:limite_entrenamiento].mean()
  if volatilidad_promedio >= umbral_vol_crit_:
    ventana = 10
    features = features[features["Fila"] >= 10]
    features.drop(columns = ["Media_Movil20Dias","Volatilidad_Movil20Dias", "Volumen_Normalizado_20", "ATR_Normalizado_20", "RSI_Normalizado_20","ADX_Normalizado_20" ,"Fila"], inplace = True)
  else:
    ventana = 20
    features = features[features["Fila"] >= 20]
    features.drop(columns = ["Media_Movil10Dias","Volatilidad_Movil10Dias", "Volumen_Normalizado_10" ,"ATR_Normalizado_10", "RSI_Normalizado_10","ADX_Normalizado_10" ,"Fila"], inplace = True)

  return features,ventana, volatilidad_promedio


############################
#4. Target Column Creation.#
############################

#Create target column based on 5-day moving average of logarithmic returns criteria.


def Obtencion_De_Objetivo(datos,features,ventana):
  auxiliar = datos.copy()
  rend_log_auxiliar = auxiliar["Rendimientos_Logaritmicos"].copy()
  rend_log_auxiliar.dropna(inplace = True)
  auxiliar_MA5 = rend_log_auxiliar.rolling(window = 5).mean().shift(1)
  rend_log = features["Rendimientos_Logaritmicos"].copy()
  auxiliar_MA5 = auxiliar_MA5.loc[rend_log.index]
  
  #Calculate average ATR for position threshold.
  
  if ventana == 10:
    ATR_medio = features["ATR_Normalizado_10"].mean().copy()
  else:
    ATR_medio = features["ATR_Normalizado_20"].mean().copy()
  
  #Threshold for target column to determine financial position.
  
  umbral_adaptativo = 0.15 * ATR_medio

  #Establish conditions to decide financial position.
  
  condiciones = [
      (rend_log > auxiliar_MA5 + umbral_adaptativo),
      (rend_log < auxiliar_MA5 - umbral_adaptativo)
    ]

  features["Objetivo"] = np.select(condiciones, [1,-1], default = 0)

  features["Objetivo_Futuro"] = features["Objetivo"].shift(-1)

  return features

###########################
#5. Train-test Split Data.#
###########################

#Split features table' data on 80% training data and 20% test data. 

def Datos_Entrenamiento_Prueba(features):

  datos_entrenamiento = features.copy()
  datos_entrenamiento.dropna(inplace = True)
  datos_entrenamiento.drop(columns = ["Objetivo"], inplace = True)

  division = int(0.8*len(datos_entrenamiento))
  datosentrenamiento_train = datos_entrenamiento.iloc[:division]
  datosentrenamiento_test = datos_entrenamiento.iloc[division:]

  return datosentrenamiento_train, datosentrenamiento_test

##################
#6. Data Scaling.#
##################

# Standardize feature sets using training statistics to prevent data leakage into the test partition.

def Escalado_De_Datos(datosentrenamiento_train, datosentrenamiento_test):
  escalador_previo = StandardScaler()
  escalador_previo.fit(datosentrenamiento_train.drop(columns = ["Fecha", "Ticker", "Objetivo_Futuro"]))
  datos_entrenamiento_escalado = escalador_previo.transform(datosentrenamiento_train.drop(columns = ["Fecha", "Ticker", "Objetivo_Futuro"]))
  datos_prueba_escalado = escalador_previo.transform(datosentrenamiento_test.drop(columns = ["Fecha", "Ticker", "Objetivo_Futuro"]))

  return datos_entrenamiento_escalado, datos_prueba_escalado

# Format scaled arrays into DataFrames and generate 3D sequence tensors for recurrent neural network processing.

def Creacion_De_Ventanas(datos_entrenamiento_escalado, datos_prueba_escalado, datosentrenamiento_train, datosentrenamiento_test, Crear_Ventanas, ventana):
  if ventana == 10:
    columnas = ["Rendimientos_Logaritmicos", "Media_Movil10Dias", "Volatilidad_Movil10Dias", "Volumen_Normalizado_10", "ATR_Normalizado10Dias", "RSI_Normalizado10Dias", "ADX_Normalizado10Dias"]
  else:
    columnas = ["Rendimientos_Logaritmicos", "Media_Movil20Dias", "Volatilidad_Movil20Dias", "Volumen_Normalizado_20", "ATR_Normalizado20Dias", "RSI_Normalizado20Dias", "ADX_Normalizado20Dias"]
  datos_train = pd.DataFrame(datos_entrenamiento_escalado, columns = columnas, index = datosentrenamiento_train.index)
  datos_test = pd.DataFrame(datos_prueba_escalado, columns = columnas, index = datosentrenamiento_test.index)

  X_entrenamiento, Y_entrenamiento = Crear_Ventanas(datos_train, datosentrenamiento_train["Objetivo_Futuro"], ventana)
  X_prueba, Y_prueba = Crear_Ventanas(datos_test, datosentrenamiento_test["Objetivo_Futuro"], ventana)

  return X_entrenamiento, Y_entrenamiento, X_prueba, Y_prueba

##############################################
# 7. Model Construction & Training Execution.#
##############################################

# Assemble the GRU deep network architecture, handle class weighting, and reindex labels for categorical crossentropy.

def Construccion_Del_Modelo(X_entrenamiento, Y_entrenamiento, Y_prueba, ventana):
  pesos = compute_class_weight(class_weight = 'balanced', classes = np.unique(Y_entrenamiento), y = Y_entrenamiento)
  pesos = dict(enumerate(pesos))

  Modelo = Sequential([
    Input(shape = (X_entrenamiento.shape[1], X_entrenamiento.shape[2])),
    GRU(64, activation = 'tanh', return_sequences = True),
    Dropout(0.2),
    GRU(32, activation = 'tanh', return_sequences = False),
    Dropout(0.2),
    Dense(16, activation = 'relu'),
    Dropout(0.2),
    Dense(3, activation = 'softmax')
  ])

  Modelo.compile(optimizer = Adam(learning_rate= 0.0001), loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
  
  #Fit model output to softmax function allowed values (0,1,2)
  
  Y_entrenamiento_preparado = Y_entrenamiento + 1
  Y_prueba_preparado = Y_prueba + 1

  return Modelo, pesos, Y_entrenamiento_preparado, Y_prueba_preparado

# Train model using EarlyStopping to avoid overfitting, returning trading position predictions and target probabilities.

def Ejecucion_Del_Modelo(Modelo, X_entrenamiento, Y_entrenamiento, X_prueba, Y_prueba, pesos, Y_entrenamiento_preparado, Y_prueba_preparado):
  tf.keras.backend.clear_session()

  detencion = EarlyStopping(monitor = 'val_loss', patience = 15, verbose = 0, restore_best_weights = True)

  historial_modelo = Modelo.fit(
    X_entrenamiento,
    Y_entrenamiento_preparado,
    validation_data = (X_prueba, Y_prueba_preparado),
    class_weight = pesos,
    epochs = 100,
    batch_size = 32,
    callbacks = [detencion],
    verbose = 0)

  prediccion_resultado = Modelo.predict(X_prueba, verbose = 0)

  predicciones_mapeadas = np.argmax(prediccion_resultado, axis = 1)

  Y_predicciones_trading = predicciones_mapeadas - 1
  Y_prueba_trading = Y_prueba_preparado - 1

  return historial_modelo, Y_predicciones_trading, Y_prueba_trading, prediccion_resultado

##################################################
# #8. Financial Metrics & Data Alignment         #
##################################################

# Align historical log returns with GRU neural network directional probabilities, 
# dynamic risk-free rate fetching (^IRX), and expected matrix calculations.

def Obtencion_De_Metricas_Pymoo(probabilidades_GRU, datos_rendimientos):

  factor_de_probabilidad_GRU = dict(zip(tickers,probabilidades_GRU))
  rendimientos_log = dict(zip(tickers, datos_rendimientos))

  datos_con_probabilidades = []

  for ticker, df_prueba in rendimientos_log.items():

    prob = factor_de_probabilidad_GRU[ticker]
    n_predicciones = len(prob)
    rend_temp = df_prueba[["Fecha","Rendimientos_Logaritmicos"]].iloc[-n_predicciones:].copy()
    rend_temp["Probabilidad GRU"] = prob
    datos_con_probabilidades.append((ticker,rend_temp))

  rendimientos_alineados = {}
  probabilidades_alineadas = {}

  for ticker, df in datos_con_probabilidades:
    df_temp = df.copy()

    if "Fecha" not in df_temp.columns:
      df_temp.reset_index()

    df_temp.loc["Fecha"] = pd.to_datetime(df_temp["Fecha"])
    df_temp.set_index('Fecha', inplace=True)

    rendimientos_alineados[ticker] = df_temp["Rendimientos_Logaritmicos"]
    probabilidades_alineadas[ticker] = df_temp["Probabilidad GRU"]

  rendimientos_reales_alineados = pd.concat(rendimientos_alineados, axis=1, join= 'inner').dropna()
  probabilidades_GRU_alineadas = pd.concat(probabilidades_alineadas, axis=1, join= 'inner').dropna()
  
  # Probability-weighted return signals
  
  rendimientos_esperados = rendimientos_reales_alineados * probabilidades_GRU_alineadas

  ventana_efectiva = 30

  rendimiento_promedio_diario = rendimientos_esperados.ewm(span = ventana_efectiva, adjust = True).mean().iloc[-1]

  rendimiento_promedio_anual = rendimiento_promedio_diario * 252
  
  # Annualized covariance matrix computation
  
  matriz_de_covarianza_anual = rendimientos_reales_alineados.cov() * 252

  Hoy = pd.Timestamp.today()

  Inicio = pd.Timedelta(days = len(rendimientos_reales_alineados))
  
  # Dynamic risk-free rate fetching via Yahoo Finance (^IRX 13 Week Treasure Bill)
  
  Tasa_Libre_De_Riesgo = yf.download(tickers = "^IRX", start = Hoy - Inicio, end = Hoy, auto_adjust= True)['Close'].copy()

  ultima_fecha_rendimientos = rendimientos_reales_alineados.index[-1]

  rf_anual = Tasa_Libre_De_Riesgo.loc[ultima_fecha_rendimientos]

  rf_anual = rf_anual / 100

  return rendimiento_promedio_anual, matriz_de_covarianza_anual, rf_anual, rendimientos_reales_alineados, probabilidades_GRU_alineadas

##################################################
# #9. NSGA-II Multi-Objective Optimization       #
##################################################

# Evolutionary optimization via pymoo targeting Sharpe Ratio maximization and CVaR minimization 
# subject to maximum drawdown limits and asset weight constraints.

def Optimizacion_Pymoo(rendimiento_promedio_anual, matriz_de_covarianza_anual, rendimientos_reales_alineados, tickers, rf_anual):

  from pymoo.algorithms.moo.nsga2 import NSGA2
  from pymoo.optimize import minimize
  from pymoo.core.problem import ElementwiseProblem
  from pymoo.visualization.scatter import Scatter

  class Multi_Objective_Portfolio(ElementwiseProblem):

    def __init__(self, expected_returns, cov_matrix, historical_returns, n_assets, rf, min_weight_limit = 0.00, max_weight_limit = 0.35, max_drawdown_limit = 0.15):
      self.expected_returns = expected_returns
      self.cov_matrix = cov_matrix
      self.historical_returns = historical_returns
      self.n_assets = n_assets
      self.rf = rf
      self.min_weight_limit = min_weight_limit
      self.max_weight_limit = max_weight_limit
      self.max_drawdown_limit = max_drawdown_limit
      super().__init__(n_var = n_assets, n_obj = 2, n_constr = 2, xl = np.full(n_assets, min_weight_limit), xu = np.full(n_assets,max_weight_limit))

    def _evaluate(self, x, out, *args, **kwargs):
      
      # Objective 1: Maximize Sharpe Ratio (F1)
      
      portfolio_volatility = np.sqrt(x @ self.cov_matrix @ x)
      portfolio_return = np.sum(self.expected_returns * x)

      if portfolio_volatility < 1e-8:
              sharpe_ratio = 0.0
      else:
            sharpe_ratio = (portfolio_return - self.rf) / portfolio_volatility

      f1 = sharpe_ratio
      
      # Objective 2: Minimize Conditional Value at Risk at 95% Confidence (F2)
      
      portfolio_returns = np.dot(self.historical_returns, x)
      portfolio_VaR = np.percentile(portfolio_returns, 5)
      portfolio_losses = portfolio_returns[portfolio_returns <= portfolio_VaR]
      if len(portfolio_losses) == 0:
            portfolio_CVaR = portfolio_VaR
      else:
            portfolio_CVaR = -np.mean(portfolio_losses)
      f2 = portfolio_CVaR
      
      # Inequality Constraints: Max Drawdown Limit & Fully Invested Weight Sum = 1 With Tolerance 0.01 And Weight Threshold (0.0, 0.35)
      
      cumulative_returns = np.cumsum(portfolio_returns)
      peak = np.maximum.accumulate(cumulative_returns)
      drawdowns = cumulative_returns - peak
      max_drawdown = np.abs(np.min(drawdowns))

      g1 = max_drawdown - self.max_drawdown_limit

      g2 = np.abs(np.sum(x) - 1) - 0.01

      out["F"] = [-f1, f2]
      out["G"] = [g1,g2]

  problem = Multi_Objective_Portfolio(expected_returns = rendimiento_promedio_anual.values,
                                    cov_matrix = matriz_de_covarianza_anual.values,
                                    historical_returns = rendimientos_reales_alineados.values,
                                    n_assets = len(tickers),
                                    rf = rf_anual.iloc[0])

  algorithm = NSGA2(pop_size = 200)

  results = minimize(problem,
                   algorithm,
                   ('n_gen',200),
                   seed = 1,
                   verbose = True)

  X = results.X
  F = results.F.copy()
  
  # Invert sign of Sharpe Ratio back to positive for visualization
  
  F = F * [-1,1]

  plt.scatter(F[:, 1], F[:, 0], facecolor="none", edgecolors="green", alpha=0.5, label="Optimal Portfolios")
  plt.title("Portfolio's Efficiency Curve")
  plt.legend(loc=7)
  plt.xlabel("Conditional Value At Risk")
  plt.ylabel("Sharpe Ratio")
  plt.show()

  return X, F

##################################################
# #3. Database Schema & Profile Profiling       #
##################################################

# Identify key investment profiles (Max Sharpe, Min CVaR, Knee Point), unpivot weights, 
# and fetch latest close prices for relational database integration.

def Obtencion_De_Tablas_Base_De_Datos(X,F):

  with sq3.connect("Activos Portafolio.db") as conexion:

    Precios_de_Activo = pd.read_sql('''
      WITH Precios_Ordenados AS (
        SELECT Fecha,
               Ticker,
               Precio_de_Cierre,
               ROW_NUMBER() OVER (PARTITION BY Ticker ORDER BY Fecha DESC) AS Fila
        FROM Activos_Financieros_Actualizados
      )
      SELECT Fecha,
             Ticker,
             Precio_de_Cierre
      FROM Precios_Ordenados
      WHERE Fila = 1
    ''', conexion)

  pesos_optimos = pd.DataFrame(X,columns = tickers)

  resultados_portafolio = pd.DataFrame(F, columns = ["Sharpe Ratio", "Conditional Value At Risk"])

  portafolios_optimos = pd.merge(
    pesos_optimos,
    resultados_portafolio,
    left_index = True,
    right_index = True,
    how = "left"
  )

  portafolios_optimos_pareto = portafolios_optimos.copy()

  portafolios_optimos_pareto['Portfolio_ID'] = portafolios_optimos_pareto.index
  
  # Identify strategic allocation profiles
  
  id_max_sharpe = portafolios_optimos_pareto['Sharpe Ratio'].idxmax()
  id_min_cvar = portafolios_optimos_pareto['Conditional Value At Risk'].idxmin()

  sharpe = portafolios_optimos_pareto['Sharpe Ratio'].values
  cvar = portafolios_optimos_pareto['Conditional Value At Risk'].values

  sharpe_norm = (sharpe - sharpe.min()) / (sharpe.max() - sharpe.min())
  cvar_norm = (cvar - cvar.min()) / (cvar.max() - cvar.min())
  
  # Calculate optimal Trade-Off (Knee Point) via Euclidean distance to the ideal target
  distancia_punto_optimo = np.sqrt((1 - sharpe_norm)**2 + (cvar_norm)**2)

  idx_knee = np.argmin(distancia_punto_optimo)

  def perfil_de_riesgo(row):

    if row.name == id_max_sharpe:
      return "Max Sharpe"
    elif row.name == id_min_cvar:
      return "Min CVar"
    elif row.name == idx_knee:
      return "Knee"
    else:
      return "Pareto Frontier"

  portafolios_optimos_pareto["Perfil_De_Riesgo"] = portafolios_optimos_pareto.apply(perfil_de_riesgo, axis = 1)

# Unpivot wide weights to long format for Power BI star-schema normalization

  pesos_ajustados = pd.melt(
    portafolios_optimos_pareto,
    id_vars = ["Portfolio_ID"],
    value_vars = tickers,
    var_name = "Ticker",
    value_name = "Peso Teorico"
  )

  return portafolios_optimos_pareto, pesos_ajustados, Precios_de_Activo

#####################################################
# MAIN EXECUTION PIPELINE & RESULTS CONSOLIDATION.#
#####################################################

# Extract unique asset tickers from the database

with sq3.connect("Activos Portafolio.db") as conexion:
  tickers = pd.read_sql("SELECT DISTINCT Ticker FROM Datos_Modelo", conexion).values
  tickers = tickers.flatten()

datos_rendimientos = []
probabilidades_GRU = []

# Iterative execution loop across all assets

for ticker in tickers:
  datos = Obtencion_De_Datos(ticker)
  features, ventana, volatilidad_promedio = Obtencion_De_Features(datos)
  features_obj = Obtencion_De_Objetivo(datos,features,ventana)
  datos_entrenamiento, datos_prueba = Datos_Entrenamiento_Prueba(features_obj)
  datos_entrenamiento_escalado, datos_prueba_escalado = Escalado_De_Datos(datos_entrenamiento, datos_prueba)
  X_entrenamiento, Y_entrenamiento, X_prueba, Y_prueba = Creacion_De_Ventanas(datos_entrenamiento_escalado, datos_prueba_escalado, datos_entrenamiento, datos_prueba, Crear_Ventanas, ventana)
  Modelo, pesos, Y_entrenamiento_preparado, Y_prueba_preparado = Construccion_Del_Modelo(X_entrenamiento, Y_entrenamiento, Y_prueba, ventana)
  historial_modelo, Y_predicciones_trading, Y_prueba_trading, prediccion_resultado = Ejecucion_Del_Modelo(Modelo, X_entrenamiento, Y_entrenamiento_preparado, X_prueba, Y_prueba, pesos, Y_entrenamiento_preparado, Y_prueba_preparado)
#Append historical returns and GRU probabilities to their respective lists
  datos_rendimientos.append(datos_prueba[["Fecha", "Rendimientos_Logaritmicos"]])
  probabilidades_GRU.append(prediccion_resultado[:,2] - prediccion_resultado[:,0])

#Obtain expected returns, covariance matrix and risk-free rate for Pymoo optimization
rendimiento_promedio_anual, matriz_de_covarianza_anual, rf_anual, rendimientos_reales_alineados, probabilidades_GRU_alineadas = Obtencion_De_Metricas_Pymoo(probabilidades_GRU, datos_rendimientos)

#Install pymoo library
pip install pymoo

#Obtain thereotetical weights of the 200 optimal portfolio candidates with their respective Sharpe Ratio and  CVaR
X,F = Optimizacion_Pymoo(rendimiento_promedio_anual, matriz_de_covarianza_anual, rendimientos_reales_alineados, tickers, rf_anual)

#Build relational database schema to use in Power BI, containing optimal portfolios weights and metrics, thereotetical weights by asset and close asset prices
portafolios_optimos_pareto, pesos_ajustados, Precios_de_Activo = Obtencion_De_Tablas_Base_De_Datos(X,F)

# Store portfolio results output into SQLite database
conexion = create_engine('sqlite:///Optimal_Pymoo_Portfolios.db')
portafolios_optimos_pareto.to_sql('Portafolios_Optimos_Con_Perfil', con = conexion, if_exists = 'replace')
pesos_ajustados.to_sql('Pesos_Optimos_Por_Ticker', con = conexion, if_exists = 'replace')
Precios_de_Activo.to_sql('Precios_De_Activo', con = conexion, if_exists = 'replace')

# Optional: Download database file if running on Google Colab
try:
    from google.colab import files
    files.download('Optimal_Pymoo_Portfolios.db')
except ImportError:
    pass  # Executing locally; database is saved in the working directory
