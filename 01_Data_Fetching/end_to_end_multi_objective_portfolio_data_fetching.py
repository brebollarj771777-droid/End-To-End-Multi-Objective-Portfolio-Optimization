"""
Multi-Objective Portfolio Optimization Data Fetching Module
Extracts historical market data for portfolio assets and saves it to a SQLite database.
"""

# Third Party Library Imports
import yfinance as yf
import pandas as pd
from sqlalchemy import create_engine

##################################
#Data Fetching from Yahoo Finance#
##################################

#Establish the start and end dates for the fetching, in a 4-year span to skip COVID-19 data.

Hoy = pd.Timestamp.today()
Cuatro_Anios = pd.Timedelta(days = 365*4)

#Define portfolio constituents and sort them alphabetically.

tickers = ["SPY","QQQ","BIMBOA.MX","WALMEX.MX","GLD","TLT","BTC-USD"]
tickers.sort()

#Fetch Yahoo Finance historical data, including the adjusted close price to avoid dividends bias.

datos = []
for ticker in tickers:
  datos_ticker = yf.download(ticker, start = Hoy - Cuatro_Anios, end = Hoy, auto_adjust = False).copy()
  
  # Flatten MultiIndex columns returned by yfinance to single-level column names.
  
  if isinstance(datos_ticker.columns, pd.MultiIndex):
    datos_ticker.columns = datos_ticker.columns.get_level_values(0)
  
  #Reset DataFrame indexes to remove the index of the column date for later use.
  
  datos_ticker = datos_ticker.reset_index()
  
  #Rename DataFrame columns for SQLite further usage and to fit the developer's native language.
  
  datos_ticker.rename(columns = {
      'Date': 'Fecha',
      'Adj Close': 'Precio_de_Cierre_Ajustado',
      'High': 'Precio_Maximo',
      'Low': 'Precio_Minimo',
      'Close': 'Precio_de_Cierre',
      'Open': 'Precio_de_Apertura',
      'Volume': 'Volumen'
      }, inplace = True)
  
  #Insert 'Ticker' column to act as a composite key component alongside Date. 
  
  datos_ticker.insert(1,'Ticker',ticker)
  datos.append(datos_ticker)

#Concatenate all assets' historical data in a singular table. 

activos_financieros = pd.concat(datos, ignore_index = True)

#Extract exchange rate data from 'USD' to 'MXN' for currency homogeneity.

usd_to_mxn = yf.download('USDMXN=X', start = Hoy - Cuatro_Anios, end = Hoy, auto_adjust = False)['Close'].copy()
usd_to_mxn = usd_to_mxn.reset_index()
usd_to_mxn.rename(columns = {'Date': 'Fecha_Cambio', 'USDMXN=X': 'Precio_de_Cierre_Cambio'}, inplace = True)

#Create SQLite Database through SQLalchemy with portfolio and exchange rate DataFrames as tables.

conexion = create_engine('sqlite:///Activos Portafolio.db')
activos_financieros.to_sql('Activos_Financieros', conexion, if_exists = 'replace', index = False)
usd_to_mxn.to_sql('Tipo_de_Cambio_Dolar_a_Peso', conexion, if_exists = 'replace', index = False)


# Optional: Download database file if running on Google Colab
try:
    from google.colab import files
    files.download('Activos Portafolio.db')
except ImportError:
    pass  # Executing locally; database is saved in the working directory
