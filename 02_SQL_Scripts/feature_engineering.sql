/*
===============================================================================
Script: Financial Feature Engineering & Data Pipeline
Description: Normalizes multi-currency prices (USD to MXN) and constructs 
             technical indicators (ATR, RSI, ADX, Rolling Volatility/Volume) 
             for deep learning input features.
Engine: SQLite
===============================================================================
*/

----------------------------------------------------------------------------------------------------------
-- SECTION 1: Currency Normalization (USD to MXN)
----------------------------------------------------------------------------------------------------------

-- Join raw financial data with historical USD-MXN exchange rates
CREATE TABLE Cambio AS 
SELECT * FROM [Activos_Financieros] a LEFT JOIN [Tipo_de_Cambio_Dolar_a_Peso] t
WHERE a.Fecha = t.Fecha_Cambio
;

-- Impute missing exchange rate values using the previous day's rate (Forward Fill)
WITH Datos_Activos AS(
SELECT Fecha, 
	Ticker, 
	Precio_de_Cierre_Ajustado, 
	Precio_Maximo,
	Precio_Minimo,
	Precio_de_Cierre, 
	Precio_de_Apertura, 
	Volumen, 
	Precio_de_Cierre_Cambio,
	LAG(Precio_de_Cierre_Cambio,1) 
	OVER (PARTITION BY Ticker ORDER BY Fecha ASC) 
	AS Anterior
FROM Cambio
)
UPDATE Cambio
SET Precio_de_Cierre_Cambio = Datos_Activos.Anterior
FROM Datos_Activos
WHERE Cambio.Fecha = Datos_Activos.Fecha 
AND Cambio.Precio_de_Cierre_Cambio IS NULL 
AND Datos_Activos.Anterior IS NOT NULL
;

-- Convert USD asset prices to MXN; Mexican assets (.MX) remain unchanged
CREATE TABLE Activos_Financieros_Actualizados AS SELECT Fecha,
       Ticker,
	   CASE 
		WHEN Ticker LIKE '%.MX' THEN Precio_de_Cierre_Ajustado
		ELSE Precio_de_Cierre_Ajustado * Precio_de_Cierre_Cambio
	   END AS Precio_de_Cierre_Ajustado,
	   CASE 
		WHEN Ticker LIKE '%.MX' THEN Precio_Maximo
		ELSE Precio_Maximo * Precio_de_Cierre_Cambio
	   END AS Precio_Maximo,
           CASE 
		WHEN Ticker LIKE '%.MX' THEN Precio_Minimo
		ELSE Precio_Minimo * Precio_de_Cierre_Cambio
	   END AS Precio_Minimo,
	   CASE 
		WHEN Ticker LIKE '%.MX' THEN Precio_de_Cierre
		ELSE Precio_de_Cierre * Precio_de_Cierre_Cambio
	   END AS Precio_de_Cierre,
	   CASE 
		WHEN Ticker LIKE '%.MX' THEN Precio_de_Apertura
		ELSE Precio_de_Apertura * Precio_de_Cierre_Cambio
	   END AS Precio_de_Apertura,
	   Volumen
	   FROM Cambio
;	 

----------------------------------------------------------------------------------------------------------
-- SECTION 2: Logarithmic Returns And Rolling Statistical Indicators
----------------------------------------------------------------------------------------------------------

-- Calculate logarithmic returns based on adjusted close price
CREATE VIEW Tabla_Auxiliar AS
SELECT Fecha,
	   Ticker,
	   Precio_de_Cierre,
	   ln(Precio_de_Cierre / LAG(Precio_de_Cierre,1) OVER (
	   PARTITION BY Ticker
	   ORDER BY FECHA ASC)) AS Rendimientos_Logaritmicos,
           Volumen
FROM Activos_Financieros_Actualizados
;

-- Compute 10-day and 20-day Moving Averages, Rolling Volatility, and Standardized Volume
CREATE TABLE Datos_Preeliminares AS 
WITH Calculos AS (
	SELECT Fecha,
	       Ticker,
		   Rendimientos_Logaritmicos,
		   Volumen,
		   AVG(Rendimientos_Logaritmicos) OVER (
		   PARTITION BY Ticker
		   ORDER BY Fecha ASC
		   ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) AS Promedio_20,
           -- 20-day & 10-day Log-Return Moving Averages
           AVG(Rendimientos_Logaritmicos) OVER (
		   PARTITION BY Ticker
		   ORDER BY Fecha ASC
		   ROWS BETWEEN 9 PRECEDING AND CURRENT ROW) AS Promedio_10,
           -- Second Moments for Rolling Volatility Calculation
		   AVG(Rendimientos_Logaritmicos*Rendimientos_Logaritmicos) OVER (
		   PARTITION BY Ticker
		   ORDER BY Fecha ASC
		   ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) AS Promedio2_20,
           AVG(Rendimientos_Logaritmicos*Rendimientos_Logaritmicos) OVER (
		   PARTITION BY Ticker
		   ORDER BY Fecha ASC
		   ROWS BETWEEN 9 PRECEDING AND CURRENT ROW) AS Promedio2_10,
		   AVG(Volumen) OVER (
		   PARTITION BY Ticker
                   ORDER BY Fecha ASC
                   ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
	       ) AS Promedio_Volumen_20,
           -- Volume Moving Averages & Second Moments
                   AVG(Volumen) OVER (
		   PARTITION BY Ticker
                   ORDER BY Fecha ASC
                   ROWS BETWEEN 9 PRECEDING AND CURRENT ROW
	       ) AS Promedio_Volumen_10,
		   AVG(Volumen*Volumen) OVER (
		   PARTITION BY Ticker
                   ORDER BY Fecha ASC
                   ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
	       ) AS Promedio_Volumen2_20,
                   AVG(Volumen*Volumen) OVER (
		   PARTITION BY Ticker
                   ORDER BY Fecha ASC
                   ROWS BETWEEN 9 PRECEDING AND CURRENT ROW
	       ) AS Promedio_Volumen2_10
		   FROM Tabla_Auxiliar
	)
SELECT Fecha,
	   Ticker,
	   Rendimientos_Logaritmicos,
	   Promedio_20 AS Media_Movil20Dias,
           Promedio_10 AS Media_Movil10Dias,
	   SQRT(Promedio2_20 - POW(Promedio_20,2)) AS Volatilidad_Movil20Dias,
           SQRT(Promedio2_10 - POW(Promedio_10,2)) AS Volatilidad_Movil10Dias,
	   (Volumen - Promedio_Volumen_20)/SQRT(Promedio_Volumen2_20 - POW(Promedio_Volumen_20,2)) AS Volumen_Normalizado_20,
           (Volumen - Promedio_Volumen_10)/SQRT(Promedio_Volumen2_10 - POW(Promedio_Volumen_10,2)) AS Volumen_Normalizado_10
FROM Calculos	   
;

----------------------------------------------------------------------------------------------------------
-- SECTION 3: Technical Indicators Computation
----------------------------------------------------------------------------------------------------------

-- 1. Average True Range (ATR) Normalization.

CREATE TABLE ATR_Norm AS

WITH Calculo_ATR AS (
	SELECT Fecha,
	       Ticker,
		   Precio_de_Cierre,
		   Precio_Maximo - Precio_Minimo AS ATR_1,
		   ABS(Precio_Maximo - LAG (Precio_de_Cierre,1)
		   OVER (PARTITION BY Ticker ORDER BY Fecha ASC)) AS ATR_2,
		   ABS(Precio_Minimo - LAG (Precio_de_Cierre,1)
		   OVER (PARTITION BY Ticker ORDER BY Fecha ASC)) AS ATR_3
    FROM Activos_Financieros_Actualizados
),
Seleccion_Max AS (
    SELECT Fecha,
	       Ticker,
		   Precio_de_Cierre,
		   CASE
		   WHEN ATR_1 >= ATR_2 AND ATR_1 >= ATR_3 THEN ATR_1
		   WHEN ATR_2 >= ATR_1 AND ATR_2 >= ATR_3 THEN ATR_2
		   ELSE ATR_3
		   END AS TR
    FROM Calculo_ATR
),
ATR_Media AS (
	SELECT Fecha,
	       Ticker,
		   Precio_de_Cierre,
		   AVG(TR) OVER (
		   PARTITION BY Ticker
		   ORDER BY Fecha ASC
		   ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
		   ) AS ATR_20,
                   AVG(TR) OVER (
		   PARTITION BY Ticker
		   ORDER BY Fecha ASC
		   ROWS BETWEEN 9 PRECEDING AND CURRENT ROW
		   ) AS ATR_10
 
	FROM Seleccion_Max
)

SELECT Fecha,
       Ticker,
	   Precio_de_Cierre,
	   ATR_20,
           ATR_10,
	   ATR_20 / Precio_de_Cierre AS ATR_Normalizado_20,
           ATR_10 / Precio_de_Cierre AS ATR_Normalizado_10 
FROM ATR_Media 
;

-- 2. Relative Strength Index (RSI) Normalization.

CREATE TABLE RSI_Norm AS

WITH Cambios_de_Precio AS (
	SELECT Fecha,
	       Ticker, 
		   Precio_de_Cierre,
		   Precio_de_Cierre - LAG(Precio_de_Cierre,1)
		   OVER (PARTITION BY Ticker ORDER BY Fecha ASC) AS Diferencia_de_Precios
    FROM ATR_Norm
),
Seleccion_Diferencias AS(
	SELECT Fecha,
	       Ticker,
		   Precio_de_Cierre,
		   CASE WHEN Diferencia_de_Precios > 0 THEN ABS(Diferencia_de_Precios)
		   ELSE 0 END AS Ganancia,
		   CASE WHEN Diferencia_de_Precios < 0 THEN ABS(Diferencia_de_Precios)
		   ELSE 0 END AS Perdida
    FROM Cambios_de_Precio
),
Promedios_RSI AS (
	SELECT Fecha,
		   Ticker,
		   Precio_de_Cierre,
		   AVG(Ganancia) OVER (
		   PARTITION BY Ticker
		   ORDER BY Fecha ASC
		   ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
		   ) AS Promedio_Ganancias_20,
                   AVG(Ganancia) OVER (
		   PARTITION BY Ticker
		   ORDER BY Fecha ASC
		   ROWS BETWEEN 9 PRECEDING AND CURRENT ROW
		   ) AS Promedio_Ganancias_10,
		   AVG(Perdida) OVER (
		   PARTITION BY Ticker
		   ORDER BY Fecha ASC
		   ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
		   ) AS Promedio_Perdidas_20,
                   AVG(Perdida) OVER (
		   PARTITION BY Ticker
		   ORDER BY Fecha ASC
		   ROWS BETWEEN 9 PRECEDING AND CURRENT ROW
		   ) AS Promedio_Perdidas_10
	 FROM Seleccion_Diferencias
)

SELECT Fecha,
       Ticker,
	   Precio_de_Cierre,
	   CASE
	   WHEN Promedio_Perdidas_20 = 0 THEN 1.0
	   ELSE (100 - (100 / (1 + (Promedio_Ganancias_20 / Promedio_Perdidas_20)))) / 100.0
	   END AS RSI_Normalizado_20,
           CASE
	   WHEN Promedio_Perdidas_10 = 0 THEN 1.0
	   ELSE (100 - (100 / (1 + (Promedio_Ganancias_10 / Promedio_Perdidas_10)))) / 100.0
	   END AS RSI_Normalizado_10
FROM Promedios_RSI

;

-- 3. Average Directional Index (ADX) Normalization.

CREATE TABLE ADX_Norm AS

WITH Base_Precios AS (
	SELECT A.Fecha,
	       A.Ticker,
		   A.Precio_de_Cierre,
		   A.ATR_20,
                   A.ATR_10,
		   F.Precio_Maximo,
		   F.Precio_Minimo,
		   LAG(F.Precio_Maximo,1) OVER (PARTITION BY A.Ticker ORDER BY A.Fecha ASC)
		   AS Maximo_Anterior,
		   LAG(F.Precio_Minimo,1) OVER (PARTITION BY A.Ticker ORDER BY A.Fecha ASC)
		   AS Minimo_Anterior
	 FROM [ATR_Norm] A INNER JOIN [Activos_Financieros_Actualizados] F 
	 ON A.Fecha = F.Fecha AND A.Ticker = F.Ticker
),
Direccion AS (
	SELECT Fecha,
	       Ticker,
		   Precio_de_Cierre,
		   ATR_20,
                   ATR_10,
		   CASE
		   WHEN (Precio_Maximo - Maximo_Anterior) > (Minimo_Anterior - Precio_Minimo)
		   AND (Precio_Maximo - Maximo_Anterior) > 0
		   THEN (Precio_Maximo - Maximo_Anterior)
		   ELSE 0 END AS DM_Positivo_Crudo,
		   CASE
		   WHEN (Minimo_Anterior - Precio_Minimo) > (Precio_Maximo - Maximo_Anterior)
		   AND (Minimo_Anterior - Precio_Minimo) > 0
		   THEN (Minimo_Anterior - Precio_Minimo)
		   ELSE 0 END AS DM_Negativo_Crudo
   FROM Base_Precios
) ,
Promedios_DM AS (
	SELECT Fecha,
	       Ticker, 
		   Precio_de_Cierre,
		   ATR_20,
                   ATR_10,
		   AVG(DM_Positivo_Crudo) OVER (
		   PARTITION BY Ticker
		   ORDER BY Fecha ASC
		   ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) AS DM_Positivo_Suavizado_20,
                   AVG(DM_Positivo_Crudo) OVER (
		   PARTITION BY Ticker
		   ORDER BY Fecha ASC
		   ROWS BETWEEN 9 PRECEDING AND CURRENT ROW) AS DM_Positivo_Suavizado_10,
		   AVG(DM_Negativo_Crudo) OVER (
		   PARTITION BY Ticker
		   ORDER BY Fecha ASC
		   ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) AS DM_Negativo_Suavizado_20,
                   AVG(DM_Negativo_Crudo) OVER (
		   PARTITION BY Ticker
		   ORDER BY Fecha ASC
		   ROWS BETWEEN 9 PRECEDING AND CURRENT ROW) AS DM_Negativo_Suavizado_10
	  FROM Direccion	   
) ,
Indicadores AS (
	SELECT Fecha,
	       Ticker,
		   Precio_de_Cierre,
		   CASE
		   WHEN ATR_20 = 0 THEN 0 
		   ELSE (DM_Positivo_Suavizado_20 / ATR_20) * 100
		   END AS DI_Positivo_20,
                   CASE
		   WHEN ATR_10 = 0 THEN 0 
		   ELSE (DM_Positivo_Suavizado_10 / ATR_10) * 100
		   END AS DI_Positivo_10,
		   CASE
		   WHEN ATR_20 = 0 THEN 0 
		   ELSE (DM_Negativo_Suavizado_20 / ATR_20) * 100
		   END AS DI_Negativo_20,
                   CASE
		   WHEN ATR_10 = 0 THEN 0 
		   ELSE (DM_Negativo_Suavizado_10 / ATR_10) * 100
		   END AS DI_Negativo_10
	 FROM Promedios_DM
),
Indice_Direccional AS (
    SELECT Fecha,
	       Ticker,
		   Precio_de_Cierre,
		   CASE 
		   WHEN DI_Positivo_20 + DI_Negativo_20 = 0 THEN 0
		   ELSE ABS((DI_Positivo_20 - DI_Negativo_20) / (DI_Positivo_20 + DI_Negativo_20)) * 100
		   END AS DX_20,
                   CASE 
		   WHEN DI_Positivo_10 + DI_Negativo_10 = 0 THEN 0
		   ELSE ABS((DI_Positivo_10 - DI_Negativo_10) / (DI_Positivo_10 + DI_Negativo_10)) * 100
		   END AS DX_10
    FROM Indicadores
) ,

Calculo_ADX AS (
	SELECT Fecha,
	       Ticker,
		   Precio_de_Cierre,
		   AVG(DX_20) OVER (
		   PARTITION BY Ticker
		   ORDER BY Fecha ASC
		   ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
		   ) AS ADX_20,
                   AVG(DX_10) OVER (
		   PARTITION BY Ticker
		   ORDER BY Fecha ASC
		   ROWS BETWEEN 9 PRECEDING AND CURRENT ROW
		   ) AS ADX_10  
	FROM Indice_Direccional
)

SELECT Fecha,
       Ticker,
	   Precio_de_Cierre,
	   (ADX_20 / 100) AS ADX_Normalizado_20,
           (ADX_10 / 100) AS ADX_Normalizado_10
FROM Calculo_ADX

;

-------------------------------------------------------------------------------
-- SECTION 4: Master Table Consolidation for ML Training
-------------------------------------------------------------------------------

-- Consolidate all technical features into a single dataset for model ingestion
CREATE TABLE Datos_Modelo AS
	SELECT D.Fecha,
               D.Ticker,
               D.Rendimientos_Logaritmicos,
	       D.Media_Movil20Dias,
               D.Media_Movil10Dias,
	       D.Volatilidad_Movil20Dias,
               D.Volatilidad_Movil10Dias,
	       D.Volumen_Normalizado_20,
               D.Volumen_Normalizado_10,
               A.ATR_Normalizado_20,
               A.ATR_Normalizado_10,
               R.RSI_Normalizado_20,
               R.RSI_Normalizado_10,
               AD.ADX_Normalizado_20,
               AD.ADX_Normalizado_10
       FROM [Datos_Preeliminares] D INNER JOIN [ATR_Norm] A
       ON D.Fecha = A.Fecha AND D.Ticker = A.Ticker
       INNER JOIN [RSI_Norm] R
       ON D.Fecha = R.Fecha AND D.Ticker = R.Ticker
       INNER JOIN [ADX_Norm] AD
       ON D.Fecha = AD.Fecha AND D.Ticker = AD.Ticker

;  
                                   





# Tabla Dim_Activos



