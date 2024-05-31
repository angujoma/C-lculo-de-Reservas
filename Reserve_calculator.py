import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import math

from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
class SQLDataExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, query, conn):
        self.query = query
        self.conn = conn

    def fit(self, X=None, y=None):
        return self

    def transform(self, X=None):
        data = pd.read_sql(self.query, self.conn)
        return data
    


class DataCleaner(BaseEstimator, TransformerMixin): 
    def __init__(self, columnas_riesgo=feats_riesgo, columnas_fecha=dates, columna_nula=dates2, columna_espacios=espacios):
        self.columnas_riesgo = columnas_riesgo
        self.columnas_fecha = columnas_fecha
        self.columna_nula = columna_nula
        self.columna_espacios = columna_espacios

    def codificar_segmento_riesgo(self, df):
        df = pd.get_dummies(df, columns=self.columnas_riesgo)
        df.columns = [col.strip() for col in df.columns]
        return df

    def formato_fecha(self, fecha_str):
        try:
            return pd.to_datetime(fecha_str, format="%Y%m%d")
        except ValueError:
            return pd.NaT

    def formatear_fecha(self, df, columna):
        df[columna] = df[columna].apply(self.formato_fecha)
        return df

    def eliminar_filas_nulas(self, df, columna):
        df = df.dropna(subset=[columna])
        return df

    def quitar_espacios_blancos(self, df, columna):
        df[columna] = df[columna].str.strip()
        return df

    def fit(self, X, y=None): 
        return self

    def transform(self, X, y=None):
        X = self.codificar_segmento_riesgo(X)
        for columna in self.columnas_fecha:
            X = self.formatear_fecha(X, columna)
        X = self.eliminar_filas_nulas(X, self.columna_nula)
        X = self.quitar_espacios_blancos(X, self.columna_espacios)
        return X

class ReserveCalculator_CG(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.copy()  

        try:
            X[["ALTO_C", "MEDIO_C", "BAJO_C"]] = X.apply(lambda row: pd.Series(self.calculo_NR(row)), axis=1)
            X['Zi_C'] = X.apply(self.calculo_zi, axis=1)
            X['Pi_C'] = X.apply(self.calculo_pi, axis=1)
            X['EXP_C'] = X.apply(self.calculo_EXP, axis=1)
            X['PLA_REM_N_C'] = X.apply(self.calculo_PLA_REM_N, axis=1)
            X["TASA_INTERES_ANUAL_C"] = X["TASA_INTERES"].apply(self.calculo_TASA_INTERES_ANUAL)
            X['PAGO_C'] = X.apply(self.calculo_PAGO, axis=1)
            X['RESERVA_C'] = X.apply(self.calculo_RESERVA, axis=1)
            X['RESERVA_VC'] = X.apply(self.calculo_RESERVA_VC, axis=1)
            X['RESERVA_IFSR09_C'] = X.apply(self.calculo_RESERVA_IFSR09, axis=1)
            tabla_calif = self.tabla_calificaciones()
            X['CALIFICACION_CARTERA_C'] = X.apply(lambda row: self.asignar_calificacion(row['RESERVA_IFSR09_C'] / row["EXP_C"], tabla_calif), axis=1)
        except Exception as e:
            print(f"Error during transformation: {e}")
            raise

        return X

 
    @staticmethod
    def calculo_pi(row):
        try:
            ATR = row['NUMERO_ATR']
            ETAPA = row['ETAPA']
            Zi = row['Zi_C']
            
            if ATR > 3 or ETAPA == 3:
                resultado = 1
            else:
                resultado = 1 / (1 + math.exp(-Zi))
            return round(resultado, 8)
        except Exception as e:
            print(f"Error in calculo_pi: {e}")
            return np.nan

    @staticmethod
    def calculo_EXP(row):
        try:
            if row['ETAPA'] < 3:
                return row['SALDO_CAPITAL'] + row['SALDO_INTERES']
            else:
                return row['SALDO_CAPITAL']
        except Exception as e:
            print(f"Error in calculo_EXP: {e}")
            return np.nan

    @staticmethod
    def calculo_PLA_REM_N(row):
        try:
            if (row['FECHA_TERMINO'] - row['FECHA_PROCESO']).days <= 0:
                return int(365.25 / 365.25)
            else:
                return int(((row['FECHA_TERMINO'] - row['FECHA_PROCESO']).days / 365.25) * 100000) / 100000
        except Exception as e:
            print(f"Error in calculo_PLA_REM_N: {e}")
            return np.nan


    def calculo_PAGO(self, row):
        try:
            tasa_interes_anual = self.calculo_TASA_INTERES_ANUAL(row['TASA_INTERES'])
            pago_c = np.trunc((row['EXP_C'] * (1 + tasa_interes_anual) *
                              ((1 - ((1 + tasa_interes_anual) ** -1)) / 
                              (1 - ((1 + tasa_interes_anual) ** -row['PLA_REM_N_C'])))) * 100) / 100
            return pago_c
        except Exception as e:
            print(f"Error in calculo_PAGO: {e}")
            return np.nan

    @staticmethod
    def calculo_RESERVA(row):
        try:
            Pi = float(row['Pi_C'])
            SPi = float(row['SPi_C'])
            EXPi = float(row['EXP_C'])
            reserva = Pi * SPi * EXPi
            return reserva
        except Exception as e:
            print(f"Error in calculo_RESERVA: {e}")
            return np.nan


    @staticmethod
    def calculo_RESERVA_IFSR09(row):
        try:
            if row["ETAPA"] == 2:
                return max(row["RESERVA_C"], row["RESERVA_VC"])
            else:
                return row["RESERVA_C"]
        except Exception as e:
            print(f"Error in calculo_RESERVA_IFSR09: {e}")
            return np.nan

    @staticmethod
    
    def asignar_calificacion(valor, tabla_calif):
        for index, rango in enumerate(tabla_calif['PORCENTAJE_RESERVAS']):
            limite_superior = float(rango.split(" - ")[-1].replace(' o más', ''))
            if index == 0:
                limite_inferior = 0
            else:
                limite_inferior = float(tabla_calif.iloc[index - 1, 0].split(" - ")[0])
            if limite_inferior <= valor < limite_superior:
                return tabla_calif.iloc[index, 1]  
        return tabla_calif.iloc[-1, 1]  
    


class ReserveCategoryTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, threshold):
        self.threshold = threshold

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.copy()
        X['RESERVA_CATEGORIA'] = X['RESERVA_IFSR09'].apply(self.categorize)
        return X

    def categorize(self, value):
        if value >= self.threshold:
            return 1  # 'alta' codificada como 1
        else:
            return 0  # 'baja' codificada como 0
  

class ColumnNameUpdater(BaseEstimator, TransformerMixin):
    def __init__(self, new_names):
        self.new_names = new_names
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        return X.rename(columns=self.new_names)
    


class DataFrameStorer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.dataframe = None

    def fit(self, X, y=None):
        self.dataframe = X.copy()
        return self

    def transform(self, X, y=None):
        return X

    def get_dataframe(self):
        return self.dataframe


class ClassificationTreeTrainer(BaseEstimator):
    def __init__(self, independent_variables, dependent_variable, test_size=0.2, random_state=None):
        self.independent_variables = independent_variables
        self.dependent_variable = dependent_variable
        self.test_size = test_size
        self.random_state = random_state
        self.model = DecisionTreeClassifier(max_depth=3)
        self.X_test = None
        self.y_test = None
        self.y_pred = None

    def fit(self, X, y=None):
        X_train, X_test, y_train, y_test = train_test_split(
            X[self.independent_variables], X[self.dependent_variable], 
            test_size=self.test_size, random_state=self.random_state
        )

    def transform(self, X, y=None):
        pass  # No es necesario transformar nada

    def predict(self, X):
        return self.model.predict(X[self.independent_variables])

    def plot_tree(self, **kwargs):
        plt.figure(figsize=(20, 10)) 
       
        
        return plot_tree(self.model, **kwargs)
        

    def print_metrics(self):
        if self.y_test is None or self.y_pred is None:
            raise AttributeError("El modelo no ha sido ajustado. Llama al método fit antes de imprimir las métricas.")
        
        print("Accuracy:", accuracy_score(self.y_test, self.y_pred))
        print("Precision:", precision_score(self.y_test, self.y_pred, average='weighted'))
        print("Recall:", recall_score(self.y_test, self.y_pred, average='weighted'))
        print("F1 Score:", f1_score(self.y_test, self.y_pred, average='weighted'))
        print("\nClassification Report:\n", classification_report(self.y_test, self.y_pred))
