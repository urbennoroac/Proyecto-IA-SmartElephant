


def run_ClasifU(testSize):

    import streamlit as st
    from PIL import Image                 # Para crear vectores y matrices n dimensionales
    import matplotlib.pyplot as plt     # Para la generación de gráficas a partir de los datos
    from apyori import apriori         # Para la implementación de reglas de asociación
    from st_aggrid import AgGrid
    from st_aggrid.grid_options_builder import GridOptionsBuilder
    from numpy.lib.shape_base import split
    from matplotlib import text
    import streamlit as st
    import pandas as pd                 # Para la manipulación y análisis de los datos
    import matplotlib.pyplot as plt
    from scipy.spatial.distance import cdist    # Para el cálculo de distancias
    from scipy.spatial import distance # Para el cálculo de distancias 
    import seaborn as sns     # Para la generación de gráficas a partir de los datos
    import numpy as np                  # Para crear vectores y matrices n dimensionales
    from apyori import apriori         # Para la implementación de reglas de asociación
    from fpdf import FPDF
    import base64
    from io import BytesIO
    from typing import BinaryIO
    import time
    timestr = time.strftime("%Y%m%d-%H%M%S")
    from sklearn.cluster import KMeans
    from sklearn.metrics import pairwise_distances_argmin_min
    from kneed import KneeLocator
    import scipy.cluster.hierarchy as shc
    from sklearn.cluster import AgglomerativeClustering
    from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
    import streamlit as st
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    import sqlite3 
    conn = sqlite3.connect('data.db')
    c = conn.cursor()


    def create_configA():
        c.execute('CREATE TABLE IF NOT EXISTS configurationAlgo(idA INT, valores TEXT)')

    def create_Clasificacion():
        c.execute('CREATE TABLE IF NOT EXISTS Clasificacion(nombre TEXT, nombreClasificacion TEXT, Clasificacion TEXT)')

    def add_Clasificacion(nombre, nombrec, clasif):
        c.execute('INSERT INTO Clasificacion(nombre, nombreClasificacion, Clasificacion) VALUES (?, ?, ?)' , (nombre, nombrec, clasif))
        conn.commit()

    def add_configA(idA, valores):
        c.execute('INSERT INTO configurationAlgo(idA, valores) VALUES (?,?)',(idA, valores))
        conn.commit()

    def delte_configA():
        c.execute('DELETE FROM configurationAlgo')
        conn.commit()

    def view_alldata():
        c.execute('SELECT * FROM configurationAlgo')
        data = c.fetchall()
        return data

    def view_alldataC():
        c.execute('SELECT * FROM Clasificacion')
        data = c.fetchall()
        return data



    def create_download_link(val, filename):
        b64 = base64.b64encode(val)  # val looks like b'...'
        return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="{filename}.pdf">Download file</a>'

    def get_image_download_link(img,filename,text):
        buffered = BytesIO()
        img.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        href =  f'<a href="data:file/txt;base64,{img_str}" download="{filename}">{text}</a>'
        return href

    def csv_downloader(data):
        csvfile = data.to_csv()
        b64 = base64.b64encode(csvfile.encode()).decode()
        new_filename = "new_text_file_{}_.csv".format(timestr)
        st.markdown("#### Download CSV ###")
        href = f'<a href="data:file/csv;base64,{b64}" download="{new_filename}">Descarga la matriz</a>'
        st.markdown(href,unsafe_allow_html=True)

    st.set_option('deprecation.showPyplotGlobalUse', False) # Para evitar que se muestre el warning de matplotlib

    st.info("El tamaño de los datos de prueba fue: ")
    st.info(testSize)
    st.title('Módulo: Clasificación')
    texto = '<p style="font-family:monospace;color:Black; font-size: 20px;">Regresión Logística</p>'
    st.markdown(texto, unsafe_allow_html=True)
    st.markdown("""
        Predice etiquetas de una o más clases de tipo discretas (0, 1, 2) o nominales (A, B, C o positivo, negativo; y otros).
        Para esta clasificación se construye un modelo a través de un conjunto de entrenamiento (training).
        Se evalúa el modelo con un conjunto de prueba, que es independiente del entrenamiento.
        De lo contrario, se produce un sobre-ajuste (ajuste excesivo).
        La regresión logística es otro tipo de algoritmo de aprendizaje supervisado cuyo objetivo es
        predecir valores binarios (0 o 1).
    """)

    datosRegresionL = st.file_uploader("Selecciona un archivo válido para trabajar con la regresión logística: ", type=["csv","txt"])
    if datosRegresionL is not None:
        DatosRegresionL = pd.read_csv(datosRegresionL)
        datosDelPronostico = []
        texto = '<p style="font-family:monospace;color:Black; font-size: 20px;">Datos Cargados</p>'
        st.markdown(texto, unsafe_allow_html=True)
        gb = GridOptionsBuilder.from_dataframe(DatosRegresionL)
        gb.configure_pagination()
        gb.configure_side_bar()
        gb.configure_default_column(groupable=True, value=True, enableRowGroup=True, aggFunc="sum", editable=True)
        gridOptions = gb.build()
        AgGrid(DatosRegresionL, gridOptions=gridOptions, enable_enterprise_modules=True, height= 500, fit_columns_on_grid_load=True)
        for i in range(0, len(DatosRegresionL.columns)):
            datosDelPronostico.append(DatosRegresionL.columns[i])


        MatrizCorr = DatosRegresionL.corr(method='pearson')
        MatrizInf = np.triu(MatrizCorr)
        data_result = view_alldata()
        clean_db = pd.DataFrame(data_result,columns=["idA", "valores"])
        variablePronostico = clean_db.iloc[0, 1]


        with st.expander("Obtener Mapa de Calor: "):
            st.header("Mapa de Calor: ")
            with st.spinner("Cargando mapa de calor..."):
                plt.figure(figsize=(14,7))
                MatrizInf = np.triu(MatrizCorr)
                sns.heatmap(MatrizCorr, cmap='RdBu_r', annot=True, mask=MatrizInf)
                plt.title('Mapa de calor de la correlación que existe entre variables')
                st.pyplot()

        
        Y = np.array(DatosRegresionL[variablePronostico])
            

        # Variables predictoras
       

        lon = len(clean_db) - 1 
        datos2= clean_db.loc[1:lon, "valores"]



        X = np.array(DatosRegresionL[datos2])
        
          
                
        # Seleccionar los datos que se quieren visualizar
        

        # Aplicación del algoritmo: Regresión Logística
        # Se importan las bibliotecas necesarias 
        from sklearn import linear_model # Para la regresión lineal / pip install scikit-learn
        from sklearn import model_selection 
        from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
        st.session_state.counter2 = testSize
        X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=testSize, random_state=1234, shuffle=True)
        # Datos de entrenamiento: 70, 75 u 80% de los datos
        # Datos de prueba: 20, 25 o 30% de los datos

        # DIVISIÓN DE LOS DATOS EN ENTRENAMIENTO Y PRUEBA
        #st.dataframe(X_train)
        #st.dataframe(Y_train)

        # Se entrena el modelo a partir de los datos de entrada
        Clasificacion = linear_model.LogisticRegression() # Se crea el modelo
        Clasificacion.fit(X_train, Y_train) # Se entrena el modelo

        # A partir de las probabilidades obtenidas anteriormente se hacen las predicciones

        
        # Matriz de clasificación
        st.subheader('Matriz de clasificación')
        Y_Clasificacion = Clasificacion.predict(X_validation)
        Matriz_Clasificacion = pd.crosstab(Y_validation.ravel(), Y_Clasificacion, rownames=['Real'], colnames=['Clasificación'])
        #st.table(Matriz_Clasificacion)
        
        col1, col2 = st.columns(2)
        col1.success('Verdaderos Positivos (VP): '+str(Matriz_Clasificacion.iloc[1,1]))
        col2.error('Falsos Negativos (FN): '+str(Matriz_Clasificacion.iloc[1,0]))
        col2.error('Verdaderos Negativos (VN): '+str(Matriz_Clasificacion.iloc[0,0]))
        col1.success('Falsos Positivos (FP): '+str(Matriz_Clasificacion.iloc[0,1]))

        # Reporte de clasificación
        st.subheader('Reporte de clasificación')
        with st.expander("Información y Reporte del Algoritmo de Clasificacion"):
            #st.write(classification_report(Y_validation, Y_Clasificacion))
            st.info("Exactitud promedio de la validación: "+str(Clasificacion.score(X_validation, Y_validation).round(6)*100)+" %")
            precision = float(classification_report(Y_validation, Y_Clasificacion).split()[10])*100
            st.info("Precisión: "+ str(precision)+ " %")
            st.error("Tasa de error: "+str((1-Clasificacion.score(X_validation, Y_validation))*100)+" %")
            sensibilidad = float(classification_report(Y_validation, Y_Clasificacion).split()[11])*100
            st.info("Sensibilidad: "+ str(sensibilidad)+ " %")
            especificidad = float(classification_report(Y_validation, Y_Clasificacion).split()[6])*100
            st.info("Especificidad: "+ str(especificidad)+" %")
        
        

        st.subheader('Clasificación del Modelo de Entrenamiento')
        create_Clasificacion()
        with st.expander("Clasificación de Nuevos Datos"):
            st.subheader('Clasificación de casos')
            sujetoN = st.text_input("Ingrese el nombre o ID del sujeto u objeto que desea clasificar: ")

            dato = []
            x = 0
            for p in range(1, len(datos2)+1):
                dato.append(st.number_input(datos2[p][:], step=0.1))
 
            
            if st.checkbox("Dar clasificación: "):
                if Clasificacion.predict([dato])[0] == 0:
                    clasif = (Clasificacion.predict([dato])[0])
                    clasifs = str(clasif)
                    st.error("Con un algoritmo que tiene una exactitud del: "+str(round(Clasificacion.score(X_validation, Y_validation)*100,2))+"%, la clasificación para el sujeto "+str(sujetoN)+", tomando en cuenta como variable predictora: '"+str(variablePronostico)+"', fue de 0 (CERO)")
                    add_Clasificacion(sujetoN, variablePronostico, clasifs)
                elif Clasificacion.predict([dato])[0] == 1:
                    clasif = (Clasificacion.predict([dato])[0])
                    clasifs = str(clasif)
                    st.success("Con un algoritmo que tiene una exactitud del: "+str(round(Clasificacion.score(X_validation, Y_validation)*100,2))+"%, la clasificación para el sujeto "+str(sujetoN)+", tomando en cuenta como variable predictora: '"+str(variablePronostico)+"', fue de 1 (UNO)")
                    add_Clasificacion(sujetoN, variablePronostico, clasifs)
                else:
                    st.warning("El resultado no pudo ser determinado, intenta hacer una buena selección de variables")
        
        data_result = view_alldataC()
        cldb = pd.DataFrame(data_result,columns=["nombre", "nombreClasificacion", "Clasificacion"])
        gb = GridOptionsBuilder.from_dataframe(cldb)
        gb.configure_pagination()
        gb.configure_side_bar()
        gb.configure_default_column(groupable=True, value=True, enableRowGroup=True, aggFunc="sum", editable=True)
        gridOptions = gb.build()
        AgGrid(cldb, gridOptions=gridOptions, enable_enterprise_modules=True, height= 200, fit_columns_on_grid_load=True)

        