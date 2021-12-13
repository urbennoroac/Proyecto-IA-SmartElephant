  


def run_Arboles():

    import streamlit as st
    from PIL import Image                # Para crear vectores y matrices n dimensionales
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
    import time
    timestr = time.strftime("%Y%m%d-%H%M%S")
    
    import streamlit as st
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    import math      
    import os   # Para la visualización de datos basado en matplotlib
    #%matplotlib inline 
    import streamlit as st   
    import sklearn         # Para la generación de gráficas interactivas
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    from sklearn import model_selection
    import sqlite3 
    conn = sqlite3.connect('data.db')
    c = conn.cursor()





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

    def create_configAC():
        c.execute('CREATE TABLE IF NOT EXISTS configurationADC(idA INT, valores TEXT)')

    def add_configAC(idA, valores):
        c.execute('INSERT INTO configurationADC(idA, valores) VALUES (?,?)',(idA, valores))
        conn.commit()

    def delte_configAC():
        c.execute('DELETE FROM configurationADC')
        conn.commit()

    def view_alldata():
        c.execute('SELECT * FROM configurationADC')
        data = c.fetchall()
        return data

    def create_configAR():
        c.execute('CREATE TABLE IF NOT EXISTS configurationADR(valores TEXT)')

    def add_configAR(valores):
        c.execute('INSERT INTO configurationADR(valores) VALUES (?)',(valores,))
        conn.commit()

    def create_configARDi():
        c.execute('CREATE TABLE IF NOT EXISTS configurationADCDi(valores TEXT)')

    def add_configARDi(valores):
        c.execute('INSERT INTO configurationADCDi(valores) VALUES (?)',(valores,))
        conn.commit()

    def delte_configARDi():
        c.execute('DELETE FROM configurationADCDi')
        conn.commit()

    def delte_configAR():
        c.execute('DELETE FROM configurationADR')
        conn.commit()


    def create_testSize():
        c.execute('CREATE TABLE IF NOT EXISTS Test(idA INT, val REAL)')

    def add_testSize(idA, valores):
        c.execute('INSERT INTO Test(idA, val) VALUES (?, ?)',(idA, valores))
        conn.commit()

    def delete_testSize():
        c.execute('DELETE FROM Test')
        conn.commit()

    def create_minmaxdep():
        c.execute('CREATE TABLE IF NOT EXISTS ArbolConfig(idA INT, dep REAL, mins REAL, minl REAL)')

    def add_minmaxdep(idA, dep, mins, minl):
        c.execute('INSERT INTO ArbolConfig(idA, dep, mins, minl) VALUES (?, ?, ?, ?)',(idA, dep, mins, minl))
        conn.commit()

    def delete_minmaxdep():
        c.execute('DELETE FROM ArbolConfig')
        conn.commit()

    #Librerías para Clustering Particional


    st.set_option('deprecation.showPyplotGlobalUse', False) # Para evitar que se muestre el warning de matplotlib

    st.title('Módulo: Árboles de Decisión')
    texto = '<p style="font-family:monospace;color:Black; font-size: 20px;">Regresión y Clasificación</p>'
    st.markdown(texto, unsafe_allow_html=True)
    datosArboles = st.file_uploader("Seleccione el conjunto de datos con el que se va a trabajar: ", type=["csv","txt"])
    if datosArboles is not None:
        datosArbolesDecision = pd.read_csv(datosArboles)
        texto2 = '<p style="font-family:monospace;color:Black; font-size: 20px;">Datos Cargados</p>'
        st.markdown(texto2, unsafe_allow_html=True)
        gb = GridOptionsBuilder.from_dataframe(datosArbolesDecision)
        gb.configure_pagination()
        gb.configure_side_bar()
        gb.configure_default_column(groupable=True, value=True, enableRowGroup=True, aggFunc="sum", editable=True)
        gridOptions = gb.build()
        AgGrid(datosArbolesDecision, gridOptions=gridOptions, enable_enterprise_modules=True, height= 500, fit_columns_on_grid_load=True)

        datosDelPronostico = []
        for i in range(0, len(datosArbolesDecision.columns)):
            datosDelPronostico.append(datosArbolesDecision.columns[i])
        
        st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
        opcionArbol1O2 = st.radio("Selecciona el tipo de árbol de decisión que deseas utilizar: ", ("Árbol de Decisión (Regresión)", "Árbol de Decisión (Clasificación)"))
        

        if opcionArbol1O2 == "Árbol de Decisión (Regresión)":

            opcionVisualizacionArbolD = st.select_slider('Selecciona una opción: ', options=["Visualización de Datos", "Correlación","Aplicar Algoritmo"], value="Visualización de Datos")

            if opcionVisualizacionArbolD == "Visualización de Datos":
                
                dAD = pd.DataFrame(datosArbolesDecision)
                texto2 = '<p style="font-family:monospace;color:Black; font-size: 20px;">Estadisticas</p>'
                st.markdown(texto2, unsafe_allow_html=True)
                eAD = pd.DataFrame(datosArbolesDecision.describe())
                st.dataframe(eAD)
                
            if opcionVisualizacionArbolD == "Correlación":
                MatrizCorr = datosArbolesDecision.corr(method='pearson')
                texto2 = '<p style="font-family:monospace;color:Black; font-size: 20px;">Matriz de correlaciones</p>'
                st.markdown(texto2, unsafe_allow_html=True)
                mt = pd.DataFrame(MatrizCorr)
                gb = GridOptionsBuilder.from_dataframe(mt)
                gb.configure_pagination()
                gb.configure_side_bar()
                gb.configure_default_column(groupable=True, value=True, enableRowGroup=True, aggFunc="sum", editable=True)
                gridOptions = gb.build()
                AgGrid(mt, gridOptions=gridOptions, enable_enterprise_modules=True, height= 500, fit_columns_on_grid_load=True)

                # SELECCIONAR VARIABLES PARA PRONOSTICAR
                #try:
                    #st.subheader("Correlación de variables: ")
                    #variableCorrelacion = st.selectbox("", MatrizCorr.columns) 
                    #st.table(MatrizCorr[variableCorrelacion].sort_values(ascending=False)[:10])  #Top 10 valores
                #except:
                    #st.warning("Selecciona una variable con datos válidos...")

                # Mapa de calor de la relación que existe entre variables
                st.header("Mapa de calor de la correlación entre variables: ")
                plt.figure(figsize=(14,7))
                MatrizInf = np.triu(MatrizCorr)
                sns.heatmap(MatrizCorr, cmap='RdBu_r', annot=True, mask=MatrizInf)
                plt.title('Mapa de calor de la correlación que existe entre variables')
                st.pyplot()

            if opcionVisualizacionArbolD == "Aplicar Algoritmo":
                texto2 = '<p style="font-family:monospace;color:Black; font-size: 20px;">Definición de Variables de Predictoras</p>'
                st.markdown(texto2, unsafe_allow_html=True)


                st.subheader('Variables Predictoras')
                datosADeciR = st.multiselect("Datos", datosDelPronostico)
                create_configAR()
                delte_configAR()
                for e in datosADeciR:
                    add_configAR(e)

                X = np.array(datosArbolesDecision[datosADeciR]) 

                st.subheader('Variable Clase')
                variablePronostico = st.selectbox("Variable a clasificar", datosArbolesDecision.columns.drop(datosADeciR),index=3)
                create_configARDi()
                delte_configARDi()
                add_configARDi(variablePronostico)
                Y = np.array(datosArbolesDecision[variablePronostico])
                
                if X.size > 0:
                        
                    
                    # Aplicación del algoritmo: Regresión Logística
                    # Se importan las bibliotecas necesarias 
                    from sklearn import linear_model # Para la regresión lineal / pip install scikit-learn
                    from sklearn import model_selection 
                    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

                    st.header('Criterio de división')
                    st.markdown("""
                    División de datos:\n
                    **Entrenamiento**\n
                    **Datos de Prueba** Rango de 20% a 30%\n
                    """)
                    testSize = st.number_input('Selecciona el tamaño del test', min_value=0.2, max_value=0.3, value=0.2, step=0.01)
                    create_testSize()
                    delete_testSize()
                    add_testSize(1, testSize)
                    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=testSize, random_state=1234, shuffle=True)
                    
                    
                    st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
                    opcionCaracteristicas = st.radio("Selecciona el tipo de parametros para generar el árbol de decisión que deseas utilizar: ", ("Valores Predeterminados", "Asignar Configuraciones"))

                    if opcionCaracteristicas == "Asignar Configuraciones":
                        st.write("Selecciona los valores que requieras para entrenar el modelo: ")
                        column1, column2, column3 = st.columns(3)

                        Max_depth = column1.number_input('Máxima profundidad del árbol (max_depth)', min_value=1, value=8)
                        Min_samples_split = column2.number_input('min_samples_split', min_value=1, value=2)
                        Min_samples_leaf = column3.number_input('min_samples_leaf', min_value=1, value=1)
                        create_minmaxdep()
                        delete_minmaxdep()
                        add_minmaxdep(1, Max_depth, Min_samples_split, Min_samples_leaf)

                        PronosticoAD = DecisionTreeRegressor(max_depth=Max_depth, min_samples_split=Min_samples_split, min_samples_leaf=Min_samples_leaf)
                    if opcionCaracteristicas == "Valores Predeterminados":
                        delete_minmaxdep()
                        PronosticoAD = DecisionTreeRegressor()
                    
                    
                    PronosticoAD.fit(X_train, Y_train)
                    
                    Y_Pronostico = PronosticoAD.predict(X_test)
                    st.subheader('Datos Reales vs Datos del Modelo')
                    Valores = pd.DataFrame(Y_test, Y_Pronostico)
                    st.dataframe(Valores)
                    

                    st.subheader('Gráfico: ')
                    plt.figure(figsize=(20, 5))
                    plt.plot(Y_test, color='green', marker='o', label='Y_test')
                    plt.plot(Y_Pronostico, color='red', marker='o', label='Y_Pronostico')
                    plt.xlabel('Paciente')
                    plt.ylabel('Tamaño del tumor')
                    plt.title('Pacientes con tumores cancerígenos')
                    plt.grid(True)
                    plt.legend('Datos Reales vs Datos del Modelo')
                    plt.savefig('RealesvsModelo.png')
                    st.pyplot()
                    image = Image.open('RealesvsModelo.png')
                    rgb_im = image.convert('RGB')
                    rgb_im.save('RealesvsModelo.png')
                    image = Image.open('RealesvsModelo.png')
                    st.markdown(get_image_download_link(image,'RealesvsModelo.png','Download '+'RealesvsModelo.png'), unsafe_allow_html=True)

                    # Reporte de clasificación
                    texto2 = '<p style="font-family:monospace;color:Black; font-size: 20px;">Reporte de Clasificación</p>'
                    st.markdown(texto2, unsafe_allow_html=True)
                    with st.expander("Obtener Reporte"):
                        st.info('Importancia variables: '+str(PronosticoAD.feature_importances_))
                        st.info("MAE: "+str(mean_absolute_error(Y_test, Y_Pronostico)))
                        st.info("MSE: "+str(mean_squared_error(Y_test, Y_Pronostico)))
                        varmse = mean_squared_error(Y_test, Y_Pronostico)
                        varmse = math.sqrt(varmse)
                        st.info("RMSE: "+str(varmse))   #True devuelve MSE, False dvuelve RMSE
                        st.info('Score (exactitud promedio de la validación): '+str(r2_score(Y_test, Y_Pronostico).round(6)*100)+" %")
                    
                    st.subheader('Importancia de las variables: ')
                    Importancia = pd.DataFrame({'Variable': list(datosArbolesDecision[datosADeciR]),
                            'Importancia': PronosticoAD.feature_importances_}).sort_values('Importancia', ascending=False)
                    st.table(Importancia)


                    import graphviz
                    from sklearn.tree import export_graphviz

                    st.subheader('Árbol de decisión')
                    from sklearn.tree import plot_tree
                    with st.expander('Árbol de decisión'):
                        with st.spinner('Generando árbol de decisión...'):
                            plt.figure(figsize=(16,16))  
                            plot_tree(PronosticoAD, feature_names = list(datosArbolesDecision[datosADeciR]))
                            plt.savefig('ArbolR.png')
                            st.pyplot()
                            image = Image.open('ArbolR.png')
                            rgb_im = image.convert('RGB')
                            rgb_im.save('ArbolR.png')
                            image = Image.open('ArbolR.png')
                            st.markdown(get_image_download_link(image,'ArbolR.png','Download '+'ArbolR.png'), unsafe_allow_html=True)

                    from sklearn.tree import export_text
                    with st.expander('Árbol en formato de texto: '):
                        Reporte = export_text(PronosticoAD, feature_names = list(datosArbolesDecision[datosADeciR]))
                        st.text(Reporte)
                        st.download_button(label = "Descarga Reporte", data = Reporte, file_name = "Reporte.txt")

                    


                elif X.size == 0:
                    st.warning("No se ha seleccionado ninguna variable")

        if opcionArbol1O2 == "Árbol de Decisión (Clasificación)":
            opcionVisualizacionArbolD = st.select_slider('Selecciona una opción', options=["Visualización de Datos", "Correlación","Aplicar Algoritmo"], value="Visualización de Datos")

            if opcionVisualizacionArbolD == "Visualización de Datos":

                variablePronostico = st.selectbox("Variable a clasificar", datosDelPronostico,index=1)
                st.write(datosArbolesDecision.groupby(variablePronostico).size())


                try:
                    datos = st.multiselect("Datos", datosDelPronostico, default=[datosDelPronostico[2], datosDelPronostico[3]])
                    dato1=datos[0][:]
                    dato2=datos[1][:]

                    with st.expander("Gráfico de dispersión: "):
                        with st.spinner("Cargando gráfico de dispersión..."):
                            sns.scatterplot(x=dato1, y=dato2, data=datosArbolesDecision, hue=variablePronostico)
                            plt.title('Gráfico de dispersión')
                            plt.xlabel(dato1)
                            plt.ylabel(dato2)
                            st.pyplot() 

                except:
                    st.warning("Selecciona solo dos datos")

                
            if opcionVisualizacionArbolD == "Correlación":
                MatrizCorr = datosArbolesDecision.corr(method='pearson')
                textocor = '<p style="font-family:monospace;color:Black; font-size: 20px;">Matriz de Correlación: </p>'
                st.markdown(textocor, unsafe_allow_html=True)
                mt = pd.DataFrame(MatrizCorr)
                gb = GridOptionsBuilder.from_dataframe(mt)
                gb.configure_pagination()
                gb.configure_side_bar()
                gb.configure_default_column(groupable=True, value=True, enableRowGroup=True, aggFunc="sum", editable=True)
                gridOptions = gb.build()
                AgGrid(mt, gridOptions=gridOptions, enable_enterprise_modules=True, height= 500, fit_columns_on_grid_load=True)

                
                textomap = '<p style="font-family:monospace;color:Black; font-size: 20px;">Mapa de Calor: </p>'
                st.markdown(textomap, unsafe_allow_html=True)
                plt.figure(figsize=(14,7))
                MatrizInf = np.triu(MatrizCorr)
                sns.heatmap(MatrizCorr, cmap='RdBu_r', annot=True, mask=MatrizInf)
                plt.title('Mapa de calor de la correlación que existe entre variables')
                st.pyplot()

            if opcionVisualizacionArbolD == "Aplicar Algoritmo":

                
                textocor = '<p style="font-family:monospace;color:Black; font-size: 20px;">Definición de Variables Predictoras y de Clase:: </p>'
                st.markdown(textocor, unsafe_allow_html=True)
               
                st.subheader('Selección de la variable Clase')
                st.markdown('**Seleccione variables con valores discretos**')
                variablePronostico = st.selectbox("Variable a clasificar", datosArbolesDecision.columns,index=1)

                # Comprobando que la variable clase sea binaria
                if datosArbolesDecision[variablePronostico].nunique() == 2:
                    st.info("Selecciona las etiquetas que desea manejar")
                    col1, col2 = st.columns(2)
                    # Comprobando el tipo de dato de la variable clase
                    if type(datosArbolesDecision[variablePronostico].value_counts().index[1]) and type(datosArbolesDecision[variablePronostico].value_counts().index[0]) != np.int64:
                        
                        binario1 = col1.text_input("", datosArbolesDecision[variablePronostico].value_counts().index[0])
                        binario2 = col2.text_input("", datosArbolesDecision[variablePronostico].value_counts().index[1])

                        col1.success("La etiqueta '"+str(datosArbolesDecision[variablePronostico].value_counts().index[0])+"', cambió por la etiqueta: "+binario1)
                        col2.success("La etiqueta '"+str(datosArbolesDecision[variablePronostico].value_counts().index[1])+"', cambió por la etiqueta: "+binario2)

                        with st.expander("DataFrame con las nuevas etiquetas: "):
                            st.subheader("Dataframe corregido: ")
                            datosArbolesDecision = datosArbolesDecision.replace({str(datosArbolesDecision[variablePronostico].value_counts().index[1]): binario2, str(datosArbolesDecision[variablePronostico].value_counts().index[0]): binario1})
                            st.dataframe(datosArbolesDecision)
                            Y = np.array(datosArbolesDecision[variablePronostico])
                        

                    

                    datos = st.multiselect("Selecciona las variables predictoras", datosArbolesDecision.columns.drop(variablePronostico))
                    create_configAC()
                    delte_configAC()
                    add_configAC(1, variablePronostico)
                    n = 2
                    for e in datos:
                        add_configAC(n,e)
                        n += 1

                    data_result = view_alldata()
                    clean_db = pd.DataFrame(data_result,columns=["idA", "valores"])
                    gb = GridOptionsBuilder.from_dataframe(clean_db)

                    gb.configure_pagination()
                    gb.configure_side_bar()
                    gb.configure_default_column(groupable=True, value=True, enableRowGroup=True, aggFunc="sum", editable=True)
                    gridOptions = gb.build()
                    AgGrid(clean_db, gridOptions=gridOptions, enable_enterprise_modules=True, height= 200, fit_columns_on_grid_load=True)
                    X = np.array(datosArbolesDecision[datos]) 
                    if X.size > 0:
                        
                    
                        
                        from sklearn.tree import DecisionTreeClassifier
                        from sklearn.metrics import classification_report,confusion_matrix, accuracy_score
                        from sklearn import model_selection
                        # Aplicación del algoritmo: Regresión Logística
                        st.header('Criterio de división')
                        st.markdown("""
                        División de datos:\n
                        **Entrenamiento** \n
                        **Datos de Prueba** Rango de 20% a 30%\n
                        """)
                        testSize = st.number_input('Selecciona el tamaño del test', min_value=0.2, max_value=0.3, value=0.2, step=0.01)
                        create_testSize()
                        delete_testSize()
                        add_testSize(2, testSize)
                        X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=testSize, random_state=0, shuffle=True)
                       
                        st.header('Parámetros del árbol de decisión: ')
                        

                        st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
                        opcionCaracteristicas = st.radio("Selecciona el tipo de parametros para generar el árbol de decisión que deseas utilizar: ", ("Valores Predeterminados", "Asignar Configuraciones"))

                        if opcionCaracteristicas == "Asignar Configuraciones":
                            st.write("Selecciona los valores que requieras para entrenar el modelo: ")
                            column1, column2, column3 = st.columns(3)

                            Max_depth = column1.number_input('Máxima profundidad del árbol (max_depth)', min_value=1, value=8)
                            Min_samples_split = column2.number_input('min_samples_split', min_value=1, value=2)
                            Min_samples_leaf = column3.number_input('min_samples_leaf', min_value=1, value=1)
                            create_minmaxdep()
                            delete_minmaxdep()
                            add_minmaxdep(1, Max_depth, Min_samples_split, Min_samples_leaf)
                
                            ClasificacionAD = DecisionTreeClassifier(max_depth=Max_depth, min_samples_split=Min_samples_split, min_samples_leaf=Min_samples_leaf)
                        if opcionCaracteristicas == "Valores Predeterminados":
                            delete_minmaxdep()
                            ClasificacionAD = DecisionTreeClassifier()

                        ClasificacionAD.fit(X_train, Y_train)

                        #Se etiquetan las clasificaciones
                        Y_Clasificacion = ClasificacionAD.predict(X_validation)
                        st.markdown('Se etiquetan las clasificaciones (Real vs Clasificado)')
                        Valores = pd.DataFrame(Y_validation, Y_Clasificacion)
                        st.dataframe(Valores)


                        # Matriz de clasificación
                        st.subheader('Matriz de clasificación o de Confusión')
                        Y_Clasificacion = ClasificacionAD.predict(X_validation)
                        Matriz_Clasificacion = pd.crosstab(Y_validation.ravel(), Y_Clasificacion, rownames=['Real'], colnames=['Clasificación'])
                        
                        
                        col1, col2 = st.columns(2)
                        col1.success('Verdaderos Positivos (VP): '+str(Matriz_Clasificacion.iloc[1,1]))
                        col2.error('Falsos Negativos (FN): '+str(Matriz_Clasificacion.iloc[1,0]))
                        col2.success('Verdaderos Negativos (VN): '+str(Matriz_Clasificacion.iloc[0,0]))
                        col1.error('Falsos Positivos (FP): '+str(Matriz_Clasificacion.iloc[0,1]))

                     
                        st.subheader('Reporte del algoritmo de clasificación:')
                        with st.expander("Reporte de Clasificación"):
                            
                            st.success("Criterio: "+str(ClasificacionAD.criterion))
                            importancia = ClasificacionAD.feature_importances_.tolist()
                            
                            st.success("Importancia de las variables: "+str(importancia))
                            st.success("Exactitud promedio de la validación: "+ str(ClasificacionAD.score(X_validation, Y_validation).round(6)*100)+" %")
                            precision = float(classification_report(Y_validation, Y_Clasificacion).split()[10])*100
                            st.success("Precisión: "+ str(precision)+ "%")
                            st.error("Tasa de error: "+str((1-ClasificacionAD.score(X_validation, Y_validation))*100)+"%")
                            sensibilidad = float(classification_report(Y_validation, Y_Clasificacion).split()[11])*100
                            st.success("Sensibilidad: "+ str(sensibilidad)+ "%")
                            especificidad = float(classification_report(Y_validation, Y_Clasificacion).split()[6])*100
                            st.success("Especificidad: "+ str(especificidad)+"%")
                        

                        st.subheader('Importancia de las variables: ')
                        Importancia = pd.DataFrame({'Variable': list(datosArbolesDecision[datos]),
                                'Importancia': ClasificacionAD.feature_importances_}).sort_values('Importancia', ascending=False)
                        st.table(Importancia)


                        import graphviz
                        from sklearn.tree import export_graphviz
                        # Se crea un objeto para visualizar el árbol
                        # Se incluyen los nombres de las variables para imprimirlos en el árbol
                        st.subheader('Árbol de decisión (Clasificación)')

                        from sklearn.tree import plot_tree
                        with st.expander('Visualizar Arbol:'):
                            with st.spinner('Generando árbol de decisión...'):
                                plt.figure(figsize=(16,16))  
                                plot_tree(ClasificacionAD, feature_names = list(datosArbolesDecision[datos]),class_names=Y_Clasificacion)
                                plt.savefig('ArbolC.png')
                                st.pyplot()
                                image = Image.open('ArbolC.png')
                                rgb_im = image.convert('RGB')
                                rgb_im.save('ArbolC.png')
                                image = Image.open('ArbolC.png')
                                st.markdown(get_image_download_link(image,'ArbolC.png','Download '+'ArbolC.png'), unsafe_allow_html=True)

                        from sklearn.tree import export_text
                        with st.expander('Generar Arbol Formato TXT: '):
                            Reporte = export_text(ClasificacionAD, feature_names = list(datosArbolesDecision[datos]))
                            st.text(Reporte)
                            st.download_button(label = "Descarga Reporte", data = Reporte, file_name = "Reporte.txt")

                    

                    elif X.size == 0:
                        st.warning("No se ha seleccionado ninguna variable")

                elif datosArbolesDecision[variablePronostico].nunique() != 2:
                    st.error("Seleccione una variable que contena valores binarios")

