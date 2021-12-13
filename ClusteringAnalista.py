  


def run_Cluster():

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
    from sklearn.cluster import KMeans
    from sklearn.metrics import pairwise_distances_argmin_min
    from kneed import KneeLocator
    import scipy.cluster.hierarchy as shc
    from sklearn.cluster import AgglomerativeClustering
    from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
    import streamlit as st
    from sklearn.preprocessing import StandardScaler, MinMaxScaler





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



    #Librerías para el clustering jerárquico 
    

    #Librerías para Clustering Particional

    st.set_option('deprecation.showPyplotGlobalUse', False) # Para evitar que se muestre el warning de matplotlib

    st.title('Módulo: Clustering')
    texto = '<p style="font-family:monospace;color:Black; font-size: 20px;">Jerárquico (Ascendente) / Particional (K-Means)</p>'
    st.markdown(texto, unsafe_allow_html=True)
    st.markdown(""" La inteligencia artificial aplicada en el análisis de conglomerados implica la segmentación e identificación de grupos de objetos (ítems), unidos por las características que tienen en común (aprendizaje no supervisado).""")
    st.markdown("""El objetivo es dividir un grupo heterogéneo de elementos en varios grupos naturales (regiones o segmentos homogéneos), en función de sus similitudes.""")
    st.markdown("""Para realizar el montaje es necesario conocer el grado de similitud (medida de distancia) entre los elementos.""")
    datosCluster = st.file_uploader("Selecciona un archivo para comenzar (CSV / TXT): ", type=["csv","txt"])
    if datosCluster is not None:
        datosClustering = pd.read_csv(datosCluster)
        datosDelPronostico = []
        for i in range(0, len(datosClustering.columns)):
            datosDelPronostico.append(datosClustering.columns[i])

        st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
        opcionClustering1O2 = st.radio("Selecciona el algoritmo de clustering que deseas trabajar: ", ('Clustering Jerárquico (Ascendente)', 'Clustering Particional (K-Means)'))

        gb = GridOptionsBuilder.from_dataframe(datosClustering)
        gb.configure_pagination()
        gb.configure_side_bar()
        gb.configure_default_column(groupable=True, value=True, enableRowGroup=True, aggFunc="sum", editable=True)
        gridOptions = gb.build()
        subtexto = '<p style="font-family:monospace;color:Black; font-size: 20px;">Datos Cargados: </p>'
        st.markdown(subtexto, unsafe_allow_html=True)
        AgGrid(datosClustering, gridOptions=gridOptions, enable_enterprise_modules=True, height= 500, fit_columns_on_grid_load=True)

        
        if opcionClustering1O2 == "Clustering Jerárquico (Ascendente)":
            st.title('Clustering Jerárquico')
            opcionVisualizacionClustersJ = st.select_slider('Etapas de Análisis', options=["Visualización", "Correlación","Aplicar Algoritmo"], value = "Visualización")

            if opcionVisualizacionClustersJ == "Visualización":
                st.subheader("Selecciona la tercera variable del grafico de dispersión")
                variablePronostico = st.selectbox("", datosClustering.columns,index=9)
            
                try:
                    # Seleccionar los datos que se quieren visualizar
                    st.subheader("Selecciona dos variables que quieras visualizar en el gráfico de dispersión: ")
                    datos = st.multiselect("", datosClustering.columns, default=[datosClustering.columns[3],datosClustering.columns[0]])
                    dato1=datos[0][:]
                    dato2=datos[1][:]

                    
                    with st.expander("Gráfico de dispersión"):
                        with st.spinner("Cargando datos..."):
                            plt.figure(figsize=(5,5))
                            sns.scatterplot(x=dato1, y=dato2, data=datosClustering, hue=variablePronostico)
                            plt.title('Gráfico de dispersión')
                            plt.xlabel(dato1)
                            plt.ylabel(dato2)
                            plt.savefig('GraficoDispersion.png')
                            image = Image.open('GraficoDispersion.png')
                            new_image = image.resize((500, 500))
                            new_image.save('GraficoDispersion.png')
                            st.pyplot()

                except:
                    st.warning("Tiene que seleccionar dos Variables")
                    

            if opcionVisualizacionClustersJ == "Correlación":
                textoC = '<p style="font-family:monospace;color:Black; font-size: 20px;">Matriz de Correlación: </p>'
                st.markdown(textoC, unsafe_allow_html=True)
                # MATRIZ DE CORRELACIONES
                MatrizCorr = datosClustering.corr(method='pearson')
                mt = pd.DataFrame(MatrizCorr)
                gb = GridOptionsBuilder.from_dataframe(MatrizCorr)
                gb.configure_pagination()
                gb.configure_side_bar()
                gb.configure_default_column(groupable=True, value=True, enableRowGroup=True, aggFunc="sum", editable=True)
                gridOptions = gb.build()
                AgGrid(MatrizCorr, gridOptions=gridOptions, enable_enterprise_modules=True, height= 400, fit_columns_on_grid_load=True)
                csv_downloader(mt)
                #try:
                    #st.subheader("Selecciona una variable para observar cómo se correlaciona con las demás: ")
                    #variableCorrelacion = st.selectbox("", datosClustering.columns) 
                    #st.markdown("**Matriz de correlaciones con la variable seleccionada:** ")
                    #st.table(MatrizCorr[variableCorrelacion].sort_values(ascending=False)[:10])  #Top 10 valores 
                #except:
                    #st.warning("Selecciona una variable con datos válidos.")

                # Mapa de calor de la relación que existe entre variables
                st.header("Generamos una mapa de calor para observar estas relaciones de manera grafica: ")
                with st.expander("Mapa de Calor"):
                    with st.spinner("Cargando mapa de calor..."):
                        plt.figure(figsize=(14,7))
                        MatrizInf = np.triu(MatrizCorr)
                        sns.heatmap(MatrizCorr, cmap='RdBu_r', annot=True, mask=MatrizInf)
                        plt.title('Mapa de calor de la correlación que existe entre variables')
                        plt.savefig('MapadeCalor.png')
                        image = Image.open('MapadeCalor.png')
                        new_image = image.resize((500, 500))
                        new_image.save('MapadeCalor.png')
                        st.pyplot()
            
            if opcionVisualizacionClustersJ == "Aplicar Algoritmo":
                

                st.header('Selecciona las variables para hacer el análisis: ')
                SeleccionVariablesJ = st.multiselect("Selecciona las variables para hacer el análisis: ", datosClustering.columns)
                MatrizClusteringJ = np.array(datosClustering[SeleccionVariablesJ])
                if MatrizClusteringJ.size > 0:
                    # Aplicación del algoritmo: 
                    estandarizar = StandardScaler()                               # Se instancia el objeto StandardScaler o MinMaxScaler 
                    MEstandarizada = estandarizar.fit_transform(MatrizClusteringJ)   # Se calculan la media y desviación y se escalan los datos
                    textoE = '<p style="font-family:monospace;color:Black; font-size: 20px;">Matriz Estandarizada: </p>'
                    st.markdown(textoE, unsafe_allow_html=True)
                    with st.expander("Matriz Estandarizada"):
                        st.table(MEstandarizada) 

                    st.subheader("Selecciona la métrica de distancias para generar el algoritmo: ")
                    metricaElegida = st.selectbox("", ('euclidean','chebyshev','cityblock','minkowski'),index=0)
                    ClusterJerarquico = shc.linkage(MEstandarizada, method='complete', metric=metricaElegida)
                    
                    graficaClusteringJ = plt.figure(figsize=(10, 5))
                    plt.title("Clustering Jerárquico (Ascendente)")
                    plt.xlabel('Observaciones')
                    plt.ylabel('Distancia')
                    Arbol = shc.dendrogram(ClusterJerarquico) #Utilizamos la matriz estandarizada
                    SelectAltura = st.slider('Selecciona el nivel de corte: ', min_value=0.0, max_value=np.max(Arbol['dcoord']),step=0.1)
                    plt.axhline(y=SelectAltura, color='black', linestyle='--') # Hace un corte en las ramas
                    with st.spinner("Cargando gráfico..."):
                        st.pyplot(graficaClusteringJ)
                    
                    numClusters = fcluster(ClusterJerarquico, t=SelectAltura, criterion='distance')
                    NumClusters = len(np.unique(numClusters))
                    st.success("El número de clústeres elegido fue de: "+ str(NumClusters))
                    
                    if st.checkbox("Ver los clústeres obtenidos: "):
                        with st.spinner("Cargando..."):
                            #Se crean las etiquetas de los elementos en los clústeres
                            MJerarquico = AgglomerativeClustering(n_clusters=NumClusters, linkage='complete', affinity=metricaElegida)
                            MJerarquico.fit_predict(MEstandarizada)
                            #MJerarquico.labels_

                            datosClustering = datosClustering[SeleccionVariablesJ]
                            datosClustering['clusterH'] = MJerarquico.labels_
                            textoCl = '<p style="font-family:monospace;color:Black; font-size: 20px;">DatFrame con Clusters Obtenidos: </p>'
                            st.markdown(textoCl, unsafe_allow_html=True)
                            dcf = pd.DataFrame(datosClustering)
                            gb = GridOptionsBuilder.from_dataframe(dcf)
                            gb.configure_pagination()
                            gb.configure_side_bar()
                            gb.configure_default_column(groupable=True, value=True, enableRowGroup=True, aggFunc="sum", editable=True)
                            gridOptions = gb.build()
                            AgGrid(dcf, gridOptions=gridOptions, enable_enterprise_modules=True, height= 400, fit_columns_on_grid_load=True)
                            csv_downloader(dcf)

                            #Cantidad de elementos en los clusters
                            cantidadElementos = datosClustering.groupby(['clusterH'])['clusterH'].count() 
                            datosClustering['clusterH'] = MJerarquico.labels_
                            textoCl = '<p style="font-family:monospace;color:Black; font-size: 20px;">Cantidad de Elementos en Cada Cluster </p>'
                            st.markdown(textoCl, unsafe_allow_html=True)
                            for c in cantidadElementos.index:
                                mystring = ''
                                mystring += "En el clúster "
                                mystring += str(c)
                                mystring += " hay **"
                                mystring += str(cantidadElementos[c])
                                mystring += " elementos.**"
                                st.info(mystring)

                            # Centroides de los clusters
                            CentroidesH = datosClustering.groupby('clusterH').mean()
                            st.header("Centroides de los clústeres: ")
                            st.table(CentroidesH)

                            # Interpretación de los clusters
                            st.header("Clusters Obtenidos ")
                            with st.expander("Información de los clusters: "):
                                for i in range(NumClusters):
                                    st.subheader("Clúster "+str(i))
                                    st.write(datosClustering[datosClustering['clusterH'] == i])
                            
                            st.subheader("Centroides Obtenidos  ")
                            with st.expander("Información de los Centroides Obtenidos: "):
                                for i in range(NumClusters):
                                    st.subheader("Clúster "+str(i))
                                    st.table(CentroidesH.iloc[i])

                            with st.expander("Resultados Obtenidos para cada cluster: "):
                                for n in range(NumClusters):
                                    mystring2 = ''
                                    mystring2 += "Clúster "
                                    mystring2 += str(n)
                                    mystring2 += " Conformado por: "
                                    mystring2 += str(cantidadElementos[n])
                                    mystring2 += " elementos."
                                    st.success(mystring2)
                                    for m in range(CentroidesH.columns.size):
                                        mystring1 = ''
                                        mystring1 += "Con "
                                        mystring1 += str(CentroidesH.columns[m])
                                        mystring1 += " promedio de: "
                                        mystring1 += str(CentroidesH.iloc[n,m].round(5))
                                        st.info(mystring1)

                            st.write("")
                            conclusions = st.text_area("Conclusiones del analista de datos sobre los clústers obtenido "+str(n), " :")
                                
                            try: 
                                # Gráfico de barras de la cantidad de elementos en los clusters
                                st.header("Representación gráfica de los clústeres obtenidos: ")
                                plt.figure(figsize=(10, 5))
                                plt.scatter(MEstandarizada[:,0], MEstandarizada[:,1], c=MJerarquico.labels_)
                                plt.grid()
                                plt.savefig('ClusterG.png')
                                image = Image.open('ClusterG.png')
                                new_image = image.resize((500, 500))
                                new_image.save('ClusterG.png')
                                st.pyplot()
                            except:
                                st.warning("No se pudo graficar.")

                            export_as_pdf = st.button("Export Report")
                            item2 = []
                            if export_as_pdf:
                                pdf = FPDF()
                                pdf.add_page()
                                pdf.set_font('Arial', 'B', 12)
                                pdf.multi_cell(0, 5, 'Reporte de Resultados' + '\n')
                                pdf.image('SmarthChicoMenuReportes.png', x = None, y = None, w = 0, h = 0, type = '', link = '')
                                for c in cantidadElementos.index:
                                    mystring = ''
                                    mystring += "En el clúster "
                                    mystring += str(c)
                                    mystring += " hay **"
                                    mystring += str(cantidadElementos[c])
                                    mystring += " elementos.**"

                                    pdf.multi_cell(0, 5, mystring + '\n')
                                for n in range(NumClusters):
                                    mystring2 = ''
                                    mystring2 += "Clúster "
                                    mystring2 += str(n)
                                    mystring2 += " Conformado por: "
                                    mystring2 += str(cantidadElementos[n])
                                    mystring2 += " elementos."
                                    
                                    pdf.multi_cell(0, 5, mystring2 + '\n')
                                    for m in range(CentroidesH.columns.size):
                                        mystring1 = ''
                                        mystring1 += "Con "
                                        mystring1 += str(CentroidesH.columns[m])
                                        mystring1 += " promedio de: "
                                        mystring1 += str(CentroidesH.iloc[n,m].round(5))

                                        pdf.multi_cell(0, 5, mystring1 + '\n')
                                pdf.multi_cell(0, 5, conclusions + '\n')
                                    
                                pdf.add_page()
                                pdf.multi_cell(0, 5, 'Grafico Utilizado' + '\n')
                                pdf.multi_cell(0, 5, 'Grafico dispersión' + '\n')
                                pdf.image('GraficoDispersion.png', x = None, y = None, w = 0, h = 0, type = '', link = '')
                                pdf.add_page()
                                pdf.multi_cell(0, 5, 'MapadeCalor' + '\n')
                                pdf.image('MapadeCalor.png', x = None, y = None, w = 0, h = 0, type = '', link = '')
                                pdf.add_page()
                                pdf.multi_cell(0, 5, 'ClusterG.png' + '\n')
                                pdf.image('ClusterG.png', x = None, y = None, w = 0, h = 0, type = '', link = '')
                                html = create_download_link(pdf.output(dest="S").encode("latin-1"), "testfile")
                                st.markdown(html, unsafe_allow_html=True)
                                
                            

                elif MatrizClusteringJ.size == 0:
                    st.warning("No se ha seleccionado ninguna variable.")


        if opcionClustering1O2 == "Clustering Particional (K-Means)":
            st.title('Clustering Particional')
            VisualizacionClustersP = st.select_slider('Etapas de Análisis', options=["Visualización", "Correlación","Aplicar Algoritmo"], value = "Visualización")


            if VisualizacionClustersP == "Visualización":
                st.subheader("Selecciona la tercera variable del grafico de dispersión ")
                variablePronostico = st.selectbox("", datosClustering.columns,index=9)

                try:
                    # Seleccionar los datos que se quieren visualizar
                    st.subheader("Selecciona dos variables que quieras visualizar en el gráfico de dispersión: ")
                    datos = st.multiselect("", datosClustering.columns, default=[datosClustering.columns[3],datosClustering.columns[0]])
                    dato1=datos[0][:]
                    dato2=datos[1][:]

                    
                    with st.expander("Gráfico de dispersión"):
                        with st.spinner("Cargando datos..."):
                            plt.figure(figsize=(5,5))
                            sns.scatterplot(x=dato1, y=dato2, data=datosClustering, hue=variablePronostico)
                            plt.title('Gráfico de dispersión')
                            plt.xlabel(dato1)
                            plt.ylabel(dato2)
                            plt.savefig('GraficoDispersion.png')
                            image = Image.open('GraficoDispersion.png')
                            new_image = image.resize((500, 500))
                            new_image.save('GraficoDispersion.png')
                            st.pyplot()

                except:
                    st.warning("Tiene que seleccionar dos Variables")


            if VisualizacionClustersP == "Correlación":
                textoC2 = '<p style="font-family:monospace;color:Black; font-size: 20px;">Matriz de Correlación: </p>'
                st.markdown(textoC2, unsafe_allow_html=True)
                # MATRIZ DE CORRELACIONES
                MatrizCorr = datosClustering.corr(method='pearson')
                mt = pd.DataFrame(MatrizCorr)
                gb = GridOptionsBuilder.from_dataframe(MatrizCorr)
                gb.configure_pagination()
                gb.configure_side_bar()
                gb.configure_default_column(groupable=True, value=True, enableRowGroup=True, aggFunc="sum", editable=True)
                gridOptions = gb.build()
                AgGrid(MatrizCorr, gridOptions=gridOptions, enable_enterprise_modules=True, height= 400, fit_columns_on_grid_load=True)
                csv_downloader(mt)
                #try:
                    #st.subheader("Selecciona una variable para observar cómo se correlaciona con las demás: ")
                    #variableCorrelacion = st.selectbox("", datosClustering.columns) 
                    #st.markdown("**Matriz de correlaciones con la variable seleccionada:** ")
                    #st.table(MatrizCorr[variableCorrelacion].sort_values(ascending=False)[:10])  #Top 10 valores 
                #except:
                    #st.warning("Selecciona una variable con datos válidos.")

                # Mapa de calor de la relación que existe entre variables
                st.header("Generamos una mapa de calor para observar estas relaciones de manera grafica: ")
                with st.expander("Mapa de Calor"):
                    with st.spinner("Cargando mapa de calor..."):
                        plt.figure(figsize=(14,7))
                        MatrizInf = np.triu(MatrizCorr)
                        sns.heatmap(MatrizCorr, cmap='RdBu_r', annot=True, mask=MatrizInf)
                        plt.title('Mapa de calor de la correlación que existe entre variables')
                        plt.savefig('MapadeCalor.png')
                        image = Image.open('MapadeCalor.png')
                        new_image = image.resize((500, 500))
                        new_image.save('MapadeCalor.png')
                        st.pyplot()
            
            if VisualizacionClustersP == "Aplicar Algoritmo":
                
                st.header("Selecciona las variables para hacer el análisis: ")
                variableSeleccionadas = st.multiselect("", datosClustering.columns)
                MatrizClusteringP = np.array(datosClustering[variableSeleccionadas])

                if MatrizClusteringP.size > 0:
                    st.header('Aplicación del algoritmo: K-Means')
                    
                    # Aplicación del algoritmo: 
                    estandarizar = StandardScaler()                               # Se instancia el objeto StandardScaler o MinMaxScaler 
                    MEstandarizada = estandarizar.fit_transform(MatrizClusteringP)   # Se calculan la media y desviación y se escalan los datos
                    textoC2 = '<p style="font-family:monospace;color:Black; font-size: 20px;">Matriz Estandarizada: </p>'
                    st.markdown(textoC2, unsafe_allow_html=True)
                    with st.expander('Matriz Estandarizada: '):
                        st.dataframe(MEstandarizada) 

                    
                    #Definición de k clusters para K-means
                    #Se utiliza random_state para inicializar el generador interno de números aleatorios
                    k = st.number_input('Selecciona el número de clústeres a implementar: ', min_value=0, value=12, step=1)
                    SSE = []
                    for i in range(2, k):
                        km = KMeans(n_clusters=i, random_state=0)
                        km.fit(MEstandarizada)
                        SSE.append(km.inertia_)
                    
                    #Se grafica SSE en función de k
                    plt.figure(figsize=(10, 7))
                    plt.plot(range(2, k), SSE, marker='o')
                    plt.xlabel('Cantidad de clústeres *k*')
                    plt.ylabel('SSE')
                    plt.title('Elbow Method')
                    plt.savefig('ElbowG.png')
                    st.pyplot()
                    image = Image.open('ElbowG.png')
                    new_image = image.resize((500, 500))
                    new_image.save('ElbowG.png.png')
                    

                    kl = KneeLocator(range(2, k), SSE, curve="convex", direction="decreasing")
                    st.subheader('El codo se encuentra en el clúster número: '+str(kl.elbow))

                    plt.style.use('ggplot')
                    kl.plot_knee()
                    plt.savefig('ElbowGR.png')
                    st.pyplot()
                    image = Image.open('ElbowGR.png')
                    new_image = image.resize((300, 300))
                    new_image.save('ElbowGR.png')

                    #Se crean las etiquetas de los elementos en los clústeres
                    MParticional = KMeans(n_clusters=kl.elbow, random_state=0).fit(MEstandarizada)
                    MParticional.predict(MEstandarizada)
                    
                    datosClustering = datosClustering[variableSeleccionadas]
                    datosClustering['clusterP'] = MParticional.labels_
                    st.subheader("Dataframe con las valores de los clústeres obtenidos: ")
                    dcp = pd.DataFrame(datosClustering)
                    gb = GridOptionsBuilder.from_dataframe(dcp)
                    gb.configure_pagination()
                    gb.configure_side_bar()
                    gb.configure_default_column(groupable=True, value=True, enableRowGroup=True, aggFunc="sum", editable=True)
                    gridOptions = gb.build()
                    AgGrid(dcp, gridOptions=gridOptions, enable_enterprise_modules=True, height= 400, fit_columns_on_grid_load=True)
                    csv_downloader(dcp)

                    #Cantidad de elementos en los clusters
                    numClusters = datosClustering.groupby(['clusterP'])['clusterP'].count() 
                    st.subheader("Cantidad de elementos en los clústeres: ")
                    for i in range(kl.elbow):
                        string =''
                        string += "El clúster número "
                        string += str(i)
                        string += " tiene "
                        string += str(numClusters[i])
                        string += " elementos."
                        st.info(string)
                    
                    # Centroides de los clusters
                    CentroidesP = datosClustering.groupby(['clusterP'])[variableSeleccionadas].mean()
                    st.subheader("Centroides de los clústeres: ")
                    dcep = pd.DataFrame(CentroidesP)
                    gb = GridOptionsBuilder.from_dataframe(dcep)
                    gb.configure_pagination()
                    gb.configure_side_bar()
                    gb.configure_default_column(groupable=True, value=True, enableRowGroup=True, aggFunc="sum", editable=True)
                    gridOptions = gb.build()
                    AgGrid(dcep, gridOptions=gridOptions, enable_enterprise_modules=True, height= 400, fit_columns_on_grid_load=True)
                    csv_downloader(dcep)


                    # Interpretación de los clusters
                    st.header("Información de los clustersobtenidos: ")
                    with st.expander("Haz click para visualizar los datos contenidos en cada clúster: "):
                        for i in range(kl.elbow):
                            st.subheader("Clúster "+str(i))
                            st.write(datosClustering[datosClustering['clusterP'] == i])
                    
                    st.subheader("Centroides Obtenidos: ")
                    with st.expander("Información de los Centroides Obtenidos: "):
                        for i in range(kl.elbow):
                            st.subheader("Clúster "+str(i))
                            st.table(CentroidesP.iloc[i])

                    with st.expander("Resultados obtenidos de cada uno de los centroides de cada clúster: "):
                        for n in range(kl.elbow):
                            string1 = ''
                            st.subheader("Clúster "+str(n))
                            string1 += "Conformado por: "
                            string1 += str(numClusters[n])
                            string1 += " elementos"
                            st.success(string1)
                            for m in range(CentroidesP.columns.size):
                                string2 = ''
                                string2 += " Con "
                                string2 += str(CentroidesP.columns[m])
                                string2 += " promedio de: "
                                string2 += str(CentroidesP.iloc[n,m].round(5))
                                st.info(string2)
                                
                           
                    conclusion = st.text_area("Conclusiones del analista sobre los clúster: ")

                    try:
                        st.header("Representación gráfica de los clústeres obtenidos: ")
                        from mpl_toolkits.mplot3d import Axes3D
                        plt.rcParams['figure.figsize'] = (10, 7)
                        plt.style.use('ggplot')
                        

                        fig = plt.figure()
                        ax = Axes3D(fig)
                        ax.scatter(MEstandarizada[:, 0], 
                                MEstandarizada[:, 1], 
                                MEstandarizada[:, 2], marker='o', s=60)
                        ax.scatter(MParticional.cluster_centers_[:, 0], 
                                MParticional.cluster_centers_[:, 1], 
                                MParticional.cluster_centers_[:, 2], marker='o', s=1000)
                        plt.savefig('ClusterGP.png')
                        st.pyplot()
                        image = Image.open('ClusterGP.png')
                        new_image = image.resize((500, 500))
                        new_image.save('ClusterGP.png')
                    except:
                            st.warning("Selecciona un número válido de clústeres")

                    export_as_pdf = st.button("Export Report")
                    item2 = []
                    if export_as_pdf:
                        pdf = FPDF()
                        pdf.add_page()
                        pdf.set_font('Arial', 'B', 12)
                        pdf.multi_cell(0, 5, 'Reporte de Resultados' + '\n')
                        pdf.image('SmarthChicoMenuReportes.png', x = None, y = None, w = 0, h = 0, type = '', link = '')
                        for i in range(kl.elbow):
                            string =''
                            string += "El clúster número "
                            string += str(i)
                            string += " tiene "
                            string += str(numClusters[i])
                            string += " elementos."

                            pdf.multi_cell(0, 5, string + '\n')
                        
                        for n in range(kl.elbow):
                            string1 = ''
                            string1 += "Clúster "
                            string1 += str(n)
                            string1 += "Conformado por: "
                            string1 += str(numClusters[n])
                            string1 += " elementos"
                            pdf.multi_cell(0, 5, string1 + '\n')
                            for m in range(CentroidesP.columns.size):
                                string2 = ''
                                string2 += " Con "
                                string2 += str(CentroidesP.columns[m])
                                string2 += " promedio de: "
                                string2 += str(CentroidesP.iloc[n,m].round(5))

                                pdf.multi_cell(0, 5, string2 + '\n')
                        pdf.multi_cell(0, 5, conclusion + '\n')
                            
                        pdf.add_page()
                        pdf.multi_cell(0, 5, 'Grafico Utilizado' + '\n')
                        pdf.multi_cell(0, 5, 'Grafico dispersión' + '\n')
                        pdf.image('GraficoDispersion.png', x = None, y = None, w = 0, h = 0, type = '', link = '')
                        pdf.add_page()
                        pdf.multi_cell(0, 5, 'MapadeCalor' + '\n')
                        pdf.image('MapadeCalor.png', x = None, y = None, w = 0, h = 0, type = '', link = '')
                        pdf.add_page()
                        pdf.multi_cell(0, 5, 'Elbow' + '\n')
                        pdf.image('ElbowG.png', x = None, y = None, w = 0, h = 0, type = '', link = '')
                        pdf.add_page()
                        pdf.multi_cell(0, 5, 'Cluster en el Codo' + '\n')
                        pdf.image('ElbowGR.png', x = None, y = None, w = 0, h = 0, type = '', link = '')
                        pdf.add_page()
                        pdf.multi_cell(0, 5, 'Grafica Elementos Agrupados' + '\n')
                        pdf.image('ClusterGP.png', x = None, y = None, w = 0, h = 0, type = '', link = '')
                        html = create_download_link(pdf.output(dest="S").encode("latin-1"), "testfile")
                        st.markdown(html, unsafe_allow_html=True)


                elif MatrizClusteringP.size == 0:
                    st.warning("No se ha seleccionado ninguna variable...")