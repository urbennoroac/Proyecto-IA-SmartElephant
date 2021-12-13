def run_Metricas():

                   # Para crear vectores y matrices n dimensionales
        import streamlit as st    
        from PIL import Image
        import matplotlib.pyplot as plt     # Para la generación de gráficas a partir de los datos
        from apyori import apriori         # Para la implementación de reglas de asociación
        from st_aggrid import AgGrid
        from st_aggrid.grid_options_builder import GridOptionsBuilder
        from numpy.lib.shape_base import split
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


        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.title("Módulo: Metricas de distancia")
        texto = '<p style="font-family:monospace;color:Black; font-size: 20px;"> Euclidiana, Chebyshev, Manhattan, Minkowski</p>'
        st.markdown(texto, unsafe_allow_html=True)
        st.write("""
        Muchos de estos algoritmos utilizan medidas de distancia, las cuales son importantes (más de lo que se imagina), para identificar objetos (elementos) con características similares y no similares (disímiles).
        Estas medidas de distancia, conocidas también como búsqueda de similitud vectorial, juegan unpapel importante en el aprendizaje automático.
        """)
        datosMetricas = st.file_uploader("Selecciona un archivo válido para trabajar con las Métricas de Distancia:", type=["csv","txt"])
        
        if datosMetricas is not None:
            datosMetricasMetricas = pd.read_csv(datosMetricas) 
            mt = pd.DataFrame(datosMetricasMetricas)
            gb = GridOptionsBuilder.from_dataframe(mt)
            gb.configure_pagination()
            gb.configure_side_bar()
            gb.configure_default_column(groupable=True, value=True, enableRowGroup=True, aggFunc="sum", editable=True)
            gridOptions = gb.build()
            subtexto = '<p style="font-family:monospace;color:Black; font-size: 20px;">Datos Subidos: </p>'
            st.markdown(subtexto, unsafe_allow_html=True)
            AgGrid(mt, gridOptions=gridOptions, enable_enterprise_modules=True, height= 500, fit_columns_on_grid_load=True)

            

            VisualizacionMetricas = st.select_slider('Selecciona la metrica de distancia que desea utilizar: ', options=["Euclidiana", "Chebyshev","Manhattan","Minkowski"])
            if VisualizacionMetricas == "Euclidiana":
                st.subheader("Distancia Euclidiana")

                DstEuclidiana = cdist(datosMetricasMetricas, datosMetricasMetricas, metric='euclidean') # Calcula TODA la matriz de distancias 
                matrizEuclidiana = pd.DataFrame(DstEuclidiana)
                with st.expander('Matriz de distancias Euclidiana de todos los datos:'):
                    subtexto = '<p style="font-family:monospace;color:Black; font-size: 20px;">Matriz Euclidiana: </p>'
                    st.markdown(subtexto, unsafe_allow_html=True)
                    st.dataframe(matrizEuclidiana)
                    de = pd.DataFrame(matrizEuclidiana)

                    st.subheader("Gráfico de distancias de la Matriz Euclidiana: ")
                    with st.spinner('Espere mientras cargamos los datos graficamente ..'):
                        plt.figure(figsize=(5,5))
                        plt.imshow(matrizEuclidiana, cmap='Blues')
                        plt.colorbar()
                        plt.savefig('matrizEuclidiana.png')
                        st.pyplot()

                    csv_downloader(de)        
                
                with st.expander('Distancia Euclidiana entre dos objetos'):
                    st.subheader("Selecciona dos datos para calcular la distancia entre ellos: ")
                    with st.spinner('Cargando distancia Euclidiana entre dos datos...'):
                        #Calculando la distancia entre dos objetos 
                        columna1, columna2 = st.columns([1,3])
                        with columna1:
                            objeto1 = st.selectbox('Dato 1: ', options=matrizEuclidiana.columns)
                            objeto2 = st.selectbox('Dato 2: ', options=matrizEuclidiana.columns)
                            distanciaEuclidiana = distance.euclidean(datosMetricasMetricas.iloc[objeto1], datosMetricasMetricas.iloc[objeto2])
                            st.success("La distancia entre los dos datos seleccionados es de: "+str(distanciaEuclidiana))
                        with columna2:
                            plt.figure(figsize=(9,9))
                            plt.grid(True)
                            plt.title("Distancia Euclidiana entre los dos datos seleccionados")
                            plt.scatter(distanciaEuclidiana, distanciaEuclidiana, c='red',edgecolors='black')
                            plt.xlabel('Distancia del punto '+str(objeto1)+' al punto '+str(objeto2))
                            plt.ylabel('Distancia del punto '+str(objeto2)+' al punto '+str(objeto1))
                            plt.annotate('  '+str(distanciaEuclidiana.round(2)), xy=(distanciaEuclidiana, distanciaEuclidiana), arrowprops=dict(facecolor='red',headwidth=10, headlength=15), xytext=(distanciaEuclidiana, distanciaEuclidiana))
                            plt.savefig('GraficoEntreDosPuntosE.png')
                            st.pyplot()
                            image = Image.open('GraficoEntreDosPuntosE.png')
                            new_image = image.resize((500, 500))
                            new_image.save('GraficoEntreDosPuntosE.png')
                            

                with st.expander('Reportes'):
                    conclusions = st.text_area("En este espacio, se pueden anotar las conclusiones a las que se llegaron a partir de los resultados obtenidos en las Matriz Euclidiana:", key = "1")
                    st.subheader(conclusions)
                    export_as_pdf = st.button("Export Report", key = "1")
                    if export_as_pdf:
                        pdf = FPDF()
                        pdf.add_page()
                        pdf.set_font('Arial', 'B', 12)
                        pdf.multi_cell(0, 5, 'Reporte de Resultados' + '\n')
                        pdf.image('SmarthChicoMenuReportes.png', x = None, y = None, w = 0, h = 0, type = '', link = '')
                        pdf.multi_cell(0, 5, conclusions + '\n')
                        
                        pdf.add_page()
                        pdf.multi_cell(0, 5, 'Grafica de distancia Matriz Euclidiana:' + '\n')
                        pdf.image('matrizEuclidiana.png', x = None, y = None, w = 0, h = 0, type = '', link = '')
                        pdf.add_page()
                        pdf.multi_cell(0, 5, 'Graficos de distancia entre dos puntos' + '\n')
                        pdf.image('GraficoEntreDosPuntosE.png', x = None, y = None, w = 0, h = 0, type = '', link = '')
                        html = create_download_link(pdf.output(dest="S").encode("latin-1"), "testfile")
                        st.markdown(html, unsafe_allow_html=True)
                        



            
            if VisualizacionMetricas == "Chebyshev":
                st.subheader("Distancia Chebyshev")
               
                DstChebyshev = cdist(datosMetricasMetricas, datosMetricasMetricas, metric='chebyshev') # Calcula TODA la matriz de distancias
                matrizChebyshev = pd.DataFrame(DstChebyshev)
                with st.expander('Matriz de distancias Chebyshev de todos los datos: '):
                    
                    subtexto = '<p style="font-family:monospace;color:Black; font-size: 20px;">Matriz Chebyshev: </p>'
                    st.markdown(subtexto, unsafe_allow_html=True)
                    st.dataframe(matrizChebyshev)
                    dc = pd.DataFrame(matrizChebyshev)
                    st.subheader("Gráfico de distancias de la Matriz de Chebyshev: ")
                    with st.spinner('Cargando matriz de distancias Chebyshev...'):
                        plt.figure(figsize=(10,10))
                        plt.imshow(matrizChebyshev, cmap='Blues')
                        plt.colorbar()
                        plt.savefig('matrizChebyshev.png')
                        st.pyplot()

                    csv_downloader(dc)
                with st.expander('Distancia Chebyshev entre dos datos: '):
                    st.subheader("Selecciona dos datos para calcular la distancia entre ellos: ")
                    with st.spinner('Cargando distancia Chebyshev entre dos datos...'):
                        #Calculando la distancia entre dos objetos 
                        columna1, columna2 = st.columns([1,3])
                        with columna1:
                            objeto1 = st.selectbox('Dato 1: ', options=matrizChebyshev.columns)
                            objeto2 = st.selectbox('Dato 2: ', options=matrizChebyshev.columns)
                            distanciaChebyshev = distance.chebyshev(datosMetricasMetricas.iloc[objeto1], datosMetricasMetricas.iloc[objeto2])
                            st.success("La distancia entre los dos datos seleccionados es de: "+str(distanciaChebyshev))
                        with columna2:
                            plt.figure(figsize=(9,9))
                            plt.grid(True)
                            plt.title("Distancia Chebyshev entre los dos datos seleccionados")
                            plt.scatter(distanciaChebyshev, distanciaChebyshev, c='red',edgecolors='black')
                            plt.xlabel('Distancia del dato '+str(objeto1)+' al dato '+str(objeto2))
                            plt.ylabel('Distancia del dato '+str(objeto2)+' al dato '+str(objeto1))
                            plt.annotate('  '+str(distanciaChebyshev.round(2)), xy=(distanciaChebyshev, distanciaChebyshev), arrowprops=dict(facecolor='red',headwidth=10, headlength=15), xytext=(distanciaChebyshev, distanciaChebyshev))
                            plt.savefig('GraficoEntreDosPuntosC.png')
                            st.pyplot()
                            image = Image.open('GraficoEntreDosPuntosC.png')
                            new_image = image.resize((500, 500))
                            new_image.save('GraficoEntreDosPuntosC.png')


                with st.expander('Reportes'):
                    conclusions = st.text_area("En este espacio, se pueden anotar las conclusiones a las que se llegaron a partir de los resultados obtenidos en las Matriz Chebyshev:", key = "2")
                    st.subheader(conclusions)
                    export_as_pdf = st.button("Export Report", key = "2")
                    if export_as_pdf:
                        pdf = FPDF()
                        pdf.add_page()
                        pdf.set_font('Arial', 'B', 12)
                        pdf.multi_cell(0, 5, 'Reporte de Resultados' + '\n')
                        pdf.image('SmarthChicoMenuReportes.png', x = None, y = None, w = 0, h = 0, type = '', link = '')
                        pdf.multi_cell(0, 5, conclusions + '\n')
                        
                        pdf.add_page()
                        pdf.multi_cell(0, 5, 'Grafica de distancia Matriz Chebyshev:' + '\n')
                        pdf.image('matrizChebyshev.png', x = None, y = None, w = 0, h = 0, type = '', link = '')
                        pdf.add_page()
                        pdf.multi_cell(0, 5, 'Graficos de distancia entre dos puntos' + '\n')
                        pdf.image('GraficoEntreDosPuntosC.png', x = None, y = None, w = 0, h = 0, type = '', link = '')
                        html = create_download_link(pdf.output(dest="S").encode("latin-1"), "testfile")
                        st.markdown(html, unsafe_allow_html=True)

            if VisualizacionMetricas == "Manhattan":
                st.subheader("Distancia de Manhattan")
                DstManhattan = cdist(datosMetricasMetricas, datosMetricasMetricas, metric='cityblock') # Calcula TODA la matriz de distancias
                matrizManhattan = pd.DataFrame(DstManhattan)
                with st.expander('Matriz de distancias Manhattan de todos los datos:'):
                    subtexto = '<p style="font-family:monospace;color:Black; font-size: 20px;">Matriz Manhattan: </p>'
                    st.markdown(subtexto, unsafe_allow_html=True)
                    st.dataframe(matrizManhattan)
                    dma = pd.DataFrame(matrizManhattan)
                    st.subheader("Gráfico de distancias de la Matriz de Manhattan: ")
                    with st.spinner('Cargando matriz de distancias Manhattan...'):
                        plt.figure(figsize=(10,10))
                        plt.imshow(matrizManhattan, cmap='Blues')
                        plt.colorbar()
                        plt.savefig('matrizManhattan.png')
                        st.pyplot()
                    csv_downloader(dma)

                with st.expander('Distancia Manhattan entre dos datos'):
                    st.subheader("Selecciona dos datos para calcular la distancia entre ellos: ")
                    with st.spinner('Cargando distancia Manhattan entre dos datos..'):
                        #Calculando la distancia entre dos objetos 
                        columna1, columna2 = st.columns([1,3])
                        with columna1:
                            objeto1 = st.selectbox('Dato 1: ', options=matrizManhattan.columns)
                            objeto2 = st.selectbox('Dato 2: ', options=matrizManhattan.columns)
                            distanciaManhattan = distance.cityblock(datosMetricasMetricas.iloc[objeto1], datosMetricasMetricas.iloc[objeto2])
                            st.success("La distancia entre los dos datos seleccionados es de: "+str(distanciaManhattan))
                        with columna2:
                            plt.figure(figsize=(9,9))
                            plt.grid(True)
                            plt.title("Distancia Manhattan entre los dos datos seleccionados")
                            plt.scatter(distanciaManhattan, distanciaManhattan, c='red',edgecolors='black')
                            plt.xlabel('Distancia del datos '+str(objeto1)+' al datos '+str(objeto2))
                            plt.ylabel('Distancia del datos '+str(objeto2)+' al datos '+str(objeto1))
                            plt.annotate('  '+str(distanciaManhattan.round(2)), xy=(distanciaManhattan, distanciaManhattan), arrowprops=dict(facecolor='red',headwidth=10, headlength=15), xytext=(distanciaManhattan, distanciaManhattan))
                            plt.savefig('GraficoEntreDosPuntosMa.png')
                            st.pyplot()
                            image = Image.open('GraficoEntreDosPuntosMa.png')
                            new_image = image.resize((500, 500))
                            new_image.save('GraficoEntreDosPuntosMa.png')

                with st.expander('Reportes'):
                    conclusions = st.text_area("En este espacio, se pueden anotar las conclusiones a las que se llegaron a partir de los resultados obtenidos en las Matriz Manhattan:", key = "3")
                    st.subheader(conclusions)
                    export_as_pdf = st.button("Export Report", key = "3")
                    if export_as_pdf:
                        pdf = FPDF()
                        pdf.add_page()
                        pdf.set_font('Arial', 'B', 12)
                        pdf.multi_cell(0, 5, 'Reporte de Resultados' + '\n')
                        pdf.image('SmarthChicoMenuReportes.png', x = None, y = None, w = 0, h = 0, type = '', link = '')
                        pdf.multi_cell(0, 5, conclusions + '\n')
                        
                        pdf.add_page()
                        pdf.multi_cell(0, 5, 'Grafica de distancia Matriz Manhattan:' + '\n')
                        pdf.image('matrizManhattan.png', x = None, y = None, w = 0, h = 0, type = '', link = '')
                        pdf.add_page()
                        pdf.multi_cell(0, 5, 'Graficos de distancia entre dos puntos' + '\n')
                        pdf.image('GraficoEntreDosPuntosMa.png', x = None, y = None, w = 0, h = 0, type = '', link = '')
                        html = create_download_link(pdf.output(dest="S").encode("latin-1"), "testfile")
                        st.markdown(html, unsafe_allow_html=True)


            if VisualizacionMetricas == "Minkowski":
                st.subheader("Distancia de Minkowski")
                DstMinkowski = cdist(datosMetricasMetricas, datosMetricasMetricas, metric='minkowski',p=1.5) # Calcula TODA la matriz de distancias
                matrizMinkowski = pd.DataFrame(DstMinkowski)
                with st.expander('Matriz de distancias Minkowski de todos los datos:'):
                    subtexto = '<p style="font-family:monospace;color:Black; font-size: 20px;">Matriz Minkowski: </p>'
                    st.markdown(subtexto, unsafe_allow_html=True)
                    st.dataframe(matrizMinkowski)
                    dmi = pd.DataFrame(matrizMinkowski)
                    st.subheader("Gráfico de distancias de la Matriz de Minkowski: ")
                    with st.spinner('Cargando matriz de distancias Minkowski..'):
                        plt.figure(figsize=(10,10))
                        plt.imshow(matrizMinkowski, cmap='Blues')
                        plt.colorbar()
                        plt.savefig('matrizMinkowski.png')
                        st.pyplot()
                    csv_downloader(dmi)
                with st.expander('Distancia Minkowski entre dos objetos'):
                    st.subheader("Selecciona dos objetos para calcular la distancia entre ellos: ")
                    with st.spinner('Cargando distancia Minkowski entre dos datos...'):
                        #Calculando la distancia entre dos objetos 
                        columna1, columna2 = st.columns([1,3])
                        with columna1:
                            objeto1 = st.selectbox('Dato 1: ', options=matrizMinkowski.columns)
                            objeto2 = st.selectbox('Dato 2: ', options=matrizMinkowski.columns)
                            distanciaMinkowski = distance.minkowski(datosMetricasMetricas.iloc[objeto1], datosMetricasMetricas.iloc[objeto2], p=1.5)
                            st.success("La distancia entre los dos datos seleccionados es de: "+str(distanciaMinkowski))
                        with columna2:
                            plt.figure(figsize=(9,9))
                            plt.grid(True)
                            plt.title("Distancia Minkowski entre los dos datos seleccionados")
                            plt.scatter(distanciaMinkowski, distanciaMinkowski, c='red',edgecolors='black')
                            plt.xlabel('Distancia del dato '+str(objeto1)+' al dato '+str(objeto2))
                            plt.ylabel('Distancia del dato '+str(objeto2)+' al dato '+str(objeto1))
                            plt.annotate('  '+str(distanciaMinkowski.round(2)), xy=(distanciaMinkowski, distanciaMinkowski), arrowprops=dict(facecolor='red',headwidth=10, headlength=15), xytext=(distanciaMinkowski, distanciaMinkowski))
                            plt.savefig('GraficoEntreDosPuntosMi.png')
                            st.pyplot()
                            image = Image.open('GraficoEntreDosPuntosMi.png')
                            new_image = image.resize((500, 500))
                            new_image.save('GraficoEntreDosPuntosMi.png')

                with st.expander('Reportes'):
                    conclusions = st.text_area("En este espacio, se pueden anotar las conclusiones a las que se llegaron a partir de los resultados obtenidos en las Matriz Minkowski:", key = "4")
                    st.subheader(conclusions)
                    export_as_pdf = st.button("Export Report", key = "4")
                    if export_as_pdf:
                        pdf = FPDF()
                        pdf.add_page()
                        pdf.set_font('Arial', 'B', 12)
                        pdf.multi_cell(0, 5, 'Reporte de Resultados' + '\n')
                        pdf.image('SmarthChicoMenuReportes.png', x = None, y = None, w = 0, h = 0, type = '', link = '')
                        pdf.multi_cell(0, 5, conclusions + '\n')
                        
                        pdf.add_page()
                        pdf.multi_cell(0, 5, 'Grafica de distancia Matriz Minkowski:' + '\n')
                        pdf.image('matrizManhattan.png', x = None, y = None, w = 0, h = 0, type = '', link = '')
                        pdf.add_page()
                        pdf.multi_cell(0, 5, 'Graficos de distancia entre dos puntos' + '\n')
                        pdf.image('GraficoEntreDosPuntosMi.png', x = None, y = None, w = 0, h = 0, type = '', link = '')
                        html = create_download_link(pdf.output(dest="S").encode("latin-1"), "testfile")
                        st.markdown(html, unsafe_allow_html=True)