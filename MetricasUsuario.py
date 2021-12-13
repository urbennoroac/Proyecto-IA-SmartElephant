def run_MetricasU():

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

            

            
            with st.expander("Euclidiana"):
                st.subheader("Distancia Euclidiana")

                DstEuclidiana = cdist(datosMetricasMetricas, datosMetricasMetricas, metric='euclidean') # Calcula TODA la matriz de distancias 
                matrizEuclidiana = pd.DataFrame(DstEuclidiana)
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
                    image = Image.open('matrizEuclidiana.png')
                    new_image = image.resize((500, 500))
                    new_image.save('matrizEuclidiana.png')

                csv_downloader(de)        
            
            with st.expander("Chebyshev"):
                st.subheader("Distancia Chebyshev")
               
                DstChebyshev = cdist(datosMetricasMetricas, datosMetricasMetricas, metric='chebyshev') # Calcula TODA la matriz de distancias
                matrizChebyshev = pd.DataFrame(DstChebyshev)                    
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
                    image = Image.open('matrizChebyshev.png')
                    new_image = image.resize((500, 500))
                    new_image.save('matrizChebyshev.png')

                csv_downloader(dc)


            with st.expander("Manhattan"):
                st.subheader("Distancia de Manhattan")
                DstManhattan = cdist(datosMetricasMetricas, datosMetricasMetricas, metric='cityblock') # Calcula TODA la matriz de distancias
                matrizManhattan = pd.DataFrame(DstManhattan)
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
                    image = Image.open('matrizManhattan.png')
                    new_image = image.resize((500, 500))
                    new_image.save('matrizManhattan.png')
                csv_downloader(dma)


            with st.expander("Minkowski"):
                st.subheader("Distancia de Minkowski")
                DstMinkowski = cdist(datosMetricasMetricas, datosMetricasMetricas, metric='minkowski',p=1.5) # Calcula TODA la matriz de distancias
                matrizMinkowski = pd.DataFrame(DstMinkowski)
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
                    image = Image.open('matrizMinkowski.png')
                    new_image = image.resize((500, 500))
                    new_image.save('matrizMinkowski.png')
                csv_downloader(dmi)
                

            with st.expander('Reportes'):
                conclusions = st.text_area("En este espacio, se pueden anotar las conclusiones a las que se llegaron a partir de los resultados obtenidos de las distancias:", key = "4")
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
                    text1 = """Distancia Euclidiana (euclídea, por Euclides) es una de las métricas más utilizadas para calcular la distancia entre dos puntos, conocida también como espacio euclidiano.Sus bases se encuentran en la aplicación del Teorema de Pitágoras, 
                    donde la distancia viene a ser la longitud de la hipotenusa."""
                    pdf.multi_cell(0, 5, text1 + '\n')
                    pdf.multi_cell(0, 5, 'Grafica de distancia Matriz Euclidiana:' + '\n')
                    pdf.image('matrizEuclidiana.png', x = None, y = None, w = 0, h = 0, type = '', link = '')

                    pdf.add_page()
                    text2 = """La distancia de Chebyshev es el valor máximo absoluto de las diferencias entre las coordenadas de un par de elementos. 
                    Lleva el nombre del matemático ruso Pafnuty Chebyshev, conocido por su trabajo en la geometría analítica y teoría de números.
                    Otro nombre para la distancia de Chebyshev es métrica máxima."""
                    pdf.multi_cell(0, 5, text2 + '\n')
                    pdf.multi_cell(0, 5, 'Grafica de distancia Matriz Minkowski:' + '\n')
                    pdf.image('matrizManhattan.png', x = None, y = None, w = 0, h = 0, type = '', link = '')

                    pdf.add_page()
                    text3 = """La distancia euclidiana es una buena métrica. Sin embargo, en la vida real, por ejemplo en una ciudad, es imposible moverse de un punto a otro de manera recta.
                    Se utiliza la distancia de Manhattan si se necesita calcular la distancia entre dos puntos en una ruta similar a una cuadrícula (información geoespacial).
                    Se llama Manhattan debido al diseño de cuadrícula de la mayoría de las calles dela isla de Manhattan, Nueva York (USA)."""
                    pdf.multi_cell(0, 5, text3 + '\n')
                    pdf.multi_cell(0, 5, 'Grafica de distancia Matriz Manhattan:' + '\n')
                    pdf.image('matrizManhattan.png', x = None, y = None, w = 0, h = 0, type = '', link = '')

                    pdf.add_page()
                    text4 = """La distancia de Minkowski es una distancia entre dos puntos en un espacio n-dimensional. 
                    Es una métrica de distancia generalizada: Euclidiana, Manhattan y Chebyshev."""
                    pdf.multi_cell(0, 5, text4 + '\n')
                    pdf.multi_cell(0, 5, 'Grafica de distancia Matriz Minkowski:' + '\n')
                    pdf.image('matrizManhattan.png', x = None, y = None, w = 0, h = 0, type = '', link = '')

                    html = create_download_link(pdf.output(dest="S").encode("latin-1"), "testfile")
                    st.markdown(html, unsafe_allow_html=True)