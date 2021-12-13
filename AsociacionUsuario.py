def run_AsociacionU():

        import streamlit as st               # Para crear vectores y matrices n dimensionales
        import matplotlib.pyplot as plt     # Para la generación de gráficas a partir de los datos
        from apyori import apriori         # Para la implementación de reglas de asociación
        from st_aggrid import AgGrid
        from st_aggrid.grid_options_builder import GridOptionsBuilder
        from numpy.lib.shape_base import split
        import streamlit as st
        import pandas as pd                 # Para la manipulación y análisis de los datos
        import matplotlib.pyplot as plt     # Para la generación de gráficas a partir de los datos
        import numpy as np                  # Para crear vectores y matrices n dimensionales
        from apyori import apriori         # Para la implementación de reglas de asociación
        from PIL import Image
        from fpdf import FPDF
        import base64
        from io import BytesIO
        import sqlite3 
        conn = sqlite3.connect('data.db')
        c = conn.cursor()

        def create_valuetableA():
            c.execute('CREATE TABLE IF NOT EXISTS valoresAsociacion(idA INT, soporte REAL, confianza REAL, lift REAL)')


        def get_soporteA(idA):
            
     
            return c.execute('SELECT soporte FROM valoresAsociacion WHERE idA =?',(idA,)).fetchone()[0]


        def get_confianzaA(idA):
            return c.execute('SELECT confianza FROM valoresAsociacion WHERE idA =?',(idA,)).fetchone()[0]



        def get_liftA(idA):
            return c.execute('SELECT lift FROM valoresAsociacion WHERE idA =?',(idA,)).fetchone()[0]
           

        def view_alldata():
            c.execute('SELECT * FROM valoresAsociacion')
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

        

        st.set_option('deprecation.showPyplotGlobalUse', False) # Para evitar que se muestre el warning de matplotlib

        st.title("Módulo: Reglas de asociación")
        texto = '<p style="font-family:monospace;color:Black; font-size: 20px;"> Algoritmo apriori</p>'
        st.markdown(texto, unsafe_allow_html=True)
        st.markdown("""- Las reglas de asociación es un algoritmo de aprendizaje automático basado en reglas, que se utiliza para encontrar relaciones ocultas en los datos.""")
        st.markdown("""- Se originó con el estudio de transacciones de clientes para determinar asociaciones entre los artículos comprados. También se conoce como análisis de afinidad. """)
        datosAsociacion = st.file_uploader("Selecciona un archivo válido para trabajar las reglas de asociación", type=["csv", "txt"])
        
        


        if datosAsociacion is not None:
            datosRAsociacion = pd.read_csv(datosAsociacion, header=None)
            subtexto2 = '<p style="font-family:monospace;color:Black; font-size: 18px;"> Configuraciones: </p>'
            st.markdown(subtexto2, unsafe_allow_html=True)
            with st.expander("Observar Matriz de Transacciones y Graficos: "):
                Transacciones = datosRAsociacion.values.reshape(-1).tolist() #-1 significa 'dimensión desconocida' (recomendable) o: 7460*20=149200
                ListaM = pd.DataFrame(Transacciones)
                ListaM['Frecuencia'] = 0 #Valor temporal
                #Se agrupa los elementos
                ListaM = ListaM.groupby(by=[0], as_index=False).count().sort_values(by=['Frecuencia'], ascending=True) #Conteo
                ListaM['Porcentaje'] = (ListaM['Frecuencia'] / ListaM['Frecuencia'].sum()) #Porcentaje
                ListaM = ListaM.rename(columns={0 : 'Item'})
                

                gb = GridOptionsBuilder.from_dataframe(ListaM)
                gb.configure_pagination()
                gb.configure_side_bar()
                gb.configure_default_column(groupable=True, value=True, enableRowGroup=True, aggFunc="sum", editable=True)
                gridOptions = gb.build()

                subtexto = '<p style="font-family:monospace;color:Black; font-size: 20px;">Matriz de Transacciones: </p>'
                st.markdown(subtexto, unsafe_allow_html=True)
                st.subheader("Elementos de los menos populares a los más populares:")
                AgGrid(ListaM, gridOptions=gridOptions, enable_enterprise_modules=True, height= 500, fit_columns_on_grid_load=True)

                st.subheader("De manera gráfica: ")
                with st.spinner("Generando gráfica..."):
                    # Se muestra el gráfico de las películas más populares a las menos populares
                    grafica = plt.figure(figsize=(20,30))
                    plt.xlabel('Frecuencia')
                    plt.ylabel('Elementos')
                    plt.barh(ListaM['Item'], ListaM['Frecuencia'],color='blue')
                    plt.title('Elementos de los menos populares a los más populares')
                    st.pyplot(grafica)

             

              

            with st.expander("Obtención de Reportes y Reglas: "):
                data_result = view_alldata()
                clean_db = pd.DataFrame(data_result,columns=["idA", "soporte","confianza", "lift"])
                gb = GridOptionsBuilder.from_dataframe(clean_db)

                gb.configure_pagination()
                gb.configure_side_bar()
                gb.configure_default_column(groupable=True, value=True, enableRowGroup=True, aggFunc="sum", editable=True)
                gridOptions = gb.build()

                st.subheader('Configuraciones Realizadas por el Analista:')
                AgGrid(clean_db, gridOptions=gridOptions, enable_enterprise_modules=True, height= 200, fit_columns_on_grid_load=True)
                MoviesLista = datosRAsociacion.stack().groupby(level=0).apply(list).tolist()

                st.subheader("Ingresa el id de la Configuración que deseas: ")
              
                idA =  st.number_input("idConfiguracion", min_value=1.0, value=1.0, step=1.0)
                create_valuetableA()
                min_support = get_soporteA(idA)
                min_confidence = get_confianzaA(idA)
                min_lift = get_liftA(idA)


            with st.expander("Reglas de Asociación Obtenidas: "):
                
                ReglasC1 = apriori(MoviesLista, min_support=min_support, min_confidence=min_confidence, min_lift=min_lift)
                Resultado = list(ReglasC1)
                st.success("Reglas de asociación encontradas: "+ str(len(Resultado)))                    
                if len(Resultado) == 0: 
                    st.warning("No se encontraron reglas de asociación")
                else:
                    text = []
                    st.subheader('Reglas de Asociacion Encontradas')
                    for item in Resultado:
                        mystring = ''
                        mystring += 'Regla '
                        mystring += (str(Resultado.index(item)+1) + '. ')
                        Emparejar = item[0]
                        items = [x for x in Emparejar]
                        mystring += ("("+str(", ".join(item[0]))+")")
                        mystring += (' [Soporte: ' + str(round(item[1] * 100,2))+ " % ")
                        mystring += ('Confianza: ' + str(round(item[2][0][2]*100,2))+ " % ") 
                        mystring += ('Lift: ' + str(round(item[2][0][3],2)) + ']')
                        st.info(mystring)
                             
                        
                    for item in Resultado:
                        text.append(str(Resultado.index(item)+1) + ". ")
                        text.append("Regla: ("+str(", ".join(item[0]))+") ")
                        text.append("Soporte: ("+str(round(item[1] * 100,2))+ " %) ")
                        text.append("Confianza: ("+str(round(item[2][0][2]*100,2))+ " %) ")
                        text.append("Lift: ("+str(round(item[2][0][3],2))+ " %) ") 

                    Transacciones = datosRAsociacion.values.reshape(-1).tolist() #-1 significa 'dimensión desconocida' (recomendable) o: 7460*20=149200
                    ListaM = pd.DataFrame(Transacciones)
                    ListaM['Frecuencia'] = 0 #Valor temporal
                    #Se agrupa los elementos
                    ListaM = ListaM.groupby(by=[0], as_index=False).count().sort_values(by=['Frecuencia'], ascending=True) #Conteo
                    ListaM['Porcentaje'] = (ListaM['Frecuencia'] / ListaM['Frecuencia'].sum()) #Porcentaje
                    ListaM = ListaM.rename(columns={0 : 'Item'})
                    grafica = plt.figure(figsize=(20,30))
                    plt.title("Iris Dataset Averages by Plant Type")
                    plt.xlabel("Measurement Name")
                    plt.ylabel("Centimeters (cm)")
                    plt.barh(ListaM['Item'], ListaM['Frecuencia'],color='blue')
                    plt.title('Elementos de los menos populares a los más populares')
                    plt.savefig('GraficoAsociacion.png')

                    image = Image.open('GraficoAsociacion.png')
                    new_image = image.resize((800, 800))
                    new_image.save('GraficoAsociacion.png')

                    # Concluir las reglas de asociación
                conclusions = st.text_area("En este espacio, se pueden anotar las conclusiones a las que se llegaron a partir de los resultados obtenidos en las reglas de asociación:", "")
                st.subheader(conclusions)

                    



                export_as_pdf = st.button("Export Report")
                item2 = []
                if export_as_pdf:
                    pdf = FPDF()
                    pdf.add_page()
                    pdf.set_font('Arial', 'B', 12)
                    x = 0
                    n = 0
                    mystring = ''
                    for item in text:
                        if x < 5:

                            mystring += item
                            x += 1
                            print(item)
                        else:
                            n += 1
                            print(mystring)
                            item2.append(mystring)
                            mystring = ''
                            mystring += item
                            x = 1
                    if((len(Resultado)-1) == n):
                            item2.append(mystring)
                    pdf.multi_cell(0, 5, 'Reporte de Resultados' + '\n')
                    pdf.image('SmarthChicoMenuReportes.png', x = None, y = None, w = 0, h = 0, type = '', link = '')
                    pdf.multi_cell(0, 5, conclusions + '\n')
                    for item in item2:
                        pdf.multi_cell(0, 5, item + '\n')
                    
                    pdf.add_page()
                    pdf.multi_cell(0, 5, 'Graficos Utilizados' + '\n')
                    pdf.image('GraficoAsociacion.png', x = None, y = None, w = 0, h = 0, type = '', link = '')
                    html = create_download_link(pdf.output(dest="S").encode("latin-1"), "testfile")
                    st.markdown(html, unsafe_allow_html=True)
                    image = Image.open('GraficoAsociacion.png')
                    rgb_im = image.convert('RGB')
                    rgb_im.save('GraficoAsociacion.jpg')
                    image = Image.open('GraficoAsociacion.jpg')

                    st.markdown(get_image_download_link(image,'graph1,jpg','Download '+'.jpg'), unsafe_allow_html=True)






