
def run_AsociacionProgramador():    
    import streamlit as st
    import pandas as pd                 # Para la manipulación y análisis de los datos
    import numpy as np                  # Para crear vectores y matrices n dimensionales
    import matplotlib.pyplot as plt     # Para la generación de gráficas a partir de los datos
    from apyori import apriori  
    from streamlit_metrics import metric, metric_row
    from streamlit_ace import st_ace       # Para la implementación de reglas de asociación
    import altair as alt
    import cufflinks as cf





    THEMES = [
        "ambiance", "chaos", "chrome", "clouds", "clouds_midnight", "cobalt", "crimson_editor", "dawn",
        "dracula", "dreamweaver", "eclipse", "github", "gob", "gruvbox", "idle_fingers", "iplastic",
        "katzenmilch", "kr_theme", "kuroir", "merbivore", "merbivore_soft", "mono_industrial", "monokai",
        "nord_dark", "pastel_on_dark", "solarized_dark", "solarized_light", "sqlserver", "terminal",
        "textmate", "tomorrow", "tomorrow_night", "tomorrow_night_blue", "tomorrow_night_bright",
        "tomorrow_night_eighties", "twilight", "vibrant_ink", "xcode"
    ]

    KEYBINDINGS = ["emacs", "sublime", "vim", "vscode"]

    display, editor = st.columns((2, 2))

    INITIAL_CODE = """
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


def add_valuetableA(idA, soporte, confianza, lift):
    c.execute('INSERT INTO valoresAsociacion(idA, soporte, confianza, lift) VALUES (?,?,?,?)',(idA, soporte, confianza, lift))

    conn.commit()

def delete_valuetableA():
    c.execute('DELETE FROM valoresAsociacion')

    conn.commit()


st.set_option('deprecation.showPyplotGlobalUse', False) # Para evitar que se muestre el warning de matplotlib

st.title("Módulo: Reglas de asociación")
texto = '<p style="font-family:monospace;color:Black; font-size: 20px;"> Algoritmo apriori</p>'
st.markdown(texto, unsafe_allow_html=True)

datosAsociacion = st.file_uploader("Selecciona un archivo válido para trabajar las reglas de asociación", type=["csv", "txt"])




if datosAsociacion is not None:
    datosRAsociacion = pd.read_csv(datosAsociacion, header=None)
    subtexto2 = '<p style="font-family:monospace;color:Black; font-size: 18px;"> Configuraciones: </p>'
    st.markdown(subtexto2, unsafe_allow_html=True)
    opcionVisualizacionAsociacion = st.select_slider(' ', options=["Visualizar Datos", "Procesar Datos","Implementar Algoritmo"])
    if opcionVisualizacionAsociacion == "Visualizar Datos":
        st.header("Datos cargados: ")
        st.dataframe(datosRAsociacion)
        

    if opcionVisualizacionAsociacion == "Procesar Datos":
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

        

        #Se muestra la lista de las películas menos populares a las más populares
    

        st.subheader("De manera gráfica: ")



        with st.spinner("Generando gráfica..."):
            # Se muestra el gráfico de las películas más populares a las menos populares
            grafica = plt.figure(figsize=(20,30))
            plt.xlabel('Frecuencia')
            plt.ylabel('Elementos')
            plt.barh(ListaM['Item'], ListaM['Frecuencia'],color='blue')
            plt.title('Elementos de los menos populares a los más populares')
            st.pyplot(grafica)

     

      

    if opcionVisualizacionAsociacion == "Implementar Algoritmo":
        #Se crea una lista de listas a partir del dataframe y se remueven los 'NaN'
        #level=0 especifica desde el primer índice hasta el último
        MoviesLista = datosRAsociacion.stack().groupby(level=0).apply(list).tolist()

        st.subheader("Ingresa los datos para configurar el algoritmo: ")
        colu1, colu2, colu3, colu4= st.columns(4)
        min_support =  colu1.number_input("Mínimo de Soporte", min_value=0.0, value=0.01, step=0.01)
        min_confidence = colu2.number_input("Mínimo de Confianza", min_value=0.0, value=0.3, step=0.01)
        min_lift = colu3.number_input("Mínimo de Lift", min_value=0.0, value=2.0, step=1.0)
        idA = colu4.number_input("id Configuración", min_value=1.0, value=1.0, step=1.0)
        


        with st.expander("Obtención de Reportes y Reglas: "):
            create_valuetableA()
            delete_valuetableA()
            add_valuetableA(idA, min_support, min_confidence, min_lift)
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

                        
    """

    with editor:
        st.write('### Code editor')
        code = st_ace(
            value=INITIAL_CODE,
            language="python",
            placeholder="st.header('Hello world!')",
            theme=st.sidebar.selectbox("Theme", options=THEMES, index=26),
            keybinding=st.sidebar.selectbox("Keybinding mode", options=KEYBINDINGS, index=3),
            font_size=st.sidebar.slider("Font size", 5, 24, 14),
            tab_size=st.sidebar.slider("Tab size", 1, 8, 4),
            wrap=st.sidebar.checkbox("Wrap lines", value=False),
            show_gutter=True,
            show_print_margin=True,
            auto_update=False,
            readonly=False,
            key="ace-editor"
        )
        st.write('Hit `CTRL+ENTER` to refresh')
        st.write('*Remember to save your code separately!*')

    with display:
        exec(code)

    with st.sidebar:
        libraries_available = st.expander('Available Libraries')
        with libraries_available:
            st.write("""
            * Pandas (pd)
            * Numpy (np)
            * Altair (alt)
            * Bokeh
            * Plotly
            * Cufflinks (cf)
            [Need something else?](https://github.com/samdobson/streamlit-sandbox/issues/new)
            """)






