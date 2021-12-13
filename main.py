
import streamlit as st
from PIL import Image

img = Image.open('SmarthChico.png')
st.set_page_config(page_title = 'SmartElephant', page_icon = img, layout = "wide")



padding = 0
padding2 = 4
st.markdown(f""" <style>
    .reportview-container .main .block-container{{
        padding-top: {padding}rem;
        padding-right: {padding2}rem;
        padding-left: {padding2}rem;
        padding-bottom: {padding}rem;
    }} </style> """, unsafe_allow_html=True)

import pandas as pd
import datetime
import hashlib
import numpy as np      
import base64
import pandas as pd
import matplotlib.pyplot as plt
import extra_streamlit_components as stx  
import altair as alt
import cufflinks as cf 
import base64 
import sqlite3 
import time
timestr = time.strftime("%Y%m%d-%H%M%S")

from apyori import apriori  
from st_aggrid import AgGrid
from st_aggrid.grid_options_builder import GridOptionsBuilder

from streamlit_metrics import metric, metric_row
from streamlit_ace import st_ace
from PIL import Image
from fpdf import FPDF

from io import BytesIO
from numpy.lib.shape_base import split


#Llamada a otros metodos
from AsociacionAnalista import run_Asociacion
from AsociacionUsuario import run_AsociacionU
from AsociacionProgramador import run_AsociacionProgramador
from MetricasAnalista import run_Metricas
from MetricasUsuario import run_MetricasU
from ClusteringAnalista import run_Cluster
from ClasificacionAnalista import run_Clasif
from ClasificacionUsuario import run_ClasifU
from ArbolesDAnalista import run_Arboles
from ArbolesDUsuario import run_ArbolesU




main_bg = "Background4.png"
main_bg_ext = "png"

side_bg = "background.png"
side_bg_ext = "png"

st.markdown(
    f"""
    <style>
    .reportview-container {{
        background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()})
    }}
   .sidebar .sidebar-content {{
        background: url(data:image/{side_bg_ext};base64,{base64.b64encode(open(side_bg, "rb").read()).decode()})
    }}
    </style>
    """,
    unsafe_allow_html=True
)



st.markdown(""" <style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style> """, unsafe_allow_html=True)


import hashlib
def make_hashes(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

def check_hashes(password,hashed_text):
    if make_hashes(password) == hashed_text:
        return hashed_text
    return False
# DB Management
import sqlite3 
conn = sqlite3.connect('data.db')
c = conn.cursor()
# DB  Functions
def create_usertable():
    c.execute('CREATE TABLE IF NOT EXISTS usertable(names TEXT, username TEXT,password TEXT, tipodeusuario TEXT, clavedeusuario TEXT)')


def add_userdata(name, username,password, tipo, clave):
    c.execute('INSERT INTO usertable(names, username,password, tipodeusuario, clavedeusuario) VALUES (?,?,?,?,?)',(name, username,password, tipo, clave))

    conn.commit()

def delete_userdata(username):
    c.execute('DELETE FROM usertable WHERE username = ?',(username,))
    conn.commit()

def check_user(username):
    c.execute('SELECT username FROM usertable WHERE username =?',(username,))
    data = c.fetchall()
    return data

def login_user(username,password, tipo):
    c.execute('SELECT * FROM usertable WHERE (username =? AND password = ?) AND tipodeusuario = ?',(username,password, tipo))
    data = c.fetchall()
    return data

def view_all_users():
    c.execute('SELECT * FROM usertable')
    data = c.fetchall()
    return data

st.set_option('deprecation.showPyplotGlobalUse', False) # Para evitar que se muestre el warning de matplotlib

def main():

    if 'counter' not in st.session_state:
        st.session_state.counter = 0
    if 'counter2' not in st.session_state:
        st.session_state.counter2 = 0.2
    image = Image.open('SmarthChicoMenu.png')
    col1, col2, col3 = st.columns([1,6,1])

    with col1:
        st.write("")

    with col2:
        st.image(image)

    with col3:
        st.write("")


    if st.session_state.counter == 0:
        st.sidebar.error("Inicia Sesión")
    elif st.session_state.counter == 1:
        st.sidebar.info("Iniciaste Sesión (Admin)")
    elif st.session_state.counter == 2:
        st.sidebar.info("Iniciaste Sesión (Analista)")
    elif st.session_state.counter == 3:
        st.sidebar.info("Iniciaste Sesión (Programador)")
    elif st.session_state.counter == 4:
        st.sidebar.info("Iniciaste Sesión (Usuario)")
    else:
        st.write('Hola')

    st.sidebar.title("Smarth Elephant V.1.0")
   	

    menu = ["Intro", "Login", "Sign Up", "Logout", "Usuarios", "Reglas de Asociación", "Métricas de Distancia", "Clustering", "Clasificación", "Arboles de Decision"]
    choice = st.sidebar.selectbox("Menu", menu)



    st.sidebar.header("Opciones Disponibles")
    if choice == "Intro":
        image = Image.open('IntroDescripcion.png')
        col1, col2, col3 = st.columns([1,6,1])

        with col1:
            st.write("")

        with col2:
            st.image(image)

        with col3:
            st.write("")

    elif choice == "Sign Up":
        formulario = st.form(key='formulario', clear_on_submit = True)
        name = formulario.text_input("Nombre")
        username = formulario.text_input("Username")
        password = formulario.text_input("Password", type = 'password')
        tipodeusuario = formulario.selectbox("Usuario", ["Admin", "Analista", "Programador", "Usuario"])
        clavedeusuario = formulario.text_input("Clave de registro")
        subtim_button2 = formulario.form_submit_button("Sign Up")
        if subtim_button2:
            create_usertable()
            usernamev = check_user(username)
            if usernamev:
                st.error("El usuario ya existe")
            else:
                if ((tipodeusuario == 'Admin') and (clavedeusuario == '123')):
                    create_usertable()
                    add_userdata(name, username, make_hashes(password), tipodeusuario, clavedeusuario)
                    st.success("Usuario Registrado")
                    st.info("Regresa al Menu para relaizar un Login")
                elif ((tipodeusuario == 'Analista')  and (clavedeusuario == '234')):
                    create_usertable()
                    add_userdata(name, username, make_hashes(password), tipodeusuario, clavedeusuario)
                    st.success("Usuario Registrado")
                    st.info("Regresa al Menu para relaizar un Login")
                elif ((tipodeusuario == 'Programador')  and (clavedeusuario == '345')):
                    create_usertable()
                    add_userdata(name, username, make_hashes(password), tipodeusuario, clavedeusuario)
                    st.success("Usuario Registrado")
                    st.info("Regresa al Menu para relaizar un Login")
                elif ((tipodeusuario == 'Usuario')  and (clavedeusuario == '211')):
                    create_usertable()
                    add_userdata(name, username, make_hashes(password), tipodeusuario, clavedeusuario)
                    st.success("Usuario Registrado")
                    st.info("Regresa al Menu para relaizar un Login")
                else:
                    st.error("Clave de Registro Incorrecta")

    elif choice == "Login":
        formulario = st.form(key='formulario', clear_on_submit = True)
        username = formulario.text_input("Username")
        password = formulario.text_input("Password", type = 'password')
        tipodeusuario = formulario.selectbox("Usuario", ["Admin", "Analista", "Programador", "Usuario"])
        subtim_button = formulario.form_submit_button("Login")
        if subtim_button:
            create_usertable()
            hashed_pswd = make_hashes(password)
            result = login_user(username,check_hashes(password,hashed_pswd), tipodeusuario)
            if result:
                if (tipodeusuario == 'Admin'):
                    st.success("Logged In as {}".format(username))
                    st.session_state.counter = 1
                if (tipodeusuario == 'Analista'):
                    st.success("Logged In as {}".format(username))
                    st.session_state.counter = 2
                if (tipodeusuario == 'Programador'):
                    st.success("Logged In as {}".format(username))
                    st.session_state.counter = 3
                if (tipodeusuario == 'Usuario'):
                    st.success("Logged In as {}".format(username))
                    st.session_state.counter = 4
            else:
                st.error("Usuario o Contraseña Incorrecta")

    elif choice == "Usuarios":
        if st.session_state.counter == 1: 
            new_title = '<p style="font-family:monospace;color:Black; font-size: 42px;">Usuarios</p>'
            st.markdown(new_title, unsafe_allow_html=True)
            st.info("Usuarios disponibles en la base de datos")
            user_result = view_all_users()
            clean_db = pd.DataFrame(user_result,columns=["names", "username","password", "tipodeusuario", "clavedeusuario"])
            gb = GridOptionsBuilder.from_dataframe(clean_db)

            gb.configure_pagination()
            gb.configure_side_bar()
            gb.configure_default_column(groupable=True, value=True, enableRowGroup=True, aggFunc="sum", editable=True)
            gridOptions = gb.build()

            AgGrid(clean_db, gridOptions=gridOptions, enable_enterprise_modules=True, height= 300,)

            new_title = '<p style="font-family:monospace;color:Black; font-size: 38px;">Elimina Usuarios</p>'
            st.markdown(new_title, unsafe_allow_html=True)
            eliminado = st.text_input("Ingrese el nombre de usuario")
            if st.button("Eliminar"):
                usernamev = check_user(eliminado)
                if usernamev:
                    create_usertable()
                    delete_userdata(eliminado)
                    st.success("Usuario Registrado")
                else:
                    st.error("El usuario no existe")
        else: 
            st.error("Ingrese con un Login como Administrador")

    elif choice == "Clustering":
        if st.session_state.counter == 1:
            run_Cluster()
        elif st.session_state.counter == 2:
            run_Cluster()
        elif st.session_state.counter == 3:
            st.info('Aun no se habilita un perfil para Programador (Espera a la version 2.0')
        elif st.session_state.counter == 4:
            run_Cluster()
        else:
            st.error('Inicia Sesión con una Cuenta ')  

    elif choice == "Logout":
        if st.button("LogOut"):
            st.session_state.counter = 0
            st.success("Usted ha salido de la sesión")
    elif choice == "Métricas de Distancia":
        if st.session_state.counter == 1:
            run_Metricas()
        elif st.session_state.counter == 2:
            run_Metricas()
        elif st.session_state.counter == 3:
            st.info('Aun no se habilita un perfil para Programador (Espera a la version 2.0')
        elif st.session_state.counter == 4:
            run_MetricasU()
        else:
            st.error('Inicia Sesión con una Cuenta ')    		

    elif choice == "Reglas de Asociación":
        if st.session_state.counter == 1:
            run_Asociacion()
        elif st.session_state.counter == 2:
            run_Asociacion()
        elif st.session_state.counter == 3:
            run_AsociacionProgramador()
        elif st.session_state.counter == 4:
            run_AsociacionU()
        else:
            st.error('Inicia Sesión con una Cuenta ')


    elif choice == "Clasificación":
        if st.session_state.counter == 1:
            st.session_state.counter2 = run_Clasif()
        elif st.session_state.counter == 2:
            st.session_state.counter2 = run_Clasif()
        elif st.session_state.counter == 3:
            st.info('Aun no se habilita un perfil para Programador (Espera a la version 2.0')
        elif st.session_state.counter == 4:
            run_ClasifU(st.session_state.counter2)
        else:
            st.error('Inicia Sesión con una Cuenta') 

    elif choice == "Arboles de Decision":
        if st.session_state.counter == 1:
            run_Arboles()
        elif st.session_state.counter == 2:
            run_Arboles()
        elif st.session_state.counter == 3:
            st.info('Aun no se habilita un perfil para Programador (Espera a la version 2.0')
        elif st.session_state.counter == 4:
            run_ArbolesU()
        else:
            st.error('Inicia Sesión con una Cuenta') 

              
    else:
        st.write('Adios')


if __name__ == '__main__':
    main()
