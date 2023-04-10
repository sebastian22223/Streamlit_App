import streamlit as st
from math import *
import numpy as np
#import matplotlib.pyplot as plt
import math
import sympy
import struct
import pandas as pd
from prettytable import PrettyTable
# import matplotlib.pyplot as plt


st.set_page_config(page_title="Calculadoras", page_icon=":pencil:", layout="wide")
st.sidebar.image("/Users/Juan Sebastian R/Desktop/stream/images/mmu.png", use_column_width=True)
menu = ["Simpson 1/3", "Expresiones", "Derivadas", "Cambio de Bases","Falsa Posicion","Biseccion","Metodo de la Secante","Newton Rapson","Graficadora","IEEE","Simpson 3/8","","","",""]
choice = st.sidebar.selectbox("Seleccione una opción", menu)



if choice == "Simpson 1/3":
    
            
        
    def evaluacion(x):
        copia = list(funcion)
        for j in range(len(copia)):
            if copia[j] == "x":
                copia[j] = str(x)
        return eval("".join(copia))


    def simps_method(funcion, a, b, n):
        h = (b - a) / n
        total = 0

        for i in range(1, n):
            x = a + (i * h)
            if (i % 2 == 0):
                total += 2 * evaluacion(x)
            else:
                total += 4 * evaluacion(x)

        total += evaluacion(a) + evaluacion(b)
        total = total * ((1 / 3) * h)

        return total


    st.title("Método de Integración Numérica: Simpson 1/3")
    st.write("Ingrese los datos para la integral/función:")

    funcion = st.text_input("Función")
    a = st.number_input("Intervalo inferior", value=0.0)
    b = st.number_input("Intervalo superior", value=0.0)
    n = st.number_input("Valor de n", value=1, step=1)

    if st.button("Calcular"):
        resultado = simps_method(funcion, a, b, n)
        st.write(f"Resultado de la aproximación: {resultado}")
        
        # Definimos un rango de valores para x
        x = np.linspace(a, b, num=100)
        # Evaluamos la función en cada punto del rango de valores de x
        y = [evaluacion(i) for i in x]
        
        # Creamos la tabla
        tabla = []
        tabla.append(['Subintervalo', 'Puntos evaluados', 'f(x)', 'Aproximación de la integral'])
        h = (b - a) / n
        for i in range(n):
            xi = a + (i * h)
            xf = a + ((i + 1) * h)
            puntos = np.linspace(xi, xf, num=2)
            fx = [evaluacion(j) for j in puntos]
            aproximacion = simps_method(funcion, xi, xf, 2)
            tabla.append([f'{i+1}', f'{puntos}', f'{fx}', f'{aproximacion}'])
            
        st.table(tabla)

        # Creamos la gráfica
        plt.plot(x, y)
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title('Gráfica de la función')
        # Incluimos la función en formato LaTeX en la etiqueta de la gráfica
        plt.text((a+b)/2, max(y), f'$f(x)={funcion}$', ha='center', va='top')
        st.pyplot(plt)
        
   

elif choice == "Expresiones":
        

        
        funciones = {
        "sqrt": math.sqrt,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "arctan": math.atan
        }

        def evaluar_expresion(expresion):
            for funcion in funciones:
                expresion = expresion.replace(funcion, f"funciones['{funcion}']")

            try:
                resultado = eval(expresion)
                st.write(f"El resultado es: {resultado}")
            except Exception as e:
                st.write("Error:", e)
        

        st.title("Calculadora de expresiones matemáticas")
        expresion = st.text_input("Ingresa una expresión matemática:")

        if st.button("Calcular"):
             if expresion:
                evaluar_expresion(expresion)
    

elif choice == "Derivadas":
    def calcular_derivadas(funcion, variable):
        # Convertir la ecuación en una expresión SymPy
        x = sympy.symbols(variable)
        f = sympy.sympify(funcion)

        # Calcular la primera derivada
        df = sympy.diff(f, x)

        # Calcular la segunda derivada
        ddf = sympy.diff(df, x)

        # Devolver las derivadas como expresiones SymPy
        respuesta = [df,ddf]

        return respuesta

    # Configurar la página de Streamlit
    st.title("Calculadora de derivadas")
    expresion = st.text_input("Ingresa una función:")
    variable = st.text_input("Ingresa la variable de la función:", "x")

    # Calcular las derivadas cuando el usuario hace clic en el botón
    if st.button("Calcular derivadas"):
        df, ddf = calcular_derivadas(expresion, variable)
        st.write("Primera derivada:", df)
        st.write("Segunda derivada:", ddf)

elif choice == "Cambio de Bases":

    def float_to_bin(num):
        return bin(struct.unpack('!I', struct.pack('!f', num))[0])[2:].zfill(32)

    def float_to_hex(f):
        return hex(struct.unpack('<Q', struct.pack('<d', f))[0])[2:]

    def float_octal(number, places=3):
        whole, dec = str(number).split(".")
        whole = int(whole)
        dec = int(dec)
        res = oct(whole).lstrip("0o") + "."
        for x in range(places):
            whole, dec = str((decimal_converter(dec)) * 8).split(".")
            dec = int(dec)
            res += whole
        return res

    def decimal_converter(num):
        while num > 1:
            num /= 10
        return num

    def calcular_numeros():
        try:
            numero = float(numero_input)
            binario = float_to_bin(numero)
            p = 10
            octal = float_octal(numero, places=p)
            hexa = float_to_hex(numero)
            st.write(f"Binario: {binario}")
            st.write(f"Octal: {octal}")
            st.write(f"Hexadecimal: {hexa}")
        except ValueError:
            st.write("Ingrese un número válido.")

    st.title("Conversión de números")
    numero_input = st.text_input("Ingrese un número:")
    calcular_button = st.button("Calcular")

    if calcular_button:
        calcular_numeros()

elif choice == "Falsa Posicion":
    
    def Falsa_p(funcion, xa, xb, itera=100, error_r=0.001):
        sol = None
        cont = 0
        error_C = 101
        iteraciones = []
        sol_anterior = xa
        
        if funcion(xa) * funcion(xb) <= 0:
            # calcula la solucion
            while cont <=itera and error_C >= error_r:
                cont +=1
                sol = xb - ((funcion(xb) * (xb - xa)) / (funcion(xb) - funcion(xa)))
                error_C = abs((sol - sol_anterior) / sol) * 100 
                
                # guarda la informacion de la iteracion actual
                iteracion_actual = {
                    'iteracion': cont,
                    'xa': xa,
                    'xb': xb,
                    'sol': sol,
                    'error_C': error_C
                }
                iteraciones.append(iteracion_actual)
                
                if funcion(xa) * funcion(sol) >= 0:
                    xa = sol
                else:
                    xb = sol
                    
                sol_anterior = sol
                
            raiz = str('{:.11f}'.format(sol))
            error_calculado = str('{:.3f}'.format(error_C) + '%')
            
            respuestas1 = [raiz, error_calculado]
            
            return respuestas1, iteraciones
        else:
            print('no existe solución en ese intervalo')

    st.title("Calculadora de Falsa Posición")
    
    funcion = st.text_input("Ingrese la función a evaluar", "sin(x)+2*x-3*x/cos(x)")
    xa = st.number_input("Ingrese el valor de a", -10.0, 10.0, -10.0)
    xb = st.number_input("Ingrese el valor de b", -10.0, 10.0, 10.0)
    itera = st.number_input("Ingrese el número máximo de iteraciones", 1, 10000, 100)
    error_r = st.number_input("Ingrese el error máximo", 0.00001, 1.0, 0.001)

    if st.button("Calcular"):
        respuestas, iteraciones = Falsa_p(lambda x: eval(funcion), xa, xb, itera, error_r)
        st.write(f"La raíz encontrada es: {respuestas[0]}")
        st.write(f"El error relativo es: {respuestas[1]}")

        # Convertimos las iteraciones en un DataFrame de pandas para mostrarlo en una tabla
        df_iteraciones = pd.DataFrame(iteraciones)

        # Graficamos la evolución del error relativo
        fig, ax = plt.subplots()
        ax.plot(df_iteraciones["iteracion"], df_iteraciones["error_C"])
        ax.set_xlabel("Iteración")
        ax.set_ylabel("Error relativo (%)")
        st.pyplot(fig)

        # Mostramos las iteraciones en una tabla
        st.write("Iteraciones:")
        st.table(df_iteraciones[["iteracion", "xa", "xb", "sol", "error_C"]])


# elif choice == "Biseccion":

elif choice == "Metodo de la Secante":

    
    def calcular_secante(x0, x1, n, f, ndigits):
        def f_obj(x):
            return eval(f)

        tabla = PrettyTable()
        tabla.field_names = ["Iteración", "Xn-1", "Xn", "Xn+1", "F(Xn+1)", "Error"]

        x_data = []
        y_data = []

        for i in range(n):
            try:
                x2 = x1 - f_obj(x1) * (x1 - x0) / (f_obj(x1) - f_obj(x0))
            except ZeroDivisionError:
                st.warning("Se ha producido una división por cero en la iteración {}. La ejecución ha sido detenida.".format(i+1))
                return

            error = abs(x2 - x1)

            tabla.add_row([i+1, round(x0, ndigits), round(x1, ndigits), round(x2, ndigits), round(f_obj(x2), ndigits), round(error, ndigits)])

            x_data.append(x2)
            y_data.append(f_obj(x2))

            x0 = x1
            x1 = x2

        st.write(tabla)
        st.pyplot(generar_grafico(f, x_data, y_data))

    def generar_grafico(f, x_data, y_data):
        fig, ax = plt.subplots()
        ax.plot(x_data, y_data)
        ax.set_xlabel("x")
        ax.set_ylabel("f(x)")
        ax.set_title("Gráfico de la función objetivo")
        return fig

    st.title("Calculadora del Método de la Secante")

    x0 = st.number_input("Valor Inicial 1")
    x1 = st.number_input("Valor Inicial 2")
    n = st.number_input("Número de iteraciones", step=1, min_value=1, max_value=100000)
    f = st.text_input("Función Objetivo")
    ndigits = st.number_input("Número de intervalos después de la coma", step=1, min_value=0, max_value=10)

    if st.button("Calcular"):
        calcular_secante(x0, x1, n, f, ndigits)

elif choice == "Newton Rapson":


    def eval_function(fun_text, xi):
        """
        Evalúa una función matemática en un punto dado.

        Parameters
        ----------
        fun_text : str
            La función como una cadena de texto.
        xi : float
            El punto en el que se evalúa la función.

        Returns
        -------
        float
            El resultado de evaluar la función en el punto dado.
        """
        x = sympy.Symbol('x')
        fun = sympy.sympify(fun_text)
        return float(fun.subs(x, xi))

    def newton(fun_text, x_n, epsilon, steps):
        """
        Calcula la raíz de una función utilizando el método de Newton-Raphson.

        Parameters
        ----------
        fun_text : str
            La función como una cadena de texto.
        x_n : float
            El punto inicial de la iteración.
        epsilon : float
            La tolerancia del método.
        steps : int
            El número máximo de iteraciones permitidas.

        Returns
        -------
        list of dict
            Una lista de diccionarios, cada uno con información sobre una iteración del método.
        """
        x = sympy.Symbol('x')
        fun = sympy.sympify(fun_text)
        fder = sympy.diff(fun, x)
        results = []
        i = 1
        while i <= steps:
            f_xn = eval_function(fun_text, x_n)
            fder_xn = eval_function(str(fder), x_n)
            x_n1 = x_n - f_xn / fder_xn
            abs_error = abs(x_n1 - x_n)
            rel_error = abs(abs_error / x_n1)
            results.append({
                "iteration": i,
                "approx_root": x_n1,
                "F(xi)": f_xn,
                "f'(xi)": fder_xn,
                "absolute_error": abs_error,
                "relative_error": rel_error
            })
            if abs_error < epsilon:
                break
            x_n = x_n1
            i += 1
        return results

    def main():
        st.title("Método de Newton-Raphson")
        fun_text = st.text_input("Ingrese la función:")
        x_n = st.number_input("Ingrese el punto inicial de la iteración:", value=0.0)
        epsilon = st.number_input("Ingrese la tolerancia del método:", value=1e-6)
        steps = st.number_input("Ingrese el número máximo de iteraciones permitidas:", value=50, step=1)
        if st.button("Calcular"):
            results = newton(fun_text, x_n, epsilon, steps)
            if results:
                st.write(f"La raíz aproximada es {results[-1]['approx_root']}")

    main()

elif choice == "Graficadora":

        st.title("Graficador de Funciones")

        # Función para reemplazar funciones matemáticas por sus equivalentes en numpy
        def reemplaza_funciones(funcion):
            funciones_matematicas = {
                "sin": "np.sin",
                "cos": "np.cos",
                "tan": "np.tan",
                "sqrt": "np.sqrt",
                "exp": "np.exp",
                "log": "np.log",
                "pi": "np.pi",
                "arcsin": "np.arcsin",
                "arccos": "np.arccos",
                "arctan": "np.arctan"
            }
            for f, npf in funciones_matematicas.items():
                funcion = funcion.replace(f, npf)
            return funcion

        # Widgets para ingresar la función y los límites de la variable independiente
        funcion = st.text_input("Ingrese una función:", "np.sin(x)")
        limite_inferior = st.number_input("Ingrese el límite inferior:", value=-5.0, step=0.1)
        limite_superior = st.number_input("Ingrese el límite superior:", value=5.0, step=0.1)

        # Botón para calcular y graficar la función
        if st.button("Calcular"):
            # Evaluar la función
            x = np.linspace(limite_inferior, limite_superior, 1000)
            y = eval(reemplaza_funciones(funcion))
            
            # Graficar la función
            fig, ax = plt.subplots()
            ax.plot(x, y)
            ax.axhline(0, color="gray", lw=0.5)
            ax.axvline(0, color="gray", lw=0.5)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_title("Gráfica de la función f(x) = {}".format(funcion))
            st.pyplot(fig)


# elif choice == "IEEE":






# elif choice == "Simpson 3/8":

