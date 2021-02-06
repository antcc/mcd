#!/usr/bin/env python3

import pandas as pd

#
# GLOBAL DATA
#

# Leemos el dataset
df = pd.read_csv("uam-personal-pdi-2018-anonimizado_0.csv")
# Eliminamos la columna con el mensaje informativo de anonimización
df = df.drop(columns=["IMPORTANTE"])

#
# FUNCTIONS
#

def freq(query, search = None, df = df):
    """Devuelve todas las instancias resultado de una consulta, opcionalmente
       junto a la probabilidad de que un campo concreto tome una serie de
       valores."""

    q = df.query(query)
    if search is not None:
        q = q[search].value_counts(dropna = False, normalize = True)

    if q.empty:
        print("[]")
    else:
        print(q)

#
# VARIABLES PIVOTE
#

# Utilizando el área de conocimiento obtenemos el código de unidad responsable
print("Códigos de unidad responsable únicos para aquellas entradas cuya área de conocimiento es Lenguajes y Sistemas Informáticos:")
print(
    df.query(
        'des_area_conocimiento == "Lenguajes y Sistemas Informáticos"'
    )["cod_unidad_responsable"].unique()
)

cur_pivote = 55001545
standard_query = \
    f'cod_unidad_responsable == "{cur_pivote}" & cod_genero == "H"'

# Demostramos que el área de conocimiento no puede ser NaN
show_not_nan = False
if show_not_nan:
    print("Códigos de unidad responsable posibles para área de conocimiento NaN:")
    print(
        df.query(
            'des_area_conocimiento != des_area_conocimiento'
        )["cod_unidad_responsable"].unique()
    )

    cur = 55001539
    print(f"Entradas con número de sexenios y quinquenios adecuados (o vacíos) para el código de unidad responsable {cur}:")
    print(
        df.query(
            f'num_sexenios=="4" & num_quinquenios == "3" & cod_unidad_responsable == "{cur}"'))
    print(
        df.query(
            f'num_sexenios=="4" & num_quinquenios != num_quinquenios & cod_unidad_responsable == "{cur}"'))
    print(
        df.query(
            f'num_sexenios != num_sexenios & num_quinquenios == "3" & cod_unidad_responsable == "{cur}"'))
    print(
        df.query(
            f'num_sexenios != num_sexenios & num_quinquenios != num_quinquenios & cod_unidad_responsable == "{cur}"'))

    print("Entradas con número de sexenios y quinquenios adecuados (o vacíos) para el código de unidad responsable NaN:")
    print(
        df.query(
            'num_sexenios=="4" & num_quinquenios == "3" & cod_unidad_responsable != cod_unidad_responsable'))
    print(
        df.query(
            'num_sexenios=="4" & num_quinquenios != num_quinquenios & cod_unidad_responsable != cod_unidad_responsable'))
    print(
        df.query(
            'num_sexenios != num_sexenios & num_quinquenios == "3" & cod_unidad_responsable != cod_unidad_responsable'))
    print(
        df.query(
            'num_sexenios != num_sexenios & num_quinquenios != num_quinquenios & cod_unidad_responsable != cod_unidad_responsable'))

#
# BLOQUES DE COHERENCIA
#

print(" ==== BLOQUE 1 ====")
print("Universidades incluidas en el dataset:")
print(df['des_universidad'].unique())
print("Años incluidos en el dataset:")
print(df['anio'].unique())

print(" ==== BLOQUE 2 ====")
freq(standard_query, "des_pais_nacionalidad")
freq(standard_query, "des_continente_nacionalidad")
freq(standard_query, "des_agregacion_paises_nacionalidad")

print(" ==== BLOQUE 3 ==== ")
freq(standard_query, "des_comunidad_residencia")
freq(standard_query, "des_provincia_residencia")
freq(standard_query, "des_municipio_residencia")

print(" ==== BLOQUE 4 =====")
freq(standard_query, "anio_nacimiento")

print(" ==== BLOQUE 5 =====")
# Comprobamos que no hay NaN
#freq(standard_query, "cod_categoria_cuerpo_escala")
query5 = standard_query + ' & cod_categoria_cuerpo_escala == "5"'
freq(query5, "des_tipo_personal")
freq(query5, "des_categoria_cuerpo_escala")
freq(query5, "des_tipo_contrato")
freq(query5, "des_dedicacion")
freq(query5, "num_horas_semanales_tiempo_parcial")
freq(query5, "des_situacion_administrativa")

print(" ==== BLOQUE 6 =====")
freq(standard_query, "ind_cargo_remunerado")

print(" ==== BLOQUE 7 =====")
print("Personas que cumplen los requisitos de leer su tesis y recibir el título de doctorado en el año 2000 en la UAM (o bien tener alguno de estos tres campos a NaN), desglosadas por género:")
print(
    df.query(
        'anio_lectura_tesis == "2000.0"\
        & anio_expedicion_titulo_doctor == "2000.0"\
        & des_universidad_doctorado == "Universidad Autónoma de Madrid"\
        & cod_unidad_responsable == "55001545.0"\
        '
    )["des_genero"])
print(
    df.query(
        'anio_lectura_tesis != anio_lectura_tesis\
         & anio_expedicion_titulo_doctor == "2000.0"\
         & des_universidad_doctorado == "Universidad Autónoma de Madrid"\
         & cod_unidad_responsable == "55001545.0"\
         '
    )["des_genero"])
print(
    df.query(
        'anio_lectura_tesis == "2000.0"\
        & anio_expedicion_titulo_doctor != anio_expedicion_titulo_doctor\
        & des_universidad_doctorado == "Universidad Autónoma de Madrid"\
        & cod_unidad_responsable == "55001545.0"\
        '
    )["des_genero"])
print(
    df.query(
        'anio_lectura_tesis == "2000.0"\
        & anio_expedicion_titulo_doctor == "2000.0"\
        & des_universidad_doctorado != des_universidad_doctorado\
        & cod_unidad_responsable == "55001545.0"\
        '
    )["des_genero"])
print(
    df.query(
        'anio_lectura_tesis != anio_lectura_tesis\
        & anio_expedicion_titulo_doctor != anio_expedicion_titulo_doctor\
        & des_universidad_doctorado == "Universidad Autónoma de Madrid"\
        & cod_unidad_responsable == "55001545.0"\
        '
    )["des_genero"])
print(
    df.query(
        'anio_lectura_tesis != anio_lectura_tesis\
        & anio_expedicion_titulo_doctor == "2000.0"\
        & des_universidad_doctorado != des_universidad_doctorado\
        & cod_unidad_responsable == "55001545.0"\
        '
    )["des_genero"])
print(
    df.query(
        'anio_lectura_tesis == "2000.0"\
        & anio_expedicion_titulo_doctor != anio_expedicion_titulo_doctor\
        & des_universidad_doctorado != des_universidad_doctorado\
        & cod_unidad_responsable == "55001545.0"\
        '
    )["des_genero"])
print(
    df.query(
        'anio_lectura_tesis != anio_lectura_tesis\
        & anio_expedicion_titulo_doctor != anio_expedicion_titulo_doctor\
        & des_universidad_doctorado != des_universidad_doctorado\
        & cod_unidad_responsable == "55001545.0"\
        '
    )["des_genero"])

# Resultan los siguientes índices que sean hombres
b7_i = [
    220, 227, 249, 218, 222, 223, 228, 233, 235, 238, 250, 253, 255, 256,
    262
]
b7 = df.iloc[b7_i]

# Utilizamos que todos los individuos de b7 tienen un valor válido en
# "des_titulo_doctorado", y que sabemos que tiene 1
print("Valores únicos para des_titulo_doctorado entre los seleccionados:")
print(b7["des_titulo_doctorado"].unique())

query7 = 'des_titulo_doctorado == "Uno" & ' + standard_query
print("Información del bloque 7:")
freq(query7, "des_pais_doctorado", df = b7)
freq(query7, "des_continente_doctorado", df = b7)
freq(query7, "des_agregacion_paises_doctorado", df = b7)
freq(query7, "des_mencion_europea", df=b7)

print("====  BLOQUE 8 ==== ")
# Utilizando su area de conocimiento
freq('des_area_conocimiento == "Lenguajes y Sistemas Informáticos"', "des_tipo_unidad_responsable")

print("==== BLOQUE 9 =====")
# Utilizamos la informacion de sus sexenion y quinquenios, comprobando valores NaN
query9 = 'num_sexenios=="4" & num_quinquenios == "3" & cod_unidad_responsable == "55001545.0"'
print("Personas que cumplen requisitos de sexenios y quinquenios, con posibles NaN:")
freq(query9)
freq('num_sexenios != num_sexenios & num_quinquenios == "3" & cod_unidad_responsable == "55001545.0"')
freq('num_sexenios=="4" & num_quinquenios != num_quinquenios & cod_unidad_responsable == "55001545.0"')
freq('num_sexenios != num_sexenios & num_quinquenios != num_quinquenios & cod_unidad_responsable == "55001545.0"')
# Solo hay una persona que cumple los requisitos

# Mostramos datos
print("Información bloque 9:")
freq(query9, "anio_incorporacion_ap")
freq(query9, "anio_incorpora_cuerpo_docente")
freq(query9, "num_trienios")

print(" ==== BLOQUE 10 =====")
freq(standard_query, "num_tesis")

print(" ==== BLOQUE 11 =====")
freq(standard_query, "ind_investigador_principal")
