# coding: utf-8

import pandas as pd

ccaa_iso = {
    'AN': 'Andalucía',
    'AR': 'Aragón',
    'AS': 'Principado de Asturias',
    'CN': 'Canarias',
    'CB': 'Cantabria',
    'CM': 'Castilla-La Mancha',
    'CL': 'Castilla y León',
    'CT': 'Cataluña',
    'EX': 'Extremadura',
    'GA': 'Galicia',
    'IB': 'Islas Baleares',
    'RI': 'La Rioja (Spain)',
    'MD': 'Comunidad de Madrid',
    'MC': 'Región de Murcia',
    'NC': 'Comunidad Foral de Navarra',
    'PV': 'País Vasco',
    'VC': 'Comunidad Valenciana',
    'CE': 'Ceuta',
    'ML': 'Melilla'
}

df = pd.read_csv("datos_ccaas.csv")

# Change column name
df.columns = ['Comunidad Autónoma', 'Fecha',
              'Casos confirmados', 'Casos con prueba PCR',
              'Casos con test Anticuerpos',
              'Casos con otras pruebas',
              'Sin información de pruebas']

# Change CCAA id
df['Comunidad Autónoma'] = df['Comunidad Autónoma'].map(ccaa_iso)
df.to_csv('datos_procesados.csv', index=False)

# Check that there are no duplicated or missing values
print(df.isna().any())  # false
print(df.duplicated().any())  # false
