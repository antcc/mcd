DESCRIPCIÓN DEL DATASET

Se proporcionan las puntuaciones de los estudiantes en dos formatos:

a) Un archivo all-ratings.dat, con todos los datos por columnas.

b) Un archivo all-matrix.dat, con todos los datos en forma matricial.

Además, los ficheros all-ratings-no0.dat y all-matrix-no0.dat contienen los mismos datos usando el formato correspondiente pero eliminando los ceros.

Para la práctica se usarán las particiones de entrenamiento y test que se pueden encontrar en training-ratings.dat (training-matrix.dat) y test-ratings.dat (test-matrix.dat) obtenidos a partir de all-ratings-no0.dat (all-matrix-no0.dat) separando los datos de cada alumno en 80% training y 20% test respectivamente.

OBSERVACIONES

1.- Los usuarios se han identificado usando los ids que aparecen en la hoja de cálculo original (para que podáis revisar manualmente las recomendaciones que os genera).

2.- En los ficheros *matrix*.dat los usuarios se organizan en columnas y las películas en filas, mostrando en la casilla correspondiente de la matriz la puntuación dada por el usuario a esa película. La primera línea incluye el id del usuario correspondiente a cada columna.

3.- El formato de los ficheros *ratings*.dat es: ID de usuario, ID de película, puntuación.

4.- En los dos casos, el separador de columnas que se usa es un tabulador ('\t').

Si se detecta cualquier error, avisad a alejandro.bellogin@uam.es
