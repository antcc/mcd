library(gplots)
library(qgraph)
library(tidyverse)

# Leemos el dataframe desde el recurso web facilitado
df <- read.csv("http://cardsorting.net/tutorials/25.csv")

# Seleccionamos aquellas variables que nos interesan descartando el resto de ellas
df = subset(df, select = -c(Uniqid, Startdate, Starttime, Endtime, QID, Comment))

# Sin tener en cuenta la categoría, creamos un dataframe auxiliar únicamente con
# los valores de las diferentes tarjetas
nums = sapply(subset(df, select = -(Category)), as.numeric)

# Pintamos un histograma donde mostramos el tipo de valores númericos que nos podemos entoncontrar en las columnas correspondientes a las tarjetas
hist(nums, xlab = "Numerical value", col = "cadetblue4", main = "Histogram of unique numerical values")

# Calculamos las distancias entre las columnas haciendo uso de la función dist.
# Notamos que esta calcula la distancia a las filas del dataframe, por lo que lo
# trasponemos para obtener lo que queríamos.
dist = dist(t(nums), method="euclidean")

# Hacemos un mapa de calor para mostrar las distancias entre las tarjetas.
heatmap.2(as.matrix(dist), symkey=FALSE, density.info="none",
          trace="none", dendrogram = "row")

# Pintamos en un grafo las relaciones entre las tarjetas
qgraph(1/dist, layout='spring', vsize=3)

# Obtenemos las tarjetas más relacionadas entre sí
cat("Cards with higher similarity:\n")
min_dist <- min(dist)
which(as.matrix(dist)==min_dist, arr.ind=TRUE)

# Mostramos los histogramas de algunas tarjetas parecidas
show_categories <- function(df, c1, c2) {
  # Agrupamos las columnas c1 y c2 por categoría
  grouped_df = df[(df[,c1]!=0 | df[,c2]!=0), c("Category", c1, c2)] %>%
    group_by(Category) %>%
    summarise_all(sum)

  # Mostramos el histograma de categorías para c1 y c2, agrupado
  p <- grouped_df %>%
    gather(Card, Frequency, -Category) %>%
    ggplot(aes(x=Category, y=Frequency, fill = Card)) +
      geom_col(position = "dodge")

  p + theme(axis.text.x = element_text(angle = 90))
}

show_categories(df, "Apple", "Banana")
show_categories(df, "Pie", "Cake")
