### limpiar ambiente     

rm(list=ls())

#### **Instalar/llamar las librerías de la clase**
require(pacman)
p_load(tidyverse,rio,glue,
       hexbin,
       patchwork,vip, ## plot: 
       ggrepel, ## plot: geom_text_repel
       stringi,tidytext,stopwords, ## text-data
       tidymodels,finetune) 
# set wd
setwd('C:/Users/Diego/OneDrive/Documents/GitHub/BD-ML---PS4/scripts/diego')

# cargar bases 
base_train <- read.csv('C:data/train.csv')
base_test <- read.csv('C:data/test.csv')


### LDA 
glimpse(base_train)

# hacer tabla para contar autores de tweets
table(base_train$name)

p_load("stopwords", "stringi", "tm", "rvest")

# hacer pre procesamiento de texto preparalo para LDA

# Creamos una lista con todos los stopwords en español
stopwords_español <- stopwords::stopwords("es", source = "snowball")
# Eliminamos los acentos de los stopwords
stopwords_español <- stri_trans_general(str = stopwords_español, id = "Latin-ASCII")

### 1. Normalización de tweets
# Eliminamos los acentos
base_train$text <- stri_trans_general(str = base_train$text, id = "Latin-ASCII")
# Ponemos el texto en minúscula
base_train$text <- tolower(base_train$text)

# Reemplazamos todos los caracteres no alfanumericos con un espacio
base_train$text <- str_replace_all(base_train$text, "[^[:alnum:]]", " ")
# Eliminamos los números
base_train$text <- gsub('[[:digit:]]+', '', base_train$text)
# Quitamos stopwords
base_train$text <- removeWords(base_train$text, stopwords_español)

# Eliminamos todos los espacios extras
base_train$text <- gsub("\\s+", " ", str_trim(base_train$text))

### 2. FUNCIÓN LEMMATIZACION
# crear una función para lematizar 
# crear funcion de lemmatization: quitar sufijos y prefijos
lematiza = function(frase){
  # Se reemplazan los espacios con +
  query <- gsub(" ", "+", frase)
  url_base <- "https://www.lenguaje.com/cgi-bin/lema.exe?edition_field="
  url_final <- paste0(url_base, query,"&B1=Lematizar")
  lemma <- read_html(url_final, encoding = "latin1") %>%
    html_nodes('div') %>% 
    tail(1) %>% 
    html_nodes("li") %>% 
    html_text2() %>% 
    tail(1)
  # lemma <- read_html(url_final, encoding = "latin1") %>% 
  #   html_node(css = "div div div div div li") %>% 
  #   html_text() 
  # lemma <- gsub("La palabra:", "", lemma)
  # lemma <- gsub("tiene los siguientes lemas:", "", lemma)
  # error <- "\r\n     Palabra no encontrada\r\n     Palabra no encontrada"
  # lemma <- ifelse(lemma == error, frase, lemma)
  if (length(lemma) == 0) {
    return(frase)
  } else {
    lemma <- str_split(lemma, "\\n")[[1]][1]
    return(lemma)
  }
}
# ejemplo de la funcion
lematiza('encontramos')


# Para evitar doble computación vamos a crear un diccionario de palabras con su respectiva lematización
# 3. TOKENIZAR TEXTO: Tokenizaremos tweets
p_load(tidytext)

# crear ID de cada tweet
base_train$id <- 1:nrow(base_train)

# generar tokens a partir de tweets (texto)
tidy_tweets <- base_train %>%
  unnest_tokens(output = token, input = text)


# Tenemos 13 mil palabras únicas. 
# Esto no lo vamos a correr en la complementaria porque se demora mucho. Vamos a importar el diccionario ya melo
#diccionario_lemmatizador <- data.frame(corpus = unique(tidy_ensayos$token))

#diccionario_lemmatizador <- diccionario_lemmatizador %>% 
  # Si se hacen demasiadas querys nos van a bloquear entonces toca
  # darle con calma
  # mutate(lemma = sapply(corpus, lematiza))

# 4. Generar modelo pre-entrenado
# Usando el comando automático de R con una aproximación más eficiente
p_load(udpipe)

# Creamos el id de cada documento
base_train$id <- paste0("doc", 1:nrow(base_train))

# Descargamos el modelo pre-entrenado
udmodel <- udpipe_download_model(language = "spanish")

# extraer el file_model del modelo
# modelo <- udpipe_load_model(file = udmodel$file_model)
file_model <- udmodel$file_model

# crear modelo udpipe
modelo <- udpipe_load_model(file = file_model)
# extraer texto
x <- udpipe_annotate(modelo, x = base_train$text)

# convertir a data.frame
tidy_tweets <- as.data.frame(x)

tidy_tweets[tidy_tweets$doc_id == "doc101", "lemma"]

### CONTAR PALABRAS

# contar palabras
# agrupar por doc_id (tweet)
# contar lemma ('infinitivo)
# ungroup

word_count <- tidy_tweets %>%
  group_by(doc_id) %>%
  count(lemma) %>%
  ungroup()

# Ahora vamos a eliminar las palabras extrañas (aquellas que aparezcan menos de 20 veces en los tweets) o demasiado comunes (que aparezcan en más del 50% de los tweets)

# contar palabras
# left_join
# contar (lemma)
# mutate: crear variable filtro ( n menor a 20 ó n mayor a )

num_tweets <- nrow(base_train)
word_count2 <- word_count %>%
  left_join(
    word_count %>%
      count(lemma) %>%
      mutate(filtro = !((n <= num_tweets*0.05) | (n >= num_tweets*0.5))) %>%
      select(-n)
  ) %>%
  filter(filtro) %>%
  select(-filtro)

# Se nos colaron stopwords entonces otra vez se eliminan 
# Normalizar los tweets otra vez

# Eliminamos los acentos
word_count2$lemma <- stri_trans_general(str = word_count2$lemma, id = "Latin-ASCII")

# Quitamos stopwords
# crear filtro de stopwords para quitar de word_count2
filtro <- !(word_count2$lemma %in% stopwords_español)
word_count2 <- word_count2[filtro, ]


#### crear matriz DTM
p_load(tm, tidytext)

# data es textos limpios word_count2
# document es ID
# term son las palabras
# value es n
tweets_dtm <- cast_dtm(data = word_count2, 
                        document = doc_id, 
                        term = lemma, 
                        value = n)
# 86% de la matriz es 0 
inspect(tweets_dtm)


### Visualizacion de las palabras más relevantes
p_load(wordcloud)

# calcular frecuencia de las palabras
freq <- sort(colSums(as.matrix(tweets_dtm)), 
             decreasing = TRUE)

dev.new(width = 1000, height = 1000, unit = "px")
wordcloud(names(freq), freq, max.words = 50,
          random.order = FALSE,
          colors = brewer.pal(8, "Accent"),
          scale = c(4, 0.5), rot.per = 0)



tweets_dtm2 <- cast_dtm(data = word_count2, 
                         document = doc_id, 
                         term = lemma, 
                         value = n,
                         weighting = tm::weightTfIdf)
inspect(tweets_dtm2)

# Visualicemos las palabras más relevantes
freq2 <- sort(colSums(as.matrix(tweets_dtm2)), 
              decreasing = TRUE)

dev.new(width = 1000, height = 1000, unit = "px")
wordcloud(names(freq2), freq2, max.words = 50,
          random.order = FALSE, min.freq = 0,
          colors = brewer.pal(8, "Accent"),
          scale = c(4, 0.5), rot.per = 0)

# crear matriz
X <- as.matrix(tweets_dtm2)
X_std <- (X - min(X)) / (max(X) - min(X))
X_scaled <- X_std * (1000 - 0) + 0
X_scaled <- round(X_scaled, 0)

# hay que convertir matriz en DTM
# Por toda esta gestión toca volver a hacer el proceso desde cero
tweets_dtm3 <- X_scaled %>%
  as.data.frame() %>%
  rownames_to_column() %>%
  pivot_longer(-rowname) %>%
  cast_dtm(document = rowname,
           term = name, 
           value = value)
inspect(ensayos_dtm3)

dim(tweets_dtm3)


#### LDA
p_load(topicmodels)

# k: numero de clusters
ensayos_lda <- LDA(ensayos_dtm3, k = 3, 
                   control = list(seed = 666))

ensayos_lda

# beta: probabilidad de topicos de pertenecer a un autor
ensayos_topics <- tidy(ensayos_lda, matrix = "beta")
ensayos_topics

ensayos_top_terms <- ensayos_topics %>%
  group_by(topic) %>%
  slice_max(beta, n = 10) %>% 
  ungroup() %>%
  arrange(topic, -beta)

ensayos_top_terms %>%
  mutate(term = reorder_within(term, beta, topic)) %>%
  ggplot(aes(beta, term, fill = factor(topic))) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~ topic, scales = "free") +
  scale_y_reordered()


# calcular topic score w_ik=gamma
# probabilidad de pertenencia de un tweet a un topico
ensayos_documents <- tidy(ensayos_lda, matrix = "gamma")
ensayos_documents





