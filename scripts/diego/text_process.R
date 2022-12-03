
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


########## PROCESAMIENTO DE TEXTO

## TOKENS: word=token
# Generar una columna con los tokens, aplanando el tibble en un-token por fila. 
# word = token
db_tidy <- base_train %>%
  unnest_tokens(output = word, input = text) %>%
  anti_join(get_stopwords("es"),"word")

# contar tokens en base
db_tidy <- db_tidy[!(db_tidy$word=="https" | db_tidy$word=="t.co" | db_tidy$word=="co" | db_tidy$word=="t"), ]
db_tidy %>% count(word, sort = TRUE) %>% head(50)

## n palabras con mayor frecuencia absoluta
# contar word=token
# filtrar tokens en character
top_words <- db_tidy %>%
  count(word, sort = TRUE) %>%
  filter(!word %in% as.character(0:10)) %>%
  slice_max(n, n = 100) %>%
  pull(word)%>%head(10)

top_words
subset(base_train,id=="DgJXrP7JZuQS7trOVPj4xg==")$text

### contar combinaciones de word-name
# combinaciones de tokens y name

# count_words sobre la base db_tidy
# contar (word=token, name)
# complete (word=token, name, fill = list(n = 0)) - expandir todas las combinaciones
count_words <- db_tidy %>%
  count(word, name) %>%
  complete(word, name, fill = list(n = 0)) ## expandir todas las posibles combinaciones
count_words %>% head(20)

### obtener la frecuencia relativa word-name

# word_freqs sobre base de count_words
# agrupar por nombre
# mutate (sobreescribir/crear variable) name_sum que es suma de n
# mutate proporcion = n/name_sum
# ungroup, filter (word=token en top_words)

word_freqs <- count_words %>%
  group_by(name) %>%
  mutate(name_sum = sum(n),
         proportion = n / name_sum) %>%
  ungroup() %>%
  filter(word %in% top_words)
word_freqs



### Identificar palabras que aumentan o disminuyen con la frecuencia de nombre
# crear modelo sobre word_freqs
# nest(data = c(name, n, suma name, proporcion) )
# mutate (modelo = map(.x = data, .f = ~glm(cbind(n, price_sum) ~ price , data = . , family = "binomial"))

# modelo
# map (.x = data,
#      .f = ~ glm(cbind(n, price_name) ~ name
#      .data = .
#      .family = "binomial"

# model = map(model, tidy)

word_model <- word_freqs %>%
  nest(data = c(name, n, name_sum, proportion)) %>%
  mutate(model = map(.x = data,
                     .f = ~ glm(cbind(n, name_sum) ~ name, data = . , family = "binomial")),
         model = map(model, tidy)) %>%
  unnest(model) %>%
  filter(term == "name") %>%
  mutate(p.value = p.adjust(p.value)) %>%
  arrange(-estimate)
word_model

### plot estimate vs p-value
word_model %>%
  ggplot(aes(estimate, p.value)) +
  geom_vline(xintercept = 0, lty = 2, alpha = 0.7, color = "gray50") +
  geom_point(color = "midnightblue", alpha = 0.8, size = 2.5) +
  scale_y_log10() +
  geom_text_repel(aes(label = word))

## generar vector con palabras que mas aumentan o disminuyen con el nombre
higher_words <- word_model %>%
  filter(p.value < 0.05) %>%
  slice_max(estimate, n = 12) %>%
  pull(word)

lower_words <- word_model %>%
  filter(p.value < 0.05) %>%
  slice_max(-estimate, n = 12) %>%
  pull(word)

### plot lower_words
# base word_freqs
word_freqs %>%
  filter(word %in% lower_words) %>%
  ggplot(aes(price, proportion, color = word)) +
  geom_line(size = 2.5, alpha = 0.7, show.legend = FALSE) +
  facet_wrap(vars(word), scales = "free_y") + theme_light()

## plot higher_words
word_freqs %>%
  filter(word %in% higher_words) %>%
  ggplot(aes(price, proportion, color = word)) +
  geom_line(size = 2.5, alpha = 0.7, show.legend = FALSE) +
  facet_wrap(vars(word), scales = "free_y") + theme_light()


