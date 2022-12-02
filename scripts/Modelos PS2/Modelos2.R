####Primeros modelos, Daniel Franco.


#Limpiar el ambiente
rm(list=ls())
#Establecer directorios
#Daniel
setwd("C:/Users/danie/OneDrive/Escritorio/Uniandes/PEG/Big Data and Machine Learning/BD-ML---PS2/data")
#Diego
setwd("C:/Users/Diego/OneDrive/Documents/GitHub/BD-ML---PS2/data")
#Samuel
setwd("~/Desktop/Big Data/Repositorios/BD-ML---PS2/data")


#Importar paquetes y cargar librerías
require(pacman)
p_load(tidyverse, rvest, data.table, dplyr, skimr, caret, rio, 
       vtable, stargazer, ggplot2, boot, MLmetrics, lfe, 
       tidyverse, fabricatr, stargazer, Hmisc, writexl, viridis, here,
       modelsummary, # tidy, msummary
       gamlr,        # cv.gamlr
       ROCR, # ROC curve
       class, glmnet, janitor, doParallel, rattle, fastDummies, tidymodels, themis, AER)

data <- read.csv("train_completa.csv")

data%>%count(Dominio)

data <- data%>%subset(Dominio != "BOGOTA")
data <- data[-c(1,2)]
names(data)

data$prop_ocupados_pet <- ifelse(is.na(data$prop_ocupados_pet),0, data$prop_ocupados_pet)
data$prop_Desocupados_pet <- ifelse(is.na(data$prop_Desocupados_pet),0, data$prop_Desocupados_pet)
data$prop_cotiza <- ifelse(is.na(data$prop_cotiza),0, data$prop_cotiza)
data$prop_ninis <- ifelse(is.na(data$prop_ninis),0, data$prop_ninis)


# Check los NAs de la base
sapply(data, function(x) sum(is.na(x)))

#Revisemos qué variables tenemos
names(data)

# evaluar correlacion entre variables
cor(data$Cant_cotiza_recibe, data$prop_cotiza)




#ingreso en logaritmo
data$log_ingtot <- log(data$Ingtotugarr+1)


is.na(data$P5100)%>%table()

categoricas <- c("Pobre", "Clase", "Dominio", "P5090", "Depto", "jefe_mujer", "P6090", "jefe_cotiza", "relab_jefe",
                 "P6090", "jefe_cotiza", "relab_jefe", "max_edu_lev_h", "max_empl", "Relab1", "Relab2", "Relab3",
                 "Educ1", "Educ2", "Educ3", "hijos", "pareja", "nietos", "otros_parientes", "no_parientes", "emp_pen",
                 "recibe_arriendos")

for (var in categoricas) {
  data[,var] <- as.factor(data[,var, drop = TRUE])
}

names(data)
data <- data[-c(1:9,11,15:18,20,22,24,29,31,38,42:45,52,54,57,58,60, 66:68)]

#Se crea la matriz de dummys
df <- model.matrix(~ .  - 1, data)%>%data.frame()

#write.csv(df,"dfdummy.csv")

# Dividimos train/test (80/20)

#Se establece semilla
set.seed(1000)
n <- nrow(df)
smp_size <- floor(0.8*n)
train_ind <- sample(1:n, size = smp_size)
#Crear train set para ajustar los parámetros
train <- df[train_ind, ]
#Crear test set para evaluar el modelo
test <- df[-train_ind, ]





# Estandarizamos DESPUÉS de partir la base en train/test

names(df)
names(data)
variables_numericas <- c("P5000", "P5010", "num_mujeres", "P6040", "Horas_Hogar", "Horas_reales",
                         "Num_ocu_hogar", "Num_des_hogar", 
                         "prop_ocupados_pet",
                         "prop_Desocupados_pet", "prop_mujeres_total",
                         "prop_cotiza", "ppc", "Valor_Arriendo", "age2", "años_educ_promedio", "ninis", "prop_ninis")



escalador <- preProcess(train[, variables_numericas])
train_s <- train
test_s <- test

train_s[, variables_numericas] <- predict(escalador, train[, variables_numericas])
test_s[, variables_numericas] <- predict(escalador, test[, variables_numericas])

train_s <- data.frame(train_s)
test_s <- data.frame(test_s)
train <- data.frame(train)
test <- data.frame(test)

#max(train_s$prop_cotiza)

names(train_s)


write.csv(train_s,"train_s.csv")
write.csv(test_s,"test_s.csv")

#Se crean pobre como factores
train_s$Pobre1 <- factor(train_s$Pobre1)
test_s$Pobre1 <- factor(test_s$Pobre1)


###Undersampling
train_s_under <- recipe(Pobre1~., data = train_s)%>%
  themis::step_downsample(Pobre1, under_ratio = 1.5)%>%
  prep()%>%
  bake(new_data = NULL)

train_s_under <- as.data.frame(train_s_under)

write.csv(train_s_under,"train_s_under.csv")

prop.table(table(train_s$Pobre1))
prop.table(table(train_s_under$Pobre1))
prop.table(table(test_s$Pobre1))


y_train <- train_s_under[,"log_ingtot"]
y_test <-  test_s[,"log_ingtot"]
p_train <- train_s_under[,"Pobre1"]
p_test <-  test_s[,"Pobre1"]
names(train_s_under)
x_train_s = train_s_under[-c(1,11,105,112,113)]
x_test = test_s[-c(1,11,105,112,113)]
#X_train <- select(train, -c("Lp", "Pobre", "Ingtotugarr", "Npersug"))


#Iniciamos con la regresión lineal

names(train_s_under)

modelo1 <- lm(formula = log_ingtot ~. -Clase1-Pobre1-Lp-Ingtotugarr , data = train_s_under)
insample1 <- predict(modelo1, train_s)
y_hat_test1 <- predict(modelo1, test_s)


df_coeficientes_reg <- modelo1$coefficients %>%
  enframe(name = "predictor", value = "coeficiente")

df_coeficientes_reg %>%
  filter(predictor != "`(Intercept)`") %>%
  ggplot(aes(x = reorder(predictor, abs(coeficiente)), 
             y = coeficiente)) +
  geom_col(fill = "darkblue") +
  coord_flip() +
  labs(title = "Coeficientes del modelo de regresión", 
       x = "Variables",
       y = "Coeficientes") +
  theme_bw()

#Dentro de muestra
resultados <- train_s%>%select(Ingtotugarr, Pobre1, Npersug, Lp)
resultados$pred_lm <- exp(insample1)
resultados$pobre_lm <- ifelse(resultados$pred_lm/resultados$Npersug <= resultados$Lp, 1, 0)


cm_lm <- confusionMatrix(data=factor(resultados$pobre_lm) , 
                         reference=factor(resultados$Pobre) , 
                         mode="sens_spec" , positive="1")
cm_lm



#Fuera de muestra
resultados2 <- test_s%>%select(Ingtotugarr, Pobre1, Npersug, Lp)
resultados2$pred_lm <- exp(y_hat_test1)
resultados2$pobre_lm <- ifelse(resultados2$pred_lm/resultados2$Npersug <= resultados2$Lp, 1, 0)



cm_lm2 <- confusionMatrix(data=factor(resultados2$pobre_lm) , 
                          reference=factor(resultados2$Pobre1) , 
                          mode="sens_spec" , positive="1")
cm_lm2


#No está mal, pero tenemos muchas variables

#agregado


names(train_s_under)


##################################Modelos 1
#Edad + jefe_mujer + máximo nivel educ + núm empleados + relab1 + relab2 + tipo vivienda + prop_cotiza + prop_desocupados_pet + horas_hogar + recibe arriendos + clase 
#12 variables

x_train_s = train_s_under[c("P6040", "age2", "jefe_mujer1", "max_edu_lev_h2", "max_edu_lev_h3",
                            "max_edu_lev_h4", "max_edu_lev_h5", "max_edu_lev_h6", "max_empl1",
                            "max_empl2", "max_empl3", "max_empl4", "max_empl5", "max_empl6",
                            "max_empl7", "max_empl8", "max_empl9", "Relab12", "Relab13", 
                            "Relab14", "Relab15", "Relab16", "Relab17", "Relab18", "Relab19",
                            "Relab110", "Relab111", "Relab112", "recibe_arriendos1",
                            "P50902", "P50903", "P50904", "P50905", "P50906", "Clase2",
                            "Relab22", "Relab23", 
                            "Relab24", "Relab25", "Relab26", "Relab27", "Relab28", "Relab29",
                            "Relab210", "Relab211", "Relab212", "Relab213", "prop_cotiza",
                            "prop_Desocupados_pet", "Horas_Hogar")]
x_train_s2 = train_s[c("P6040", "age2", "jefe_mujer1", "max_edu_lev_h2", "max_edu_lev_h3",
                       "max_edu_lev_h4", "max_edu_lev_h5", "max_edu_lev_h6", "max_empl1",
                       "max_empl2", "max_empl3", "max_empl4", "max_empl5", "max_empl6",
                       "max_empl7", "max_empl8", "max_empl9", "Relab12", "Relab13", 
                       "Relab14", "Relab15", "Relab16", "Relab17", "Relab18", "Relab19",
                       "Relab110", "Relab111", "Relab112", "recibe_arriendos1",
                       "P50902", "P50903", "P50904", "P50905", "P50906", "Clase2",
                       "Relab22", "Relab23", 
                       "Relab24", "Relab25", "Relab26", "Relab27", "Relab28", "Relab29",
                       "Relab210", "Relab211", "Relab212", "Relab213", "prop_cotiza",
                       "prop_Desocupados_pet", "Horas_Hogar")]

x_test = test_s[c("P6040", "age2", "jefe_mujer1", "max_edu_lev_h2", "max_edu_lev_h3",
                  "max_edu_lev_h4", "max_edu_lev_h5", "max_edu_lev_h6", "max_empl1",
                  "max_empl2", "max_empl3", "max_empl4", "max_empl5", "max_empl6",
                  "max_empl7", "max_empl8", "max_empl9", "Relab12", "Relab13", 
                  "Relab14", "Relab15", "Relab16", "Relab17", "Relab18", "Relab19",
                  "Relab110", "Relab111", "Relab112", "recibe_arriendos1",
                  "P50902", "P50903", "P50904", "P50905", "P50906", "Clase2",
                  "Relab22", "Relab23", 
                  "Relab24", "Relab25", "Relab26", "Relab27", "Relab28", "Relab29",
                  "Relab210", "Relab211", "Relab212", "Relab213", "prop_cotiza",
                  "prop_Desocupados_pet", "Horas_Hogar")]

############################Modelos 2
#Edad + jefe_mujer + máximo nivel educ + núm empleados + relab1  + prop_cotiza + prop_desocupados_pet + horas_hogar 
#8 variables


x_train_s = train_s_under[c("P6040", "age2", "jefe_mujer1", "max_edu_lev_h2", "max_edu_lev_h3",
                            "max_edu_lev_h4", "max_edu_lev_h5", "max_edu_lev_h6", "max_empl1",
                            "max_empl2", "max_empl3", "max_empl4", "max_empl5", "max_empl6",
                            "max_empl7", "max_empl8", "max_empl9", "Relab12", "Relab13", 
                            "Relab14", "Relab15", "Relab16", "Relab17", "Relab18", "Relab19",
                            "Relab110", "Relab111", "Relab112",
                            "prop_cotiza",
                            "prop_Desocupados_pet", "Horas_Hogar")]
x_train_s2 = train_s[c("P6040", "age2", "jefe_mujer1", "max_edu_lev_h2", "max_edu_lev_h3",
                       "max_edu_lev_h4", "max_edu_lev_h5", "max_edu_lev_h6", "max_empl1",
                       "max_empl2", "max_empl3", "max_empl4", "max_empl5", "max_empl6",
                       "max_empl7", "max_empl8", "max_empl9", "Relab12", "Relab13", 
                       "Relab14", "Relab15", "Relab16", "Relab17", "Relab18", "Relab19",
                       "Relab110", "Relab111", "Relab112",
                       "prop_cotiza",
                       "prop_Desocupados_pet", "Horas_Hogar")]

x_test = test_s[c("P6040", "age2", "jefe_mujer1", "max_edu_lev_h2", "max_edu_lev_h3",
                  "max_edu_lev_h4", "max_edu_lev_h5", "max_edu_lev_h6", "max_empl1",
                  "max_empl2", "max_empl3", "max_empl4", "max_empl5", "max_empl6",
                  "max_empl7", "max_empl8", "max_empl9", "Relab12", "Relab13", 
                  "Relab14", "Relab15", "Relab16", "Relab17", "Relab18", "Relab19",
                  "Relab110", "Relab111", "Relab112", 
                  "prop_cotiza",
                  "prop_Desocupados_pet", "Horas_Hogar")]



###############################################Modelos 3
############################Modelos 2
#Edad + prop_ninis + jefe_mujer + máximo nivel educ + núm empleados + relab1  + prop_cotiza + prop_desocupados_pet + horas_hogar 
#9 variables


x_train_s = train_s_under[c("P6040", "age2", "jefe_mujer1", "max_edu_lev_h2", "max_edu_lev_h3",
                            "max_edu_lev_h4", "max_edu_lev_h5", "max_edu_lev_h6", "max_empl1",
                            "max_empl2", "max_empl3", "max_empl4", "max_empl5", "max_empl6",
                            "max_empl7", "max_empl8", "max_empl9", "Relab12", "Relab13", 
                            "Relab14", "Relab15", "Relab16", "Relab17", "Relab18", "Relab19",
                            "Relab110", "Relab111", "Relab112",
                            "prop_cotiza",
                            "prop_Desocupados_pet", "Horas_Hogar", "prop_ninis")]
x_train_s2 = train_s[c("P6040", "age2", "jefe_mujer1", "max_edu_lev_h2", "max_edu_lev_h3",
                       "max_edu_lev_h4", "max_edu_lev_h5", "max_edu_lev_h6", "max_empl1",
                       "max_empl2", "max_empl3", "max_empl4", "max_empl5", "max_empl6",
                       "max_empl7", "max_empl8", "max_empl9", "Relab12", "Relab13", 
                       "Relab14", "Relab15", "Relab16", "Relab17", "Relab18", "Relab19",
                       "Relab110", "Relab111", "Relab112",
                       "prop_cotiza",
                       "prop_Desocupados_pet", "Horas_Hogar", "prop_ninis")]

x_test = test_s[c("P6040", "age2", "jefe_mujer1", "max_edu_lev_h2", "max_edu_lev_h3",
                  "max_edu_lev_h4", "max_edu_lev_h5", "max_edu_lev_h6", "max_empl1",
                  "max_empl2", "max_empl3", "max_empl4", "max_empl5", "max_empl6",
                  "max_empl7", "max_empl8", "max_empl9", "Relab12", "Relab13", 
                  "Relab14", "Relab15", "Relab16", "Relab17", "Relab18", "Relab19",
                  "Relab110", "Relab111", "Relab112", 
                  "prop_cotiza",
                  "prop_Desocupados_pet", "Horas_Hogar", "prop_ninis")]




#####################################Modelos4

train$años_educ_promedio



x_train_s = train_s_under[c("P6040", "age2", "jefe_mujer1", "Relab12", "Relab13", 
                            "Relab14", "Relab15", "Relab16", "Relab17", "Relab18", "Relab19",
                            "Relab110", "Relab111", "Relab112", "recibe_arriendos1",
                            "P50902", "P50903", "P50904", "P50905", "P50906", "Clase2",
                            "Relab22", "Relab23", 
                            "Relab24", "Relab25", "Relab26", "Relab27", "Relab28", "Relab29",
                            "Relab210", "Relab211", "Relab212", "Relab213", "prop_cotiza",
                            "prop_Desocupados_pet", "Horas_Hogar", "años_educ_promedio")]
x_train_s2 = train_s[c("P6040", "age2", "jefe_mujer1", "Relab12", "Relab13", 
                       "Relab14", "Relab15", "Relab16", "Relab17", "Relab18", "Relab19",
                       "Relab110", "Relab111", "Relab112", "recibe_arriendos1",
                       "P50902", "P50903", "P50904", "P50905", "P50906", "Clase2",
                       "Relab22", "Relab23", 
                       "Relab24", "Relab25", "Relab26", "Relab27", "Relab28", "Relab29",
                       "Relab210", "Relab211", "Relab212", "Relab213", "prop_cotiza",
                       "prop_Desocupados_pet", "Horas_Hogar", "años_educ_promedio")]

x_test = test_s[c("P6040", "age2", "jefe_mujer1", "Relab12", "Relab13", 
                  "Relab14", "Relab15", "Relab16", "Relab17", "Relab18", "Relab19",
                  "Relab110", "Relab111", "Relab112", "recibe_arriendos1",
                  "P50902", "P50903", "P50904", "P50905", "P50906", "Clase2",
                  "Relab22", "Relab23", 
                  "Relab24", "Relab25", "Relab26", "Relab27", "Relab28", "Relab29",
                  "Relab210", "Relab211", "Relab212", "Relab213", "prop_cotiza",
                  "prop_Desocupados_pet", "Horas_Hogar", "años_educ_promedio")]




#Lasso y Ridge para regresión lineal
#Primero se corre ridge
# Ridge. Alpha = 0

#Lisra de los alphas
lista_alpha <- seq(0,1,0.05)
#df vacío que se llenará con los datos del ciclo
resultados_ridge <- data.frame()

for (alpha_en in lista_alpha) {
  
  #Modelo glmnet
  modelo_ridge <- glmnet(
    x = x_train_s,
    y = y_train,
    alpha = alpha_en,
    nlambda = 300,
    standardize = FALSE)
  
  
  #Predicción y en el train
  y_hat_train_ridge <- predict(modelo_ridge, 
                               newx = as.matrix(x_train_s))
  
  #Predicción y en el test
  y_hat_test_ridge <- predict(modelo_ridge, 
                              newx = as.matrix(x_test))
  
  #Predicción pobre en el test (del)
  p_hat_test <- ifelse(exp(predict(modelo_ridge, newx = as.matrix(x_test)))/test_s$Npersug < test_s$Lp,1,0)
  
  
  #Lambdas del modelo ridge
  lambdas_ridge <- modelo_ridge$lambda
  
  #Validación cruzada
  #Vamos a intentar escoger el modelo que maximice el recall
  
  
  #El ciclo guarda los valores de la métrica de evaluación
  for (i in 1:length(lambdas_ridge)) {
    lreg <- lambdas_ridge[i]
    p_hat_test_pred_ridge <- p_hat_test[, i]
    cm_ridge_test <- confusionMatrix(data=factor(p_hat_test_pred_ridge) , 
                                     reference=factor(p_test) , 
                                     mode="sens_spec" , positive="1")
    recall <- cm_ridge_test$byClass[[1]]
    specif <- cm_ridge_test$byClass[[2]]
    avg <- 0.75*recall + 0.25*specif
    resultado <- data.frame(Modelo = "Ridge",
                            Muestra = "Fuera",
                            Lambda = lreg,
                            ALPHA = alpha_en,
                            RECAL = recall,
                            SPEC = specif,
                            AVG = avg)
    resultados_ridge <- bind_rows(resultados_ridge, resultado)
  }
}
  

#Ver cuál es el máximo avg y escoger los valores de lambda y alpha correspondientes

filtro_recall <- resultados_ridge$AVG == max(resultados_ridge$AVG)
mejor_lambda <- resultados_ridge[filtro_recall, "Lambda"]  #0.005
mejor_alpha <- resultados_ridge[filtro_recall, "ALPHA"]


#Prediciones en el train

#Se corre el modelo con los resultados antes obtenidos
modelo_ridge <- glmnet(
  x = x_train_s,
  y = y_train,
  alpha = mejor_alpha,
  lambda = mejor_lambda,
  standardize = FALSE)

#Predicción en el train normal (sin undersample)
y_hat_in3 <- predict.glmnet(modelo_ridge,
                            newx = as.matrix(x_train_s2),
                            s = mejor_lambda)


y_hat_in3 <- as.data.frame(y_hat_in3)
y_hat_in3$ingtot <- exp(y_hat_in3$s1)
y_hat_in3$Lp <- train_s$Lp
y_hat_in3$Pobre <- train_s$Pobre1
y_hat_in3$Npersug <- train_s$Npersug
#Se predicen los 1's y 0's
y_hat_in3$Pred_Pobre <- ifelse(y_hat_in3$ingtot/y_hat_in3$Npersug < y_hat_in3$Lp, 1, 0)


#Matriz de confución de la base de entrenamiento
cm_lmridge_train <- confusionMatrix(data=factor(y_hat_in3$Pred_Pobre) , 
                                    reference=factor(y_hat_in3$Pobre) , 
                                    mode="sens_spec" , positive="1")
cm_lmridge_train$byClass[1]*0.75+cm_lmridge_train$byClass[1]*0.25



#Predicción en el test
y_hat_out <- predict.glmnet(modelo_ridge,
                            newx = as.matrix(x_test),
                            s = mejor_lambda)


y_hat_out <- as.data.frame(y_hat_out)
y_hat_out$ingtot <- exp(y_hat_out$s1)
y_hat_out$Lp <- test_s$Lp
y_hat_out$Pobre <- test_s$Pobre1
y_hat_out$Npersug <- test_s$Npersug
#Se predicen los 1's y 0's
y_hat_out$Pred_Pobre <- ifelse(y_hat_out$ingtot/y_hat_out$Npersug < y_hat_out$Lp, 1, 0)


#Matriz de confución de la base de entrenamiento
cm_lmridge_test <- confusionMatrix(data=factor(y_hat_out$Pred_Pobre) , 
                                    reference=factor(y_hat_out$Pobre) , 
                                    mode="sens_spec" , positive="1")
cm_lmridge_test$byClass[1]*0.75+cm_lmridge_test$byClass[2]*0.25




#Con lo anterior tenemos un sentitivity de 0.78 y un specificity de 0.84


#Gráfico de los mejores lambda
ggplot(resultados_ridge, aes(x = Lambda, y = RECAL)) +
  geom_point() +
  geom_line() +
  theme_bw() +
  scale_y_continuous(labels = scales::comma)

#¿Qué hubiera pasado si lo corriamos con lambda = 0?
y_hat_out0 <- predict.glmnet(modelo_ridge,
                             newx = as.matrix(x_test),
                             s = 0.0)


y_hat_out0 <- as.data.frame(y_hat_out0)
y_hat_out0$ingtot <- exp(y_hat_out0$s1)
y_hat_out0$Lp <- test_s$Lp
y_hat_out0$Pobre <- test_s$Pobre1
y_hat_out0$Npersug <- test_s$Npersug
y_hat_out0$Pred_Pobre <- ifelse(y_hat_out0$ingtot/y_hat_out0$Npersug < y_hat_out0$Lp, 1, 0)

cm_lmridge_test0 <- confusionMatrix(data=factor(y_hat_out0$Pred_Pobre) , 
                                    reference=factor(y_hat_out0$Pobre) , 
                                    mode="sens_spec" , positive="1")
cm_lmridge_test0

#Con lo anterior (lambda en 0, modelo original) tenemos un sentitivity de 0.78 y un specificity de 0.84
#Mejora ligeramente pero da casi lo mismo





#Segundo, se corre lasso
# Lasso. Alpha = 1
modelo_lasso <- glmnet(
  x = x_train_s,
  y = y_train,
  alpha = 1,
  lambda = seq(0,5,0.001),
  standardize = FALSE
)



y_hat_train_lasso <- predict(modelo_lasso, 
                             newx = as.matrix(x_train_s))

y_hat_test_lasso <- predict(modelo_lasso, 
                            newx = as.matrix(x_test))

p_hat_test_lasso <- ifelse(exp(predict(modelo_lasso, newx = as.matrix(x_test)))/test_s$Npersug < test_s$Lp,1,0)


#Lambdas del modelo ridge
lambdas_lasso <- modelo_lasso$lambda

#Validación cruzada
#Vamos a intentar escoger el modelo que maximice el recall
resultados_lasso <- data.frame()
for (i in 1:length(lambdas_lasso)) {
  lreg <- lambdas_lasso[i]
  y_hat_test_pred_lasso <- y_hat_test_lasso[, i]
  p_hat_test_pred_lasso <- p_hat_test_lasso[, i]
  r23 <- R2_Score(y_pred = y_hat_test_pred_lasso, y_true = y_test)
  rmse3 <- RMSE(y_pred = y_hat_test_pred_lasso, y_true = y_test)
  cm_lasso_test <- confusionMatrix(data=factor(p_hat_test_pred_lasso) , 
                                   reference=factor(p_test) , 
                                   mode="sens_spec" , positive="1")
  recall <- cm_lasso_test$byClass[[1]]
  resultado <- data.frame(Modelo = "Ridge",
                          Muestra = "Fuera",
                          Lambda = lreg,
                          R2_Score = r23, 
                          RMSE = rmse3,
                          RECAL = recall)
  resultados_lasso <- bind_rows(resultados_lasso, resultado)
}

#Ver cuál es el recall máximo

filtro_recall_lasso <- resultados_lasso$RECAL == max(resultados_lasso$RECAL)
mejor_lambda_lasso_recall <- resultados_lasso[filtro_recall_lasso, "Lambda"]  #0.0

#Prediciones en el train
y_hat_inlasso <- predict.glmnet(modelo_lasso,
                                newx = as.matrix(x_train_s2),
                                s = mejor_lambda_lasso_recall)


y_hat_inlasso <- as.data.frame(y_hat_inlasso)
y_hat_inlasso$ingtot <- exp(y_hat_inlasso$s1)
y_hat_inlasso$Lp <- train_s$Lp
y_hat_inlasso$Pobre <- train_s$Pobre1
y_hat_inlasso$Npersug <- train_s$Npersug
y_hat_inlasso$Pred_Pobre <- ifelse(y_hat_inlasso$ingtot/y_hat_inlasso$Npersug < y_hat_inlasso$Lp, 1, 0)

cm_lmlasso_train <- confusionMatrix(data=factor(y_hat_inlasso$Pred_Pobre) , 
                                    reference=factor(y_hat_inlasso$Pobre) , 
                                    mode="sens_spec" , positive="1")
cm_lmlasso_train

#Predicciones en el test
y_hat_outlasso <- predict.glmnet(modelo_lasso,
                                 newx = as.matrix(x_test),
                                 s = mejor_lambda_lasso_recall)


y_hat_outlasso <- as.data.frame(y_hat_outlasso)
y_hat_outlasso$ingtot <- exp(y_hat_outlasso$s1)
y_hat_outlasso$Lp <- test_s$Lp
y_hat_outlasso$Pobre <- test_s$Pobre1
y_hat_outlasso$Npersug <- test_s$Npersug
y_hat_outlasso$Pred_Pobre <- ifelse(y_hat_outlasso$ingtot/y_hat_outlasso$Npersug < y_hat_outlasso$Lp, 1, 0)

cm_lmlasso_test <- confusionMatrix(data=factor(y_hat_outlasso$Pred_Pobre) , 
                                   reference=factor(y_hat_outlasso$Pobre) , 
                                   mode="sens_spec" , positive="1")
cm_lmlasso_test



#Con lo anterior tenemos un sentitivity de 0.78 y un specificity de 0.84


#Gráfico de los mejores lambda
ggplot(resultados_ridge, aes(x = Lambda, y = RECAL)) +
  geom_point() +
  geom_line() +
  theme_bw() +
  scale_y_continuous(labels = scales::comma)

#¿Qué hubiera pasado si lo corriamos con lambda = 0?
y_hat_out0 <- predict.glmnet(modelo_ridge,
                             newx = as.matrix(x_test),
                             s = 0.0)


y_hat_out0 <- as.data.frame(y_hat_out0)
y_hat_out0$ingtot <- exp(y_hat_out0$s1)
y_hat_out0$Lp <- test_s$Lp
y_hat_out0$Pobre <- test_s$Pobre1
y_hat_out0$Npersug <- test_s$Npersug
y_hat_out0$Pred_Pobre <- ifelse(y_hat_out0$ingtot/y_hat_out0$Npersug < y_hat_out0$Lp, 1, 0)

cm_lmridge_test0 <- confusionMatrix(data=factor(y_hat_out0$Pred_Pobre) , 
                                    reference=factor(y_hat_out0$Pobre) , 
                                    mode="sens_spec" , positive="1")
cm_lmridge_test0

#Con lo anterior (lambda en 0, modelo original) tenemos un sentitivity de 0.78 y un specificity de 0.84
#Mejora ligeramente pero da casi lo mismo



#Probemos algo:
y_train <- as.data.frame(y_train)
train_unido <- cbind(y_train, x_train_s)

matriz_inter <- sparse.model.matrix(y_train~.^2, data = train_unido)[,-1]

y_test <- as.data.frame(y_test)
test_unido <- cbind(y_test, x_test)

x_test_inter <- sparse.model.matrix(y_test~.^2, data = test_unido)[,-1]


y_train <- train_s_under[,"log_ingtot"]

#Se crea el modelo
modelo_lasso_inter <- glmnet(
  x = matriz_inter,
  y = y_train,
  alpha = 1,
  lambda = seq(0,5,0.001),
  standardize = FALSE
)


#Se predicen los resultados
y_hat_test_lasso_inter <- predict(modelo_lasso_inter, 
                                  newx = as.matrix(x_test_inter))

p_hat_test_lasso_inter <- ifelse(exp(predict(modelo_lasso_inter, newx = as.matrix(x_test_inter)))/test_s$Npersug < test_s$Lp,1,0)



#Lambdas del modelo ridge
lambdas_lasso_int <- modelo_lasso_inter$lambda


#Se hace el cilclo para escoger
resultados_lasso_int <- data.frame()
for (i in 1:length(lambdas_lasso_int)) {
  lreg <- lambdas_lasso_int[i]
  y_hat_test_pred_lasso <- y_hat_test_lasso_inter[, i]
  p_hat_test_pred_lasso <- p_hat_test_lasso_inter[, i]
  r23 <- R2_Score(y_pred = y_hat_test_pred_lasso, y_true = y_test)
  rmse3 <- RMSE(y_pred = y_hat_test_pred_lasso, y_true = y_test)
  cm_lasso_test <- confusionMatrix(data=factor(p_hat_test_pred_lasso) , 
                                   reference=factor(p_test) , 
                                   mode="sens_spec" , positive="1")
  recall <- cm_lasso_test$byClass[[1]]
  resultado <- data.frame(Modelo = "Ridge",
                          Muestra = "Fuera",
                          Lambda = lreg,
                          R2_Score = r23, 
                          RMSE = rmse3,
                          RECAL = recall)
  resultados_lasso_int <- bind_rows(resultados_lasso_int, resultado)
}



filtro_recall_lasso_inter <- resultados_lasso_int$RECAL == max(resultados_lasso_int$RECAL)
mejor_lambda_lasso_recall_int <- resultados_lasso_int[filtro_recall_lasso_inter, "Lambda"]  #0.001


#Predicciones en el test
y_hat_outlasso_inter <- predict.glmnet(modelo_lasso_inter,
                                       newx = as.matrix(x_test_inter),
                                       s = mejor_lambda_lasso_recall_int)


y_hat_outlasso_inter <- as.data.frame(y_hat_outlasso_inter)
y_hat_outlasso_inter$ingtot <- exp(y_hat_outlasso_inter$s1)
y_hat_outlasso_inter$Lp <- test_s$Lp
y_hat_outlasso_inter$Pobre <- test_s$Pobre1
y_hat_outlasso_inter$Npersug <- test_s$Npersug
y_hat_outlasso_inter$Pred_Pobre <- ifelse(y_hat_outlasso_inter$ingtot/y_hat_outlasso_inter$Npersug < y_hat_outlasso_inter$Lp, 1, 0)

cm_lmlasso_test_inter <- confusionMatrix(data=factor(y_hat_outlasso_inter$Pred_Pobre) , 
                                         reference=factor(y_hat_outlasso_inter$Pobre) , 
                                         mode="sens_spec" , positive="1")
cm_lmlasso_test_inter 


#Predicción sobre el train
y_train_or <- train_s[,"log_ingtot"]
y_train_or <- as.data.frame(y_train_or)
train_unido_or <- cbind(y_train_or, x_train_s2)

matriz_inter_or <- sparse.model.matrix(y_train_or~.^2, data = train_unido_or)[,-1]


y_hat_inlasso <- predict.glmnet(modelo_lasso_inter,
                                newx = as.matrix(matriz_inter_or),
                                s = mejor_lambda_lasso_recall_int)


y_hat_inlasso <- as.data.frame(y_hat_inlasso)
y_hat_inlasso$ingtot <- exp(y_hat_inlasso$s1)
y_hat_inlasso$Lp <- train_s$Lp
y_hat_inlasso$Pobre <- train_s$Pobre1
y_hat_inlasso$Npersug <- train_s$Npersug
y_hat_inlasso$Pred_Pobre <- ifelse(y_hat_inlasso$ingtot/y_hat_inlasso$Npersug < y_hat_inlasso$Lp, 1, 0)

cm_lmlasso_train <- confusionMatrix(data=factor(y_hat_inlasso$Pred_Pobre) , 
                                    reference=factor(y_hat_inlasso$Pobre) , 
                                    mode="sens_spec" , positive="1")
cm_lmlasso_train




#Modelo sin ridge ni lasso
modelo <- lm(y_train ~ . + P6040*jefe_mujer1 + age2*jefe_mujer1
             +jefe_mujer1*Relab12+jefe_mujer1*Relab13+jefe_mujer1*Relab14+
               jefe_mujer1*Relab15+jefe_mujer1*Relab16+jefe_mujer1*Relab17+
               jefe_mujer1*Relab18 + jefe_mujer1*Relab19+jefe_mujer1*Relab110+
               jefe_mujer1*Relab111+jefe_mujer1*Relab112, data = train_unido)
summary(modelo)

insamplep <- predict(modelo, train_s)



#Dentro de muestra
resultadosp <- train_s%>%select(Ingtotugarr, Pobre1, Npersug, Lp)
resultadosp$pred_lm <- exp(insamplep)
resultadosp$pobre_lm <- ifelse(resultadosp$pred_lm/resultadosp$Npersug <= resultadosp$Lp, 1, 0)


cm_lmp <- confusionMatrix(data=factor(resultadosp$pobre_lm) , 
                          reference=factor(resultadosp$Pobre) , 
                          mode="sens_spec" , positive="1")
cm_lmp

names(data)


#######################################LOGIT Lasso Ridge

p_train <- as.data.frame(p_train)
p_train$p_train <- as.factor(p_train$p_train)
train_unido_logit <- cbind(p_train, x_train_s)


#Se crean la x y la y para los modelos
x <- sparse.model.matrix(p_train~(.)^2, data = train_unido_logit)[,-1]
y <- ifelse(p_train$p_train == 1, 1, 0)

#escoger el valor de lambda para lasso y ridge
#cv.lasso <- cv.glmnet(x, y, alpha = 1, family =  "binomial")
#cv.ridge <- cv.glmnet(x, y, alpha = 0, family =  "binomial")
n_cores <- detectCores()
cl <- makePSOCKcluster(n_cores-1)
registerDoParallel(cl)


p_train <- train_s_under[,"Pobre1"]
p_test <- test_s[,"Pobre1"]
test_unido_logit <- cbind(p_test, x_test)
x.test <- sparse.model.matrix(p_test ~(.)^2, test_unido_logit)[,-1]

p_train2 <- train_s[,"Pobre1"]
train_unido_logit2 <- cbind(p_train2, x_train_s2)
x.train <- sparse.model.matrix(p_train2 ~(.)^2, train_unido_logit2)[,-1]

resultados_logit<- data.frame()

lista_alpha <- seq(0,1,0.1)


for (elastic in lista_alpha) {
  #Modelo de lasso o ridge
  model <- glmnet(x, y, alpha = elastic, binomial(link="probit"),
                  lambda = seq(0,1,0.1), standardize = FALSE)
  
  lambdas <- model$lambda
  #predicción en el test
  
  probabilities_test <- predict(model, newx = x.test, type = "response")
  #escogencia de la regla óptima
  rules <- seq(0.1,0.5,0.05)
  #Ciclo itera sobre diversos valores de la regla y escoge la que da el máximo ponderado
  for (rule in rules) {
    for (j in 1:length(lambdas)) {
      lreg <- lambdas[j]
      resultados_log <- ifelse(probabilities_test[,j] > rule, 1, 0)
      #predicciones <- as.data.frame(predicciones)
      #resultados_log <- predicciones
      resultados_log <- data.frame(resultados_log)
      resultados_log$real <- p_test
      CM <- confusionMatrix(data = as.factor(resultados_log$resultados_log), reference = resultados_log$real, mode="sens_spec" , positive="1")
      recall <- CM$byClass[[1]]
      specif <- CM$byClass[[2]]
      avg <- 0.75*recall + 0.25*specif
      resultado <- data.frame(Modelo = "EN Logit",
                              Muestra = "Fuera",
                              Regla = rule,
                              Lambda = lreg,
                              ALPHA = elastic,
                              RECAL = recall,
                              SPEC = specif,
                              AVG = avg)
      resultados_logit <- bind_rows(resultados_logit, resultado)
    }
  }
}

stopCluster(cl)

#se escoge la que da el ponderado más alto
#Regla
optimal_rule <- resultados_logit[which.max(resultados_logit$AVG),"Regla"]
#Lambda
optimal_lambda <- resultados_logit[which.max(resultados_logit$AVG),"Lambda"]
#Alpha
optimal_alpha <- resultados_logit[which.max(resultados_logit$AVG),"ALPHA"]

#Correr modelo con valores óptimos (prueba sobre el test)
model_opt <- glmnet(x, y, alpha = optimal_alpha, binomial(link="logit"),
                    lambda = optimal_lambda, standardize = FALSE)

#Predecir las probabilidades con los valores óptimos
predichos_test <- predict(model_opt, newx = x.test, type = "response")

#Clasificación de pobres y no pobres
clasificacion_test <- ifelse(predichos_test > optimal_rule, 1, 0)

#Confussion Matrix
clas_test <- data.frame(clasificacion_test)
clas_test$real <- p_test
CM_test <- confusionMatrix(data = as.factor(clas_test$s0), reference = clas_test$real, mode="sens_spec" , positive="1")
CM_test
CM_test$byClass[1]*0.75+CM_test$byClass[2]*0.25


#Correr modelo con valores óptimos (prueba sobre elo train)
model_opt <- glmnet(x, y, alpha = optimal_alpha, binomial(link="logit"),
                    lambda = optimal_lambda, standardize = FALSE)

#Predecir las probabilidades con los valores óptimos
predichos_train <- predict(model_opt, newx = x.train, type = "response")

#Clasificación de pobres y no pobres
clasificacion_train <- ifelse(predichos_train > optimal_rule, 1, 0)

#Confussion Matrix
clas_train <- data.frame(clasificacion_train)
clas_train$real <- p_train2
CM_train <- confusionMatrix(data = as.factor(clas_train$s0), reference = clas_train$real, mode="sens_spec" , positive="1")
CM_train
CM_train$byClass[1]*0.75+CM_train$byClass[2]*0.25



#####PROBIT
#Correr modelo con valores óptimos (prueba sobre el test)
model_opt <- glmnet(x, y, alpha = 0, binomial(link="probit"),
                    lambda = 0.05, standardize = FALSE)

#Predecir las probabilidades con los valores óptimos
predichos_test <- predict(model_opt, newx = x.test, type = "response")

#Clasificación de pobres y no pobres
clasificacion_test <- ifelse(predichos_test > 0.2, 1, 0)

#Confussion Matrix
clas_test <- data.frame(clasificacion_test)
clas_test$real <- p_test
CM_test <- confusionMatrix(data = as.factor(clas_test$s0), reference = clas_test$real, mode="sens_spec" , positive="1")
CM_test
CM_test$byClass[1]*0.75+CM_test$byClass[2]*0.25


#Correr modelo con valores óptimos (prueba sobre elo train)
model_opt <- glmnet(x, y, alpha = optimal_alpha, binomial(link="probit"),
                    lambda = optimal_lambda, standardize = FALSE)

#Predecir las probabilidades con los valores óptimos
predichos_train <- predict(model_opt, newx = x.train, type = "response")

#Clasificación de pobres y no pobres
clasificacion_train <- ifelse(predichos_train > optimal_rule, 1, 0)

#Confussion Matrix
clas_train <- data.frame(clasificacion_train)
clas_train$real <- p_train2
CM_train <- confusionMatrix(data = as.factor(clas_train$s0), reference = clas_train$real, mode="sens_spec" , positive="1")
CM_train
CM_train$byClass[1]*0.75+CM_train$byClass[2]*0.25















##################################pruebas adicionales

p_train <- as.data.frame(p_train)
train_unido_logit <- cbind(p_train, x_train_s)

# Establece el k-fold cv
trctrl <- trainControl(method = "cv",number=10)

# primero vamos con ridge
enetFit <- train(p_train~., data = train_unido_logit, 
                 method = "glmnet",
                 trControl=trctrl,
                 # alpha and lambda paramters to try
                 tuneGrid = data.frame(alpha=0,
                                       lambda=seq(0,1,0.01)))
plot(varImp(enetFit),top=10)

# Mejor alpha
enetFit$bestTune #0.02

# métricas
class.res <- predict(enetFit,x_test)
confusionMatrix(class.res, p_test, mode="sens_spec" , positive="1")




#Sigue Lasso
enetFitLasso <- train(p_train~., data = train_unido_logit, 
                      method = "glmnet",
                      trControl=trctrl,
                      # alpha and lambda paramters to try
                      tuneGrid = data.frame(alpha=1,
                                            lambda=seq(0,1,0.01)))


# Mejor alpha
enetFitLasso$bestTune #0

# métricas
class.resLasso <- predict(enetFitLasso,x_test)
confusionMatrix(class.resLasso, p_test, mode="sens_spec" , positive="1")

















#Logit
logit <- glm(formula = Pobre1 ~ . -Clase1-log_ingtot-Lp-Ingtotugarr, family=binomial(link="logit") , data=train_s_under)
resultados$pobre_log <- predict(logit , newdata=train_s, type="response")

rule <- 0.5
resultados$pred_log <- ifelse(resultados$pobre_log >= rule, 1, 0)


cm_log <- confusionMatrix(data=factor(resultados$pred_log) , 
                          reference=factor(resultados$Pobre1) , 
                          mode="sens_spec" , positive="1")
cm_log






#Fuera de muestra
resultados2$pobre_log_test <- predict(logit , newdata=test_s , type="response")

rule <- 0.5
resultados2$pred_log_test <- ifelse(resultados2$pobre_log_test >= rule, 1, 0)


cm_log2 <- confusionMatrix(data=factor(resultados2$pred_log_test) , 
                           reference=factor(resultados2$Pobre1) , 
                           mode="sens_spec" , positive="1")
cm_log2


#Lo primero importante es conseguir un sensitivity alto, el del logit dio 0.53, specificity de 0.95 y accuracy de 0.86

Agregado_log <- cm_log$byClass[[1]]*0.75 +cm_log$byClass[[2]]*0.25 #0.63








ggplot(resultados_ridge, aes(x = Lambda, y = RMSE)) +
  geom_point() +
  geom_line() +
  theme_test() +
  scale_y_continuous(labels = scales::comma)


ggplot(resultados_ridge, aes(x = Lambda, y = R2_Score)) +
  geom_point() +
  geom_line() +
  theme_test() +
  scale_y_continuous(labels = scales::comma)

filtro <- resultados_ridge$RMSE == min(resultados_ridge$RMSE)
mejor_lambda_ridge <- resultados_ridge[filtro, "Lambda"]

resultados_ridge%>%head()

resultados_ridge$Lambda

# Guardamos el mejor Ridge
y_hat_in3<- predict.glmnet(modelo_ridge,
                           newx = as.matrix(X_train),
                           s = mejor_lambda_ridge)

y_pobre <- ifelse(y_hat_in3 <= predicciones$Lp, 1, 0)

predicciones$Predicho <- y_hat_in3


plot(predicciones$Predicho, predicciones$IngpcReal)

#Ideas sueltas:
#tal vez nos va mejor si predecimos el ingreso total, teniendo en cuenta el número de personas y de ahí se calcula el per cápita y la pobreza

#Cotiza o recibe pensión
ggplot(data = data, aes(x = factor(Pobre), y = prop_cotiza))+
  geom_boxplot()

ggplot(data = data, aes(x = factor(Pobre), y = prop_ocupados_pet))+
  geom_boxplot()

ggplot(data = data, aes(x = factor(Pobre), y = ppc))+
  geom_boxplot()

dm <- data%>%
  group_by(jefe_mujer, Pobre)%>%
  summarise(n = n())%>%
  mutate(perc = n/sum(n))

ggplot(data = dm, aes(x = factor(jefe_mujer), y = perc, fill = factor(Pobre)))+
  geom_bar(stat="identity", width = 0.7)


dc <- data%>%
  group_by(Clase, Pobre)%>%
  summarise(n = n())%>%
  mutate(perc = n/sum(n))

ggplot(data = dc, aes(x = factor(Clase), y = perc, fill = factor(Pobre)))+
  geom_bar(stat="identity", width = 0.7)

#Árboles de decisión y random forest



#Relación polinómica ingreso y continuas
plot(data$P6040,data$log_ingtot)
plot(data$prop_Desocupados_pet,data$log_ingtot)
plot(data$prop_cotiza,data$log_ingtot)
plot(data$años_educ_promedio,data$log_ingtot)
plot(data$Horas_Hogar,data$log_ingtot)


#prop cotiza
#horashogar
#añoseducpromedio



#Intento RF regression

