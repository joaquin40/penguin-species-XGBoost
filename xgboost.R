## ---------------------------------------------------------------------
pacman::p_load(tidyverse, xgboost, palmerpenguins, DataExplorer, caret, groupdata2)


## ---------------------------------------------------------------------
data("penguins")


## ---------------------------------------------------------------------
penguins |> 
  plot_missing()


## ---------------------------------------------------------------------
df <- drop_na(penguins)
df <- select(df,-year)


## ---------------------------------------------------------------------
df |> plot_missing()


## ---------------------------------------------------------------------
penguins |> 
  ggplot(aes(species, fill = species)) +
  geom_bar() + 
   geom_text(
    aes(label = ..count..),
    stat = "count",
    vjust = -0.5,
    color = "black",
    size = 3
  ) + 
  theme(legend.position = "none")


## ---------------------------------------------------------------------
df2 <- groupdata2::downsample(df, cat_col = "species")


## ---------------------------------------------------------------------
df2 |> 
  ggplot(aes(species, fill = species)) +
  geom_bar() + 
  geom_text(aes(label = ..count..),
           stat = "count",
           vjust = -0.5,
           color = "black",
           size = 3)+
  theme(legend.position = "none")


## ---------------------------------------------------------------------
df3 <- model.matrix(~ . , -1, data = df2[,-1])[,-1] |> 
  data.frame() |> 
  cbind(df2[,1]) |> 
  rename( "species" = "df2[, 1]" )

df3 <- select(df3, species, everything())


## ---------------------------------------------------------------------
df3$species <- as.integer(df3$species) - 1
df3


## ---------------------------------------------------------------------
set.seed(1)
index <- createDataPartition(as.factor(df3$species), p = .8, list = FALSE)

train <- df3[index,]
test <- df3[-index,]


## ---------------------------------------------------------------------
test$species |> table()
train$species |> table()



## ---------------------------------------------------------------------
params <- list(
  objective = "multi:softmax",
  num_class = 3,
  max_depth = c(3, 6, 9),     # maximum depth
  eta = c(0.01,0.1),     # learning rate
  gamma = c(0, 0.1, 0.2),      # complexity of model pruning
  colsample_bytree = c(0.6, 0.8, 1), # number of columns to sample from each tree
  min_child_weight = c(1, 3, 5), # min number instance for each node
  subsample = c(0.6, 0.8, 1) # fraction pf obs to be randomly sample for each tree
)


## ---------------------------------------------------------------------
#apply(train,2,FUN = class)

fit_xgboost <- xgb.cv(
  params = params,
  data = as.matrix(train[, -1]),  # convert data to matrix format
  label = train[,1],              # target variable
  nfold = 10,
  nrounds = 100,        
  verbose = 0,          # disable verbose output
  early_stopping_rounds = 10
)


## ---------------------------------------------------------------------
fit_xgboost


## ---------------------------------------------------------------------
dtrain <- xgb.DMatrix(data = as.matrix(train[,-1]), label = train[,1])
dtest <- xgb.DMatrix(data = as.matrix(test[,-1]), label = test[,1])


## ---------------------------------------------------------------------
# fit_xgboost |> attributes()
# 
best_iteration <- which.min(fit_xgboost$best_iteration)

final_model <- xgboost(
  params = params,
  data = dtrain,
  nrounds = best_iteration
)

predictions <- predict(final_model, newdata = dtest)



## ---------------------------------------------------------------------
library(forcats)
pred_fct <- factor(predictions, labels = levels(df2$species))
test_fct <- factor(test$species, labels = levels(df2$species))


## ---------------------------------------------------------------------
cm <- table(pred = pred_fct, actual = test_fct)
confusionMatrix(cm)

