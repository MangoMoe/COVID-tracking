list.of.packages <- c("stats","ggplot2", "Rcpp", "grf", "caret", "mltools", "rpart", "minpack.lm", "doParallel", "rattle", "anytime","rlist")
list.of.packages <- c(list.of.packages, "zoo", "dtw", "foreach", "evaluate","rlist","hash", "e1071")

new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
if(length(new.packages)) install.packages(new.packages, repos = "http://cran.us.r-project.org")

update.packages(checkBuilt=TRUE, repos = "http://cran.us.r-project.org")

lapply(list.of.packages, require, character.only = TRUE)