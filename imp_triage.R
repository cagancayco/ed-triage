library(mice)
library(parallel)
library(readr)
library(dplyr)

df <- read_csv("NHAMCS_2012-2015_2018-04-09.csv")
df$X1 <- NULL

cl <- makeCluster(7)
clusterSetRNGStream(cl,9956)
clusterExport(cl,"df")
clusterEvalQ(cl,library(mice))

imp_pars <- parLapply(cl=cl, X=1:7, fun=function(no){mice(df, m=1)})
stopCluster(cl)
imp_merged <- imp_pars[[1]]
for (n in 2:length(imp_pars)){imp_merged <- ibind(imp_merged,imp_pars[[n]])}
completed_df <- complete(imp_merged)

write_csv(completed_df, "NHAMCS_2012-2015_2018-04-09_imp.csv")