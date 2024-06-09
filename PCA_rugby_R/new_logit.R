library(FactorAssumptions)
library(factoextra)
library(ggplot2)
library(ggrepel)
library(corrplot)
library(dplyr)
library(pcaLogisticR)

rwc23_file_diff <- '../diff_df.csv'
rwc23_relative_file <- 'split_df_relative.csv'


data_diff <- read.csv(rwc23_file_diff)
variable_labels <-c('Game.ID', 'Date', 'Team')
variable_point_related <-c('Tries','Conversions','Goal.Kicks.Successful')
columns_to_remove <- c(variable_labels, variable_point_related, 'X')
#remove half
odd_indices <- seq(1, nrow(data_diff), by = 2)

# Keep only odd rows
odd_rows_df <- data_diff[odd_indices, ]

replace_na_with_mean <- function(x) {
  ifelse(is.na(x), mean(x, na.rm = TRUE), x)
}

odd_rows_df <- odd_rows_df %>%
  mutate_if(is.numeric, replace_na_with_mean)

#exlude some columns
columns_to_keep <- setdiff(names(odd_rows_df), columns_to_remove)

# Subset the data frame to keep only the desired columns
df_subset <- odd_rows_df[, columns_to_keep]
df_subset_outcome <- df_subset %>% select(-Score)
df_subset_scale <-  df_subset_outcome  %>%
  mutate_at(vars(-Outcome), ~ scale(.) %>% as.vector)
#
formula <- as.formula(paste("Outcome ~", paste(setdiff(names(df_subset_scale), "Outcome"), collapse = " + ")))
print(formula )
mylogit <- glm(formula, data = df_subset, family = binomial(link='logit'))
summary(mylogit )
#do logit
my_pcalogit <- caLogisticR(
  formula = formula,
  data = df_subset,
  n.pc = 7,
  scale = FALSE,
  center = FALSE,
  tol = 1e-04,
  max.pc = 11
)

#
pca_outcome = pcaLogisticR(
  formula = NULL,
  data = NULL,
  n.pc = 1,
  scale = FALSE,
  center = FALSE,
  tol = 1e-04,
  max.pc = NULL
)


predict.pcaLogisticR(
  object,
  newdata,
  type = c("class", "posterior", "pca.ind.coord", "all"),
  ...
)