#load data
#notes for game daat =look at sequence without interuptions instead of sequences in isolation
library(FactorAssumptions)
library(factoextra)
library(ggplot2)
library(ggrepel)
library(corrplot)
rwc23_file <- 'data/team_split.csv'
data <- read.csv(rwc23_file)
#Get Started

offensive_metrics <- c(
                       'Possession...Overall',
                       'Possession...1st.half',
                       'Possession...2nd.half',
                       'Tries',
                       'Conversions',
                       'Penalties',
                       'Metres',
                       'Defenders.Beaten',
                       'Clean.breaks',
                       'Gain.line.Carries',
                       'Passes',
                       'Offloads',
                       'Kicks.from.Hand',
                       'Rucks.Won',
                       'Rucks.Successful',
                       'Total.Rucks',
                       'Lineouts.Successful',
                       'Lineouts.Thrown',
                       'Scrums.Successful',
                       'Scrums.Total'
                       )
                       
                       
                       
defensive_metrics <-c('Turnovers.won',
                      'Tackles.Made',
                      'Tackles.Missed',
                      'Goal.Kicks.Successful',
                      'Goal.Kicks.Attempted',
                      'Penalties.Conceded.Opp.Half',
                      'Penalties.Conceded.Own.Half',
                      'Yellow.Cards',
                      'Red.Cards'
              
                      )

#perform Bartlett's test of sphericity
bart <- function(dat){ #dat is your raw data
  R <- cor(dat)
  p <- ncol(dat)
  n <- nrow(dat)
  chi2 <- -((n - 1)-((2 * p)+ 5)/ 6) * log(det(R)) #this is the formula
  df <- (p * (p - 1)/ 2)
  crit <- qchisq(.95, df) #critical value
  p <- pchisq(chi2, df, lower.tail = F) #pvalue
  cat("Bartlett's test: X2(",
      df,") = ", chi2,", p = ",
      round(p, 3), sep="" )   
}



data.offense = data[,offensive_metrics]
data.defense = data[,defensive_metrics]


do_PCA_princomp<-function(data)
##create 2 plots, 2 second with teams as label, return the first component
{
  #cortest.bartlett(data)
  #KMO(data)
  res.pca <- princomp(data, cor = TRUE, scores = TRUE, )
  print(res.pca)
  return (res)
}

do_PCA_prcomp<-function(data)
  ##create 2 plots, 2 second with teams as label, return the first component
{
  #cortest.bartlett(data)
  #KMO(data)
  res.pca <- prcomp(data, center = TRUE, scale. = TRUE)
  #x[, 1]
  return (res.pca)
}


visualize_pca<-function(res)
{
  var <- get_pca_var(res)
  fviz_eig(res)
  fviz_pca_ind(res,
               label = "none",
               col.ind = "cos2", # Color by the quality of representation
               gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"),
               repel = TRUE# Avoid text overlapping
  )
  fviz_pca_var(res, col.var = "black")
  corrplot(var$cos2, is.corr=FALSE)
}

offense_pca <- do_PCA_prcomp(data.offense)
defense_pca <- do_PCA_prcomp(data.defense)
visualize_pca(defense_pca)
visualize_pca(offense_pca)

#create matrix with 1 component of offense and defense, add Team 
data$offense_PCA = offense_pca$x[,1]
data$defense_PCA = defense_pca$x[,1]

ggplot(data = data,
       aes(x = offense_PCA, y = defense_PCA, color = Team)) +
  geom_point() +
  ggtitle("Offense vs Defense Component Projection") +
  geom_label_repel(aes(label = Team),
                   box.padding = 0.35,
                   point.padding = 0.5,
                   segment.color = 'grey50') +
  theme_classic()

