#load data
rwc23_file <- 'data/team_split.csv'
data <- read.csv(rwc23_file)
#Get Started
offensive_metrics <- c('Possession - Overall',
                       'Possession - 1st half',
                       'Possession - 2nd half',
                       'Tries',
                       'Conversions',
                       'Penalties',
                       'Metres',
                       'Defenders Beaten',
                       'Clean breaks',
                       'Gain line Carries',
                       'Passes',
                       'Offloads',
                       'Kicks from Hand',
                       'Rucks Won',
                       'Rucks Successful',
                       'Total Rucks',
                       'Lineouts Successful',
                       'Lineouts Thrown',
                       'Scrums Successful',
                       'Scrums Total',
                       
                       
                       )
                       
                       
                       
defensive_metrics <-c('Turnovers won',
                      'Tackles Made',
                      'Tackles Missed',
                      'Goal Kicks Successful',
                      'Goal Kicks Attempted',
                      'Penalties Conceded Opp Half',
                      'Penalties Conceded Own Half',
                      'Yellow Cards',
                      'Red Cards'
                      
                      )                       
                       
