#QMBE 1100-1 Intro to R
#Assignment 5
#Xander Chapman
library(tidyverse)
#R1
#a
flights <- nycflights13::flights
summary(flights$arr_delay)
# min = -86
# max = 1272

#b
ggplot(flights) + geom_histogram(aes(arr_delay))

#c
flights300 <- flights %>% filter(arr_delay>300)
ggplot(flights300) + geom_histogram(aes(arr_delay))

#d

quantile(flights$arr_delay, seq(0,1,1/10), na.rm = TRUE)
###0%  10%  20%  30%  40%  50%  60%  70%  80%  90% 100% 
###-86  -26  -19  -14  -10   -5    1    9   21   52 1272 

#e
#These things teach us a lot about the data
#When looking at the deciles, we can tell that most of the flights arrive within 20 minutes of their scheduled time.
#This is because from 20% to 80%, the values are within (-19, 21]. 
#Looking at our final bin (52, 1272], we can tell that there is some outlier data but it does not make up much of the dataset.

#R2
library(forcats)
#a
Marital <- fct_collapse(gss_cat$marital, NMarried = c("Separated", "Widowed", "Divorced", "Never married"))
fct_count(Marital)

#b
gss_cat <- gss_cat %>% mutate(marital = Marital)
maritalovertime <- gss_cat %>% count(marital, year)
View(maritalovertime)

#c
ggplot(maritalovertime) + geom_line(aes(x=year, y=n, group=marital, color=marital))
