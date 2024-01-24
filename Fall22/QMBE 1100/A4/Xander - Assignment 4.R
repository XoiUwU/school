# in case not already loaded
library(tidyverse)

# R1
# creates a tibble named eye_colors with 2 variables (eye_color and n(count))
eye_colors <- starwars %>% count(eye_color)
#plots a bar graph of eye_colors with eye_color on the x axis and n on the y axis
ggplot(eye_colors)+
 geom_col(mapping = aes(eye_color,n))

#R2
#a
#creates a histogram showing the distribution of the height of characters
#this omits 6 rows that do not have a height value recorded
ggplot(starwars, aes(x=height))+
 geom_histogram(binwidth = 8)

#b
#The species that has the widest variance in height is 'Droid'
ggplot(starwars, aes(x=height, y=species))+
 geom_boxplot()

#R3
#a
ggplot(mtcars, aes(hp, mpg))+
 geom_point()

#b
ggplot(mtcars, aes(x=hp, y=mpg))+
 geom_point()+
 geom_smooth(method=lm)

#c
ggplot(mapping = aes(x=hp, y=mpg, color=am))+
  geom_point(data=subset(mtcars, am==0), mapping = aes(hp, mpg))+
  geom_smooth(data=subset(mtcars, am==0), method=lm, se=FALSE)+
  geom_point(data=subset(mtcars, am==1), mapping = aes(hp, mpg))+
  geom_smooth(data=subset(mtcars, am==1), method=lm, se=FALSE)

#d
ggplot(mtcars, aes(hp,mpg))+
  geom_point()+
  facet_wrap(vars(am))+
  geom_smooth(method=lm)

#e
#I personally find that plot C is easier to interpret. It tells you right away that am type 0 has less mpg and hp than am type 1.

