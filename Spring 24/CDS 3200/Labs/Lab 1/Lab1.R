#---
#title: "Lab0: Intro to R"
#author: "Xander"
#date: "`r Sys.Date()`"
#output: openintro::lab_report
#---

#Before beginning this lab, you may want to read or skip Section 2.3 in the textbook. That section includes lots of helpful commands to complete the exercises in this lab.

#---

#
library(tidyverse)
library(openintro)
library(ISLR2)
#```

#This exercise relates to the College dataset, which is included in the ISLR2 R-package. It contains a number of variables for 777 different universities and colleges in the US. The variables are

#* Private : Public/private indicator
#* Apps : Number of applications received
#* Accept : Number of applications accepted
#* Enroll : Number of new students enrolled
#* Top10perc : New students from top 10% of high school class
#* Top25perc : New students from top 25% of high school class
#* F.Undergrad : Number of full-time undergraduates
#* P.Undergrad : Number of part-time undergraduates
#* Outstate : Out-of-state tuition
#* Room.Board : Room and board costs
#* Books : Estimated book costs
#* Personal : Estimated personal spending
#* PhD : Percent of faculty with Ph.D.'s
#* Terminal : Percent of faculty with terminal degree
#* S.F.Ratio : Student/Faculty ratio
#* perc.alumni : Percent of alumni who donate
#* Expend : Instructional expenditure per student
#* Grad.Rate : Graduation rate


### Exercise 1
#First, use the `glimpse()` and `head()` commands to see the variables and some sample values. How are these commands different? How are they the same?

#```{r}
glimpse(College)
head(College)
#```

#Next, use the `summary()` function to produce a numerical summary of the variables in the data set.

#```{r}
summary(College)
#```


### Exercise 2
#Use the `pairs()` function to produce a *scatterplot matrix* of the first ten columns or variables of the data. Recall that you can reference the first ten columns of a matrix A using `A[,1:10]`

#```{r, fig.width=12, fig.height=12}
pairs(College[, 1:10])
#```


### Exercise 3
#Produce side-by-side boxplots showing the distribution of numbers of students from out of state (`Outstate`) in both public and private institutions  (`Private`).

#There are two ways to complete this exercise:

#* You could use the `plot()` command as described on page 50.
#* You could use the `ggplot` package (part of `tidyverse`) and the `+ geom_boxplot()` layer. For more on `ggplot` see the [visualization cheat sheet](https://canvas.hamline.edu/courses/15410/modules/items/689958) on Canvas.

#```{r, fig.width=10, fig.height=8}
ex3 <- ggplot(College, aes(x = Private, y = Outstate)) +
  geom_boxplot()

print(ex3)
#```

### Exercise 4
#Create a new qualitative variable, call it `Elite`, by binning the `Top10perc` variable. We are going to divide universities into two groups based on whether or not the proportion of students coming from the top 10% of their high school classes exceeds 50%.

#The command shown here uses the syntax of the `tidyverse` package.

#```{r}
elite_college <- College %>%
  mutate(Elite = as_factor(ifelse(Top10perc > 50, "Yes", "No")))

head(elite_college)
#```


#Use the `summary()` function to see how many `Elite` universities there are.

#```{r}
summary(elite_college)
#```


#Produce side-by-side boxplots of `Outstate` versus `Elite`. See Exercise 3 for suggestions on how to proceed.

#```{r, fig.width=10, fig.height=8}
ex4 <- ggplot(elite_college, aes(x = Private, y = Outstate)) +
  geom_boxplot()

print(ex3)
#```


### Exercise 5
#Choose at least two quantitative variables in the data set. For each one, produce at least three histograms with differing number of bins. (In total, you should have at least 6 histograms.)

#There are two ways to complete this exercise:

#* You could use the `hist()` command as described on page 51.
#* You could use the `ggplot` package (part of `tidyverse`) and the `+ geom_histogram()` layer. For more on `ggplot` see the [visualization cheat sheet](https://canvas.hamline.edu/courses/15410/modules/items/689958) on Canvas.

#```{r, fig.width=10, fig.height=8}
outstate_hist_10 <- ggplot(College, aes(x = Outstate)) +
  geom_histogram(bins = 10, fill = "blue", color = "black") +
  ggtitle("Outstate Tuition - 10 Bins")

outstate_hist_20 <- ggplot(College, aes(x = Outstate)) +
  geom_histogram(bins = 20, fill = "green", color = "black") +
  ggtitle("Outstate Tuition - 20 Bins")

outstate_hist_30 <- ggplot(College, aes(x = Outstate)) +
  geom_histogram(bins = 30, fill = "red", color = "black") +
  ggtitle("Outstate Tuition - 30 Bins")

print(outstate_hist_10)
print(outstate_hist_20)
print(outstate_hist_30)

room_board_hist_10 <- ggplot(College, aes(x = Room.Board)) +
  geom_histogram(bins = 10, fill = "blue", color = "black") +
  ggtitle("Room and Board Costs - 10 Bins")

room_board_hist_20 <- ggplot(College, aes(x = Room.Board)) +
  geom_histogram(bins = 20, fill = "green", color = "black") +
  ggtitle("Room and Board Costs - 20 Bins")

room_board_hist_30 <- ggplot(College, aes(x = Room.Board)) +
  geom_histogram(bins = 30, fill = "red", color = "black") +
  ggtitle("Room and Board Costs - 30 Bins")

print(room_board_hist_10)
print(room_board_hist_20)
print(room_board_hist_30)
#```


### Exercise 6

#Using your *scatterplot matrix* from Exercise 2, determine which variable is the strongest predictor for the number of enrolled students (`Enroll`). Create a larger scatterplot showing this relationship and explain why what this means in the context of the data.

#There are two ways to complete the plotting in this exercise:

#* You could use the `plot()` command as described on page 50.
#* You could use the `ggplot` package (part of `tidyverse`) and the `+ geom_point()` layer. For more on `ggplot` see the [visualization cheat sheet](https://canvas.hamline.edu/courses/15410/modules/items/689958) on Canvas.

### Exercise 7

#Examine another relationship between two variables that interest you, similar to what you did in Exercise 6. 

#```{r}

#```

#Continue exploring the data, and provide a brief summary of what you discover.
