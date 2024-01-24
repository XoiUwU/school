#QMBE 1100-1 Intro to R
#Assignment 3
#Xander Chapman

#R1
#a,b
library(readr)
alc_abuse <- read_delim("alc_abuse.txt", delim = "|", escape_double = FALSE, trim_ws = TRUE, skip = 18)
View(alc_abuse)

#a: 20 variables in the file
#b: 9822 rows of data

#c
summarise(alc_abuse,mean(married)) *100
#c: ~81.64% of the participants are married.

#R2
#Finds the mean of every column in mtcars
mtcars %>% map_dbl(mean)

#R3
#Groups all flights by carrier than orders the flights by distnace
View(arrange(flights, carrier, desc(distance)))


#R4
#Displays all the flights that arrived at least 1 hour late but did not depart late
View(filter(flights, arr_delay>=60, 0>=dep_delay))

#R5
#Finds all the flights that were scheduled to depart between 00:00AM and 04:00AM
View(filter(flights, between(sched_dep_time, 000, 400)))

#R6
#Finds the plane with the highest average delay time (with at least 20 flights recorded)
View(flights %>%
       filter(!is.na(arr_delay))%>%
       group_by(tailnum)%>%
       summarise(mean = mean(arr_delay), n = n())%>%
       filter( n > 20)%>%
       arrange(desc(mean)))




#R7
#Adds a personalized row of flight data to the top row of the tibble
flights %>% add_row(.before = 1, 
                    year = 2022,
                    month = 1,
                    day = 1,
                    dep_time = 500,
                    sched_dep_time = 459,
                    dep_delay = -1,
                    arr_time = 700,
                    sched_arr_time = 701,
                    arr_delay = 1,
                    carrier = "XC",
                    flight = 12344,
                    tailnum = "H3L10",
                    origin = "MSP",
                    dest = "LHR",
                    air_time = 181,
                    distance = 4001,
                    hour = 7,
                    minute = 0)