library(tidyverse)
library(ISLR2)
library(broom)

# May - Aug | {Temp}, {Temp, Solar.R}
temperature_model <- lm(data = airquality, Ozone ~ Temp)
temperature_solar_model <- lm(data = airquality, Ozone ~ Temp + Solar.R)

cat("Temperature RSS: ", sum(resid(temperature_model)^2), "\n")
# Temperature RSS:  64109.89
cat("Temp Solar.R RSS: ", sum(resid(temperature_solar_model)^2), "\n")
# Temp Solar.R RSS:  59644.36

view(glance(temperature_model))
# r.squared: 0.4877
# adj.r.squared: 0.4832
view(glance(temperature_solar_model))
# r.squared: 0.5103
# adj.r.squared: 0.5012

#singles
wind_model <- lm(data = airquality, Ozone ~ Wind)
solar_model <- lm(data = airquality, Ozone ~ Solar.R)

cat("Wind RSS: ", sum(resid(wind_model)^2), "\n")
cat("Solar Radiation RSS: ", sum(resid(solar_model)^2), "\n")


#doubles
wind_temp_model <- lm(data = airquality, Ozone ~ Wind + Temp)
solar_wind_model <- lm(data = airquality, Ozone ~ Solar.R + Wind)

cat("Wind Temperature RSS: ", sum(resid(wind_temp_model)^2), "\n")
cat("Solar Radiation Wind RSS: ", sum(resid(solar_wind_model)^2), "\n")

#full
full_model <- lm(data = airquality, Ozone ~ Wind + Temp + Solar.R)

cat("Full Model RSS: ", sum(resid(full_model)^2), "\n")

