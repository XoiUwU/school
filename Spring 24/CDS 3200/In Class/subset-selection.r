library(tidyverse)
library(ISLR2)
library(broom)

<<<<<<< Updated upstream
# May - Aug | {Temp}, {Temp, Solar.R}
temperature_model <- lm(data = airquality, Ozone ~ Temp)
temperature_solar_model <- lm(data = airquality, Ozone ~ Temp + Solar.R)

=======
#my assigned
temperature_model <- lm(data = airquality, Ozone ~ Temp)
temperature_solar_model <- lm(data = airquality, Ozone ~ Temp + Solar.R)
>>>>>>> Stashed changes
cat("Temperature RSS: ", sum(resid(temperature_model)^2), "\n")
# Temperature RSS:  64109.89
cat("Temp Solar.R RSS: ", sum(resid(temperature_solar_model)^2), "\n")
# Temp Solar.R RSS:  59644.36
<<<<<<< Updated upstream

=======
>>>>>>> Stashed changes
view(glance(temperature_model))
# r.squared: 0.4877
# adj.r.squared: 0.4832
view(glance(temperature_solar_model))
# r.squared: 0.5103
# adj.r.squared: 0.5012

<<<<<<< Updated upstream
#singles
wind_model <- lm(data = airquality, Ozone ~ Wind)
solar_model <- lm(data = airquality, Ozone ~ Solar.R)

cat("Wind RSS: ", sum(resid(wind_model)^2), "\n")
cat("Solar Radiation RSS: ", sum(resid(solar_model)^2), "\n")

=======
#others assigned
#singles
wind_model <- lm(data = airquality, Ozone ~ Wind)
solar_model <- lm(data = airquality, Ozone ~ Solar.R)
cat("Wind RSS: ", sum(resid(wind_model)^2), "\n")
# RSS: 79,859
cat("Solar Radiation RSS: ", sum(resid(solar_model)^2), "\n")
# RSS: 107,022
view(glance(wind_model))
# r.squared: 0.3618
# adj.r.squared: 0.3562
view(glance(solar_model))
# r.squared: 0.1213
# adj.r.squared: 0.1132
>>>>>>> Stashed changes

#doubles
wind_temp_model <- lm(data = airquality, Ozone ~ Wind + Temp)
solar_wind_model <- lm(data = airquality, Ozone ~ Solar.R + Wind)
<<<<<<< Updated upstream

cat("Wind Temperature RSS: ", sum(resid(wind_temp_model)^2), "\n")
cat("Solar Radiation Wind RSS: ", sum(resid(solar_wind_model)^2), "\n")

#full
full_model <- lm(data = airquality, Ozone ~ Wind + Temp + Solar.R)

cat("Full Model RSS: ", sum(resid(full_model)^2), "\n")

=======
cat("Wind Temperature RSS: ", sum(resid(wind_temp_model)^2), "\n")
# RSS: 53,973
cat("Solar Radiation Wind RSS: ", sum(resid(solar_wind_model)^2), "\n")
# RSS: 67,052
view(glance(wind_temp_model))
# r.squared: 0.5687
# adj.r.squared: 0.5611
view(glance(solar_wind_model))
# r.squared: 0.4495
# adj.r.squared: 0.4393

#full
full_model <- lm(data = airquality, Ozone ~ Wind + Temp + Solar.R)
cat("Full Model RSS: ", sum(resid(full_model)^2), "\n")
# RSS: 48003
view(glance(full_model))
# r.squared: 0.606
# adj.r.squared: 0.595

# Best Subset Slection [6.1.1]
# Consider all possible subsets of predictors.
# Choose the one with the best fitting model.
# This is what we did for `airquality` for seven subsets of predictors we computed RSS, R^2, and R^2_adj.
# We left out the Null Model whihc would have made 8 total modls
# There were 8 = 2^3 models to train and assess.

# Best Subset Selection
# Data has P predictors, and n observations
#1. Let M_0 be the null model
#2. For k=1,2,...,p
#a. Fit P choose k models.
#b. Choose best R^2 or RSS. Call it M_k
#3. Choose final model from M_0, M_1, ..., M_p using R^2_adj or [critera discussed later]

# Step-wise Selection [6.1.2]
# Only compute ~= P^2 models
# Example: p = 50 predictors
# 2^50 = 1.12*10^15 vs. 50^2 = 2500
# Intentionally add/remove predictors
>>>>>>> Stashed changes
