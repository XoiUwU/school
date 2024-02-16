library(tidyverse)
library(GGally)

# a

data <- data.frame(x1 = c(92, 60, 100), x2 = c(60, 30, 70))

a <- ggplot(data, aes(x = x1, y = x2)) +
  geom_point()

print(a)

# b

x1 <- c(92, 60, 100)
x2 <- c(60, 30, 70)

var_x1 <- var(x1)
var_x2 <- var(x2)

print(var_x1)
print(var_x2)

# c

c <- cov(x1, x2)

print(c)

gg <- ggpairs(data = data)
print(gg)