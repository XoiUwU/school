library(ISLR2)

xdata <- data.matrix(
    NYSE[, c("DJ_return", "log_volume", "log_volatility")]
)
istrain <- NYSE[, "train"]
xdata <- scale(xdata)

lagm <- function(x, k = 1){
    n <- nrow(x)
    pad <- matrix(NA, k, ncol(x))
    rbind(pad, x[1:(n - k), ])
}

arframe <- data.frame(log_volume = xdata[, "log_volume"], L1 = lagm(xdata, 1), L2 = lagm(xdata, 2), L3 = lagm(xdata, 3), L4 = lagm(xdata, 4), L5 = lagm(xdata, 5))
arframe <- arframe[-(1:5), ]
istrain <- istrain[-(1:5)]

arfit <- lm(log_volume ~ ., data = arframe[istrain, ])
arpred <- predict(arfit, arframe[!istrain, ])
V0 <- var(arframe[!istrain, "log_volume"])
wah <- 1 - mean((arpred - arframe[!istrain, "log_volume"])^2) / V0
print(wah)