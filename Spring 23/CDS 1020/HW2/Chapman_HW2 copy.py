import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
import math
from scipy import stats

## Q1 ##
mu, sigma = 0, 1
randomgaus = np.random.normal(mu, sigma, 100000)
print(abs(mu - np.mean(randomgaus)))
print(abs(sigma - np.std(randomgaus, ddof=1)))
plt.hist(randomgaus, 30, density=True)
plt.show()

## Q2 ##
NHANES = pd.read_csv('HW2/NHANES.csv', na_values=['.'])
print(NHANES)
print(NHANES.describe())
#There are 10,000 participants (rows)
print(NHANES.dtypes.value_counts())
#float64    43
#object     31
#int64       4
#dtype: int64

## Q3 ##
SleepHrsNight = NHANES['SleepHrsNight']
print(SleepHrsNight.describe())
#mean        6.927531
#std         1.346729
#min         2.000000
#25%         6.000000
#50%         7.000000
#75%         8.000000
#max        12.000000
#iqr         2.000000

## Q4 ##
SleepHrsNight.plot(kind="box")
plt.show()
#The box plot is centered around 6-8 hours of sleep per night. 3 hours is the lower limit, 11 is the upper limit. There are some outliers at 12 and 2 hours.
#Most people in the dataset are getting 6-8 hours of sleep.


## Q5 ##

# 𝐻0: The general population gets less than 8 hours of sleep
# HA: the general population gets more than 8 hours of sleep
cleanSleepHrsNight = SleepHrsNight.dropna()
t_statistic, p_value = stats.ttest_1samp(cleanSleepHrsNight, popmean=6.927531)
alpha = 0.05
print(f'T Statistic: {t_statistic}')
print(f'P Value: {p_value:}')
if p_value <= alpha:
    print(f'Reject the null hpothesis (p-value={p_value:.3f})')
else:
    print(f'Fail to reject the null hypothesis (p-value={p_value:.3f})') 

## Q6 ##
sample_mean = np.mean(cleanSleepHrsNight)
sample_std = np.std(cleanSleepHrsNight)

z_value = (10 - sample_mean) / sample_std
print(f'z-value: {z_value:.3f}')
# The z-value is positive so we can determine that this participant gets more sleep than the average.
# Using a z-table we can determing that this participant sleeps for longer than 98.8% of all participants.

## Q7 ##
SleepStudy = pd.read_csv('HW2/SleepStudy.csv')
print(SleepStudy.describe())
AverageSleep = SleepStudy['AverageSleep']
print(AverageSleep.describe())
#mean       7.965929
#std        0.964840
#min        4.950000
#25%        7.430000
#50%        8.000000
#75%        8.590000
#max       10.620000
#iqr        1.160000


## Q8 ## 
SleepStudyAllNight = pd.get_dummies(SleepStudy, columns=['AllNighter'])
print(SleepStudyAllNight.describe())

## Q9 ##
#H0: The population proportion is equal to the sample proporiton (p = 0.134387)
#HA: The population proportion is not equal to the sample proportion (p =/= 0.134387)
alpha = 0.05
n= 253
pHat = 0.134387
s = 0.341744
SE = s / np.sqrt(n)
z_value = (pHat - 0.134387)/SE
print(f'z-value: {z_value:.3f}')
# The calculated z statistic falls within our -1.96 < z < 1.96 creitical value range.
# We fail to reject the null hypothesis and can conclude that there is not enough evidence to suggest the population proportion differs from the sample proportion

## Q10 ##
DepressionScore = SleepStudy['DepressionScore']
print(AverageSleep.describe())
print(DepressionScore.describe())

n1 = n2 = 253
mean1 = np.mean(DepressionScore)
mean2 = np.mean(AverageSleep)
std1 = np.std(DepressionScore)
std2 = np.std(AverageSleep)
print(f'depression mean{mean1}')
print(f'average mean{mean2}')
print(f'depression std{std1}')
print(f'average std{std2}')

se = np.sqrt((std1**2/n1) + (std2**2/n2))
df = n1 + n2 - 2
t_val = stats.t.ppf(0.975, df)
print(t_val)
print(se)

ci_lower = (mean1 - mean2) - t_val * se
ci_upper = (mean1 - mean2) + t_val * se

print(f'95% confidence interval for the difference between means: ({ci_lower:.3f}, {ci_upper:.3f})')
#The confidence interval does not contain zero so we can conclude there is evidence of a difference between the average DepressionScore and AverageSleep in the population at a confidence level of 0.95