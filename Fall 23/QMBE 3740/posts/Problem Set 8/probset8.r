library(arules)

## P1 3A

# Load the dataset
groceries <- read.transactions("posts/Problem Set 8/groceries.csv", format = "basket", sep = ",")

# Find the item frequency
item_frequency_table <- itemFrequency(groceries, type = "absolute")
sorted_item_frequency <- sort(item_frequency_table, decreasing = FALSE)

# Get the 10 least frequently purchased items
least_frequent_items <- head(sorted_item_frequency, 10)

# Display the result
print(least_frequent_items)

## P1 3B

# Find rules with minimum length of 3
rules_min3 <- apriori(groceries, parameter = list(supp = 0.001, conf = 0.08, minlen = 3))
number_of_rules_min3 <- length(rules_min3)

# Find rules with minimum length of 4
rules_min4 <- apriori(groceries, parameter = list(supp = 0.001, conf = 0.08, minlen = 4))
number_of_rules_min4 <- length(rules_min4)

# Display the number of rules for each case
print(paste("Number of rules with minimum length of 3:", number_of_rules_min3))
print(paste("Number of rules with minimum length of 4:", number_of_rules_min4))

## P1 3C

# Find rules with a minimum length of 2
rules <- apriori(groceries, parameter = list(supp = 0.001, conf = 0.08, minlen = 2))

# Subset rules that involve "soda" or "whipped/sour cream"
rules_soda <- subset(rules, subset = items %in% "soda")
rules_cream <- subset(rules, subset = items %in% "whipped/sour cream")

# Combine the rules
combined_rules <- unique(c(rules_soda, rules_cream))

# Inspect only the first 3 rules
inspect(head(combined_rules, 3))



## PART 2

library(arules)
print("1")
# Read the transactions from the CSV file
transactions <- read.transactions("posts/Problem Set 8/Market_Basket_Optimisation.csv", format = "basket", sep = ",")

# Display the first few transactions
inspect(head(transactions))

print("2")
# Use summary to get information about the transactions
transaction_summary <- summary(transactions)

# Print the number of transactions
print(paste("Number of transactions:", transaction_summary@Dim[1]))

# Print the number of distinct items
print(paste("Number of distinct items:", transaction_summary@Dim[2]))

# Calculate the number of possible itemsets
number_of_items <- transaction_summary@Dim[2]
number_of_possible_itemsets <- 2^number_of_items - 1

# Print the number of possible itemsets
print(paste("Number of possible itemsets:", number_of_possible_itemsets))

# Calculate the number of distinct items
number_of_items <- transaction_summary@Dim[2]

# Calculate the number of possible itemsets
# For n items, the number of possible itemsets is 2^n - 1 (excluding the empty set)
number_of_possible_itemsets <- 2^number_of_items - 1

# Print the number of possible itemsets
print(paste("Number of possible itemsets:", number_of_possible_itemsets))


print("3")
library(arules)
library(ggplot2)

# Assuming 'transactions' is already loaded from the previous step

# Get the sizes of the transactions
transaction_sizes <- size(transactions)

# Create a dataframe for plotting
transaction_sizes_df <- data.frame(Size = transaction_sizes)

# Create a histogram of transaction sizes
ggplot(transaction_sizes_df, aes(x = Size)) +
  geom_histogram(binwidth = 1, fill = "blue", color = "black") +
  scale_x_continuous(breaks = seq(0, max(transaction_sizes_df$Size), by = 1)) +
  labs(title = "Distribution of Transaction Sizes", x = "Size of Transaction", y = "Frequency")

print("4")
# Get item frequencies
item_freq <- itemFrequency(transactions, type = "absolute")
item_freq_df <- data.frame(Item = names(item_freq), Frequency = item_freq)

# Sort the items by frequency
item_freq_df_sorted <- item_freq_df[order(item_freq_df$Frequency, decreasing = TRUE), ]

# Determine the ten most frequent items
top_10_items <- head(item_freq_df_sorted, 10)

# Determine the ten least frequent items
bottom_10_items <- tail(item_freq_df_sorted, 10)

# Print the ten most frequent items
print("The ten most frequent items are:")
print(top_10_items)

# Print the ten least frequent items
print("The ten least frequent items are:")
print(bottom_10_items)


print("5")
library(arules)

# Assuming 'transactions' is already loaded from the previous step

# Calculate item frequencies
item_freq <- itemFrequency(transactions)

# Use descriptive statistics to determine a reasonable support threshold
# We'll consider items that are more frequent than the 95th percentile
support_threshold <- quantile(item_freq, probs = 0.95)

# Generate association rules with the determined support threshold
rules <- apriori(transactions, parameter = list(supp = support_threshold, conf = 0.25, minlen = 2))

# Display a summary of the rules
summary(rules)


print("6")
# Assuming 'rules' is already generated from the previous step

# Count the number of association rules generated
number_of_rules <- length(rules)

# Determine different rule lengths and count rules for each length
rules_length_distribution <- table(size(rules))

# Get the top 12 rules by confidence
top12_confidence <- head(sort(rules, by = "confidence"), 12)

# Get the top 12 rules by lift
top12_lift <- head(sort(rules, by = "lift"), 12)

# Print the results
print(paste("Number of association rules generated:", number_of_rules))
print("Rules length distribution:")
print(rules_length_distribution)
print("Top 12 rules by confidence:")
inspect(top12_confidence)
print("Top 12 rules by lift:")
inspect(top12_lift)


print("7")
# Assuming 'rules' and 'transactions' are already loaded from the previous steps

# Identify the 6 most frequent items
item_freq <- itemFrequency(transactions, type = "absolute")
top6_items <- names(sort(item_freq, decreasing = TRUE)[1:6])

# Subset the rules to exclude the 6 most frequent items
rules_without_top6 <- subset(rules, !(lhs %in% top6_items | rhs %in% top6_items))

# Get the top 10 rules by lift, excluding the 6 most frequent items
top10_lift_excluding_top6 <- head(sort(rules_without_top6, by = "lift"), 10)

# Print the top 10 rules by lift, excluding the 6 most frequent items
print("Top 10 rules by lift, excluding the 6 most frequent items:")
inspect(top10_lift_excluding_top6)


print("8")

# Set the support and confidence thresholds
support_threshold <- 0.005
confidence_threshold <- 0.25

# Generate association rules with the determined support and confidence thresholds
rules <- apriori(transactions, parameter = list(supp = support_threshold, conf = confidence_threshold, minlen = 2))

# Display a summary of the rules
summary(rules)

# Inspect the top rules by lift
top_rules_by_lift <- sort(rules, by = "lift", decreasing = TRUE)
inspect(head(top_rules_by_lift, 5))
