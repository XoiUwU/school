USE BITS;
-- REVIEW 2a
CREATE VIEW TopLevelClient AS
SELECT ClientNum, ClientName, Street, Balance, CreditLimit
FROM Client
WHERE CreditLimit >= 10000;

-- REVIEW 2b
SELECT ClientNum, ClientName, CreditLimit - Balance AS Difference
FROM TopLevelClient;

-- REVIEW 2c
SELECT ClientNum, ClientName, CreditLimit - Balance AS Difference
FROM Client
WHERE CreditLimit >= 10000;

-- REVIEW 3a
CREATE VIEW ItemOrder AS
SELECT Tasks.TaskID, Tasks.Description, Tasks.Price, 
       OrderLine.OrderNum, WorkOrders.OrderDate, OrderLine.QuotedPrice
FROM OrderLine
JOIN Tasks ON OrderLine.TaskID = Tasks.TaskID
JOIN WorkOrders ON OrderLine.OrderNum = WorkOrders.OrderNum; 

-- REVIEW 3b
SELECT TaskID, Description, OrderNum, QuotedPrice
FROM ItemOrder
WHERE QuotedPrice > 100;

-- REVIEW 3c
SELECT Tasks.TaskID, Tasks.Description, OrderLine.OrderNum, OrderLine.QuotedPrice
FROM OrderLine
JOIN Tasks ON OrderLine.TaskID = Tasks.TaskID
JOIN WorkOrders ON OrderLine.OrderNum = WorkOrders.OrderNum
WHERE OrderLine.QuotedPrice > 100;

-- REVIEW 4
-- An index is just like an index you would find at the back of the book. 
-- It will tell you everywhere a specific item is mentioned so you can quickly jump to it
-- instead of having to scan the entire database or table for a few items.

-- CREATE INDEX index_1
-- ON table_1 (column1, column2);

-- REVIEW 18
-- YES, MarketPoint would be included in the query. Since TopLevelClient includes all clients with a credit limit of 
-- $10,000 or greater, MarketPoint would be included on the list.

USE Sports;
-- Sports 4a
CREATE INDEX PatientIndex1 ON Patient (City); 

-- Sports 4b
CREATE INDEX PatientIndex2 ON Patient (LastName); 

-- Sports 4c
CREATE INDEX PatientIndex3 ON Patient (City DESC); 

-- Sports 5
DROP INDEX PatientIndex3 ON Patient;



