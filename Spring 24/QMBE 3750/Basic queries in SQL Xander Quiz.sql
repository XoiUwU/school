USE BITS;


SELECT ClientNum
FROM Client
UNION
SELECT ClientNum
FROM WorkOrders;


SELECT ClientNum, ClientName, ZipCode,
  CASE 
    WHEN ZipCode = '12345' THEN 'Region1'
    WHEN ZipCode = '67890' THEN 'Region2'
    ELSE 'Other'
  END AS Region
FROM Client;


SELECT COUNT(DISTINCT ConsltNum) AS TotalDistinctConsultants
FROM Consultant;


SELECT *
FROM WorkOrders
JOIN OrderLine USING (OrderNum);


SELECT * FROM Tasks;


