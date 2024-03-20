
-- BITS 3
USE BITS;


SELECT ClientNum, ClientName
FROM Client
WHERE ConsltNum = '22';

-- BITS 13

SELECT 
    c.LastName AS ConsultantLastName,
    AVG(cl.Balance) AS AverageClientBalance,
    COUNT(cl.ClientNum) AS NumberOfClients
FROM 
    Consultant c
JOIN 
    Client cl ON c.ConsltNum = cl.ConsltNum
GROUP BY 
    c.LastName
ORDER BY 
    c.LastName;


-- Colonial 7
USE CoLONIAL;

SELECT TripName, State
FROM Trip
WHERE Season = 'Summer'
ORDER BY State, TripName;

-- Colonial 12

SELECT 
    T.TripName, 
    G.FirstName AS GuideFirstName, 
    G.LastName AS GuideLastName
FROM 
    Trip T
JOIN 
    TripGuides TG ON T.TripID = TG.TripID
JOIN 
    Guide G ON TG.GuideNum = G.GuideNum
WHERE 
    T.State = 'NH'
ORDER BY 
    G.LastName, T.TripName;

-- Colonial 14

SELECT 
    R.ReservationID,
    T.TripName,
    C.LastName AS CustomerLastName,
    C.FirstName AS CustomerFirstName,
    (R.TripPrice + R.OtherFees) * R.NumPersons AS TotalCost
FROM 
    Reservation R
JOIN 
    Trip T ON R.TripID = T.TripID
JOIN 
    Customer C ON R.CustomerNum = C.CustomerNum
WHERE 
    R.NumPersons > 4
ORDER BY 
    R.ReservationID;

-- Sports 15
USE Sports;

SELECT 
    CONCAT(Th.FirstName, ' ', Th.LastName) AS TherapistName
FROM 
    Session S
JOIN 
    Therapist Th ON S.TherapistID = Th.TherapistID
JOIN 
    Therapies T ON S.TherapyCode = T.TherapyCode
WHERE 
    T.Description LIKE '%Massage%' OR T.Description LIKE '%Whirlpool%'
GROUP BY 
    Th.FirstName, Th.LastName
ORDER BY 
    Th.LastName, Th.FirstName;
