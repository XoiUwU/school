USE BITS;
SELECT COUNT(*) FROM Client WHERE CreditLimit = 10000;
SELECT ClientName, (CreditLimit - Balance) AS RemainingCredit FROM Client;
SELECT * FROM Tasks ORDER BY Category, Price;

USE COLONIAL;
SELECT State, COUNT(*) AS NumberOfTrips
FROM Trip
GROUP BY State;

SELECT Reservation.ReservationID, Customer.LastName, Trip.TripName
FROM Reservation
JOIN Customer ON Reservation.CustomerNum = Customer.CustomerNum
JOIN Trip ON Reservation.TripID = Trip.TripID
WHERE Reservation.NumPersons > 4;

USE Sports;
SELECT TherapyCode, Description
FROM Therapies
WHERE UnitOfTime = 15
ORDER BY Description;

SELECT AVG(LengthOfSession) AS AverageSessionTime
FROM Session
WHERE MONTH(SessionDate) = 10;