USE BITS;
/*BITS 1*/
SELECT ClientName
FROM Client
WHERE CreditLimit < 10000;

/*BITS 3*/
SELECT OrderNum
FROM WorkOrders
WHERE ClientNum = '332' AND OrderDate = '2018-09-10';

/*BITS 4*/
SELECT WorkOrders.OrderDate, OrderLine.ScheduledDate
FROM WorkOrders
JOIN OrderLine ON WorkOrders.OrderNum = OrderLine.OrderNum
JOIN Tasks ON OrderLine.TaskID = Tasks.TaskID
WHERE Tasks.Description LIKE '%SA44%';

/*BITS 7*/
SELECT SUM(Balance)
FROM Client
JOIN Consultant ON Client.ConsltNum = Consultant.ConsltNum
WHERE Consultant.FirstName = 'Christopher' AND Consultant.LastName = 'Turner';

USE COLONIAL;
/*COLONIAL 4*/
SELECT TripName
FROM Trip
WHERE Type = 'Biking' AND Distance > 20;

/*COLONIAL 5*/
SELECT TripName
FROM Trip
WHERE State = 'VT' AND MaxGrpSize > 10;

/*COLONIAL 9*/
SELECT Trip.TripName
FROM Trip
JOIN TripGuides ON Trip.TripID = TripGuides.TripID
JOIN Guide ON TripGuides.GuideNum = Guide.GuideNum
WHERE Guide.FirstName = 'Rita' AND Guide.LastName = 'Boyers' AND Trip.Type = 'Biking';

/*COLONIAL 13*/
SELECT 
    C.FirstName, 
    C.LastName, 
    T.TripName, 
    T.Type
FROM 
    Customer C
JOIN 
    Reservation R ON C.CustomerNum = R.CustomerNum
JOIN 
    Trip T ON R.TripID = T.TripID
WHERE 
    C.CustomerNum IN (
        SELECT 
            CustomerNum
        FROM 
            Reservation
        GROUP BY 
            CustomerNum
        HAVING 
            COUNT(*) > 1
    );


USE Sports;
/*SPORTS 4*/
SELECT Therapies.Description
FROM Session
JOIN Therapies ON Session.TherapyCode = Therapies.TherapyCode
JOIN Therapist ON Session.TherapistID = Therapist.TherapistID
WHERE Therapist.FirstName = 'Steven' AND Therapist.LastName = 'Wilder';

/*SPORTS 6*/
SELECT Therapist.FirstName, Therapist.LastName
FROM Session
JOIN Patient ON Session.PatientNum = Patient.PatientNum
JOIN Therapist ON Session.TherapistID = Therapist.TherapistID
WHERE Patient.FirstName = 'Ben' AND Patient.LastName = 'Odepaul';


/*SPORTS 8*/
SELECT Therapies.Description
FROM Session
JOIN Therapies ON Session.TherapyCode = Therapies.TherapyCode
JOIN Patient ON Session.PatientNum = Patient.PatientNum
WHERE Patient.FirstName = 'Joseph' AND Patient.LastName = 'Baptist';


/*SPORTS 10*/
/*
I would place the hourly rate in the Therapists table.
I would also put a hours worked column in for easy payrole calculation.
It would also be beneficial to have a secure table with tax and banking information.
*/

/*REVIEW 7*/
/*
A primary key is a unique identifier for every record in a database. 
It allows every data entry to be uniqely identified and linked between different tables.
*/
/*
GuideNum
TripID
CustomerNum
ReservationID
TripID
*/