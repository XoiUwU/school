SELECT ClientNum, ClientName FROM BITS.Client;

SELECT * FROM BITS.Tasks;


SELECT TripName
FROM COLONIAL.Trip
WHERE Type = 'Hiking'
AND Distance > 6;


SELECT TripName
FROM COLONIAL.Trip
WHERE Type = 'Paddling'
OR StartLocation = 'VT';


SELECT LastName, FirstName
FROM Sports.Therapist
WHERE City = 'Palm Rivers';


SELECT LastName, FirstName
FROM Sports.Therapist
WHERE City != 'Palm Rivers';


SELECT PatientNum, FirstName, LastName
FROM Sports.Patient
WHERE Balance >= 3000;


SELECT PatientNum, LastName
FROM Sports.Patient
WHERE City IN ('Palm Rivers', 'Waterville', 'Munster');


SELECT PatientNum, LastName
FROM Sports.Patient
WHERE City IN ('Palm Rivers', 'Waterville', 'Munster');
-- OR
SELECT PatientNum, LastName
FROM Sports.Patient
WHERE City = 'Palm Rivers' OR City = 'Waterville' OR City = 'Munster';


SELECT *
FROM Sports.Therapies
WHERE Description LIKE '%training%';


ALTER TABLE Sports.Therapies 
ADD column1 char(9);

ALTER TABLE Sports.Therapies
DROP COLUMN column1;

ALTER TABLE Sports.Therapies
RENAME TO Sports.Therapies;

