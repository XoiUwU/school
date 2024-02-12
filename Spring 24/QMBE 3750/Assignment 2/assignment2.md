# QMBE 3750 Assignment 2

Xander Chapman

## CDM Chapter 3 Exercises

### Review

-   What is the purpose of the WHERE clause in SQL? Which comparison operators can you use in a WHERE clause?

The WHERE clause is used to filter records based on conditions.

-   What is a computed field? How can you use one in an SQL query? How do you assign a name to a computed field?

A computed field is a column that is calculated on the fly by an SQL query.

-   How do you use the LIKE operator in an SQL query?

The LIKE operator is used to find patterns in columns. It is commonly used with wildcards `%` and `_`.

-   How do you use the IN operator in an SQL query?

The IN operator is used to specify multiple values in a WHERE clause.

-   Why is the data type for the ZipCode field CHAR and not SMALLINT or INTEGER? Is the length of the field long enough? Why or why not?

For the SMALLINT and INTEGER fields, leading `0`s could be lost when not defined as a CHAR. For example, all the zip codes in Alaska, Connecticut, Massachusetts, Maine, New Hampshire, New Jersey, Puerto Rico, Rhode Island, Vermont, Virgin Islands, APO Europe, and FPO Europe start with `0`. This means that any zip code in these states would be missing characters in our database.

For ZIP+4, a `-` is included: `20500-0001`. Also internationally, alphabetical characters are used in mailing codes such as Canada and the United Kingdom.

-   You need to delete the OrderLine table from the BITS database. Will the following command work? Why or why not?

``` sql
DELETE
FROM OrderLine
```

To completely remove the OrderLine table from the BITS database we would use:

``` sql
DROP TABLE OrderLine;
```

A `;` is also missing from the original commands.

### BITS

-   List the number and name of all clients.

``` sql
SELECT ClientNum, ClientName FROM Client;
```

-   List the complete Tasks table.

``` sql
SELECT * FROM Tasks;
```

### Colonial

-   List the name of each trip that has the type Hiking and that has a distance of greater than six miles.

``` sql
SELECT TripName
FROM Trip
WHERE Type = 'Hiking'
AND Distance > 6;
```

-   List the name of each trip that has the type Paddling or that is located in Vermont (VT).

``` sql
SELECT TripName
FROM Trip
WHERE Type = 'Paddling'
OR Location = 'VT';
```

### Sports

-   List the last name and first name of every therapist located in Palm Rivers.

``` sql
SELECT LastName, FirstName
FROM Therapist
WHERE City = 'Palm Rivers';
```

-   List the last name and first name of every therapist not located in Palm Rivers.

``` sql
SELECT LastName, FirstName
FROM Therapist
WHERE City != 'Palm Rivers';
```

-   List the patient number, first name, and last name of every patient whose balance is greater than or equal to \$3,000.

``` sql
SELECT PatientNum, FirstName, LastName
FROM Patient
WHERE Balance >= 3000;
```

-   List the patient number and last name for all patients who live in Palm Rivers, Waterville, or Munster.

``` sql
SELECT PatientNum, LastName
FROM Patient
WHERE City IN ('Palm Rivers', 'Waterville', 'Munster');
```

-   There are two ways to create the query in Step 11. Write the SQL command that you used and then write the alternate command that also would obtain the correct result.

``` sql
SELECT PatientNum, LastName
FROM Patient
WHERE City IN ('Palm Rivers', 'Waterville', 'Munster');
-- OR
SELECT PatientNum, LastName
FROM Patient
WHERE column_name = 'Palm Rivers' OR column_name = 'Waterville' OR column_name = 'Munster';
```

-   What WHERE clause would you use if you wanted to find all therapies where the description included the word "training" anywhere in the Description field?

``` sql
SELECT *
FROM Therapies
WHERE Description LIKE '%training%';
```

## CDM Chapter 4 Exercises

### Review

-   How would you use SQL to change a table's structure? What general types of changes are possible? Which commands are used to implement these changes?

You would use the ALTER statement. This allows you to add, drop, rename, change constrants, and change the table name. For example:

``` sql
ALTER TABLE example 
ADD column1 char;

ALTER TABEL example
DROP COLUMN column1;

ALTER TABLE example
RENAME TO example2;
```