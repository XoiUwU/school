USE school;

SHOW TABLES;

CREATE TABLE students(
age int,
height float,
city varchar(255)
);

DESCRIBE TABLE students;

SELECT *;

FROM school.students ;


INSERT INTO students (age, height, city) VALUES ('99', '5.10', 'St. Paul');

USE bit1;