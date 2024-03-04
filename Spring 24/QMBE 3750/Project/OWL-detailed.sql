CREATE DATABASE OWL;
USE OWL;

-- Teams Table
CREATE TABLE Teams (
    id VARCHAR(3) PRIMARY KEY,
    name VARCHAR(24) NOT NULL
);

-- GameMaps Table
CREATE TABLE GameMaps (
    id VARCHAR(3) PRIMARY KEY,
    name VARCHAR(24) NOT NULL,
    mode VARCHAR(8) NOT NULL
);

-- Maps Table
-- Adjusting victory and defeat to reflect they reference team IDs
CREATE TABLE Maps (
    id INT PRIMARY KEY,
    team1 VARCHAR(3) NOT NULL,
    team2 VARCHAR(3) NOT NULL,
    map VARCHAR(3) NOT NULL,
    victory VARCHAR(3),
    defeat VARCHAR(3),
    FOREIGN KEY (team1) REFERENCES Teams(id),
    FOREIGN KEY (team2) REFERENCES Teams(id),
    FOREIGN KEY (victory) REFERENCES Teams(id),
    FOREIGN KEY (defeat) REFERENCES Teams(id),
    FOREIGN KEY (map) REFERENCES GameMaps(id)
);

-- Players Table
CREATE TABLE Players (
    id INT PRIMARY KEY,
    name VARCHAR(16) NOT NULL,
    team VARCHAR(3) NOT NULL,
    number INT NOT NULL,
    FOREIGN KEY (team) REFERENCES Teams(id)
);

-- Matches Table
-- Reflecting outcome with reference to winning and losing teams directly
CREATE TABLE Matches (
    id INT PRIMARY KEY,
    team1 VARCHAR(3) NOT NULL,
    team2 VARCHAR(3) NOT NULL,
    map INT NOT NULL,
    victory VARCHAR(3),
    defeat VARCHAR(3),
    FOREIGN KEY (team1) REFERENCES Teams(id),
    FOREIGN KEY (team2) REFERENCES Teams(id),
    FOREIGN KEY (victory) REFERENCES Teams(id),
    FOREIGN KEY (defeat) REFERENCES Teams(id),
    FOREIGN KEY (map) REFERENCES Maps(id)
);
