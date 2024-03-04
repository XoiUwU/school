CREATE TABLE Teams (
    id VARCHAR(3) PRIMARY KEY,
    name VARCHAR(24)
);

CREATE TABLE GameMaps (
    id VARCHAR(3) PRIMARY KEY,
    name VARCHAR(24),
    mode VARCHAR(8)
);

CREATE TABLE Maps (
    id INT PRIMARY KEY,
    team1 VARCHAR(3),
    team2 VARCHAR(3),
    map VARCHAR(3),
    victory INT,
    defeat INT,
    FOREIGN KEY (team1) REFERENCES Teams(id),
    FOREIGN KEY (team2) REFERENCES Teams(id),
    FOREIGN KEY (map) REFERENCES GameMaps(id)
);

CREATE TABLE Players (
    id INT PRIMARY KEY,
    name VARCHAR(16),
    team VARCHAR(3),
    number INT,
    FOREIGN KEY (team) REFERENCES Teams(id)
);

CREATE TABLE Matches (
    id INT PRIMARY KEY,
    team1 VARCHAR(3),
    team2 VARCHAR(3),
    maps INT,
    win INT,
    victory INT,
    defeat INT,
    FOREIGN KEY (team1) REFERENCES Teams(id),
    FOREIGN KEY (team2) REFERENCES Teams(id),
    FOREIGN KEY (maps) REFERENCES Maps(id)
);
