DROP TABLE IF EXISTS detections;

CREATE TABLE detections (
    id INT AUTO_INCREMENT PRIMARY KEY,
    image_path VARCHAR(255) NOT NULL,
    label VARCHAR(100) NOT NULL,
    confidence FLOAT NOT NULL,
    timestamp VARCHAR(50) NOT NULL
);

