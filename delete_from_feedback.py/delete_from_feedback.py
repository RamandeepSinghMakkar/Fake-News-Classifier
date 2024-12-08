import mysql.connector

# Establish connection
conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    database="fake_news"
)
cursor = conn.cursor()

# Clear feedback table and reset AUTO_INCREMENT
cursor.execute("DELETE FROM feedback;")
cursor.execute("ALTER TABLE feedback AUTO_INCREMENT = 1;")
conn.commit()

print("Feedback table cleared, AUTO_INCREMENT reset.")
cursor.close()
conn.close()
