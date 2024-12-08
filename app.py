from flask import Flask, request, render_template, session
import joblib
import torch
import torch.nn as nn
import mysql.connector
from sklearn.feature_extraction.text import TfidfVectorizer

# Define the PyTorch model
class FakeNewsModel(nn.Module):
    def __init__(self, input_dim):
        super(FakeNewsModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return self.sigmoid(x)

# Load the trained PyTorch model
model = FakeNewsModel(input_dim=5000)
model.load_state_dict(torch.load('fake_news_model.pth'))
model.eval()

# Load the TfidfVectorizer
vectorizer = joblib.load('vectorizer.pkl')

# MySQL connection function
def get_db_connection():
    try:
        conn = mysql.connector.connect(
            host="localhost",
            user="root",
            password="",
            database="fake_news"
        )
        return conn
    except mysql.connector.Error as err:
        print(f"Error: {err}")
        return None

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get user input
    user_input = request.form['news_text']
    
    # Preprocess and transform the input using TfidfVectorizer
    transformed_input = vectorizer.transform([user_input]).toarray()
    transformed_input = torch.tensor(transformed_input, dtype=torch.float32)
    
    # Predict using the model
    with torch.no_grad():
        prediction = model(transformed_input)
    
    # Determine if the news is real or fake
    result = "Real" if prediction.item() < 0.5 else "Fake"
    
    # Save prediction to database
    conn = get_db_connection()
    if conn:
        cursor = conn.cursor()
        try:
            cursor.execute(
                "INSERT INTO predictions (news_text, prediction) VALUES (%s, %s)",
                (user_input, result)
            )
            conn.commit()
        except mysql.connector.Error as err:
            print(f"Database Error: {err}")
        finally:
            cursor.close()
            conn.close()
    
    return render_template('index.html', prediction_text=f'This news is likely {result}.')

@app.route('/query', methods=['GET', 'POST'])
def query():
    if request.method == 'POST':
        user_query = request.form['query_text']
        
        # Save query to database
        conn = get_db_connection()
        response_text = "Your query has been recorded. We'll get back to you!"
        if conn:
            cursor = conn.cursor()
            try:
                cursor.execute(
                    "INSERT INTO queries (user_query) VALUES (%s)",
                    (user_query,)
                )
                conn.commit()
            except mysql.connector.Error as err:
                print(f"Database Error: {err}")
                response_text = "There was an error recording your query."
            finally:
                cursor.close()
                conn.close()
        
        return render_template('query.html', response_text=response_text)
    
    return render_template('query.html')


@app.route('/feedback', methods=['GET', 'POST'])
def feedback():
    if request.method == 'POST':
        user_feedback = request.form['user_feedback']
        user_id = session.get('user_id', None)  # Default to None for anonymous feedback

        conn = get_db_connection()
        response_text = "Thank you for your feedback!"
        if conn:
            cursor = conn.cursor()
            try:
                # Validate user_id if present
                if user_id:
                    cursor.execute("SELECT 1 FROM users WHERE user_id = %s", (user_id,))
                    if cursor.fetchone() is None:
                        return render_template('feedback.html', response_text="Invalid user. Please log in.")

                # Insert feedback with user_id or None
                cursor.execute(
                    "INSERT INTO feedback (user_id, feedback_text) VALUES (%s, %s)",
                    (user_id, user_feedback)
                )
                conn.commit()
            except mysql.connector.Error as err:
                print(f"Database Error in /feedback: {err}")
                response_text = "There was an error saving your feedback. Please try again later."
            finally:
                cursor.close()
                conn.close()
        else:
            response_text = "Database connection error. Please try again later."

        return render_template('feedback.html', response_text=response_text)

    return render_template('feedback.html')

if __name__ == "__main__":
    app.run(debug=True)
