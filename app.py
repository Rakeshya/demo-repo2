from flask import Flask, request, jsonify,render_template
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import  os
from flask_cors import CORS

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Load model and vectorizer
model = pickle.load(open("model3.pkl", "rb"))
tfidf = pickle.load(open("vectorizer.pkl", "rb"))

# Initialize stemmer
ps = PorterStemmer()


# Text preprocessing function
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words("english") and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

# ... rest of your code ...


# Define predict route

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data.get("text")  # use .get() to avoid KeyError
    if not text:
        return jsonify({"error": "Missing 'text' in request"})

    vect = vectorizer.transform([text])
    prediction = model.predict(vect)[0]
    return jsonify({"prediction": int(prediction)})


# Run the app
if __name__ == "__main__":
   if __name__ == '__main__':
    app.run(debug=True)


