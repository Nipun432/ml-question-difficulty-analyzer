import pickle
from flask import Flask, render_template, request

app = Flask(__name__)

# Load model
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
le = pickle.load(open("label_encoder.pkl", "rb"))

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        question = request.form["question"]

        q_vector = vectorizer.transform([question])
        prediction = model.predict(q_vector)
        probability = model.predict_proba(q_vector)

        difficulty = le.inverse_transform(prediction)[0]
        confidence = round(max(probability[0]) * 100, 2)

        # Marks distribution
        if difficulty == "Easy":
            dist = "Full: 70% | Partial: 20% | Low: 10%"
        elif difficulty == "Medium":
            dist = "Full: 30% | Partial: 50% | Low: 20%"
        else:
            dist = "Full: 10% | Partial: 40% | Low: 50%"

        # 🎨 Color based on difficulty
        if difficulty == "Easy":
            color = "#22c55e"   # green
        elif difficulty == "Medium":
            color = "#facc15"   # yellow
        else:
            color = "#ef4444"   # red

        # 🧠 Reason logic
        reason = "Based on detected keywords and question complexity."

        keywords_easy = ["define", "what is", "list"]
        keywords_medium = ["explain", "describe", "compare"]
        keywords_hard = ["derive", "prove", "analyze", "design"]

        q_lower = question.lower()

        for word in keywords_hard:
            if word in q_lower:
                reason = f"Contains advanced keyword '{word}' → Hard level"
                break
        for word in keywords_medium:
            if word in q_lower:
                reason = f"Requires explanation ('{word}') → Medium level"
                break
        for word in keywords_easy:
            if word in q_lower:
                reason = f"Basic concept question ('{word}') → Easy level"
                break

        return render_template("index.html",
                               result=True,
                               question=question,
                               difficulty=difficulty,
                               confidence=confidence,
                               dist=dist,
                               color=color,
                               reason=reason)

    return render_template("index.html", result=False)

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
