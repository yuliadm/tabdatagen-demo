from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import numpy as np
import random

app = Flask(__name__)

# Load dataset and compute real data statistics
data = pd.read_csv("brownies.csv")
FEATURE_COLS = ['Sugar', 'Butter', 'Chocolate', 'Flour', 'Eggs', 'Temp', 'Time']
INGREDIENT_COLS = ['Sugar', 'Butter', 'Chocolate', 'Flour', 'Eggs']
real_stats = data[FEATURE_COLS].describe().loc[['mean', 'std']]
THRESHOLD = -0.05  # reward score threshold for "winning"

# Define AI Agent
class AIAgent:
    def __init__(self, bounds):
        self.bounds = bounds

    def act(self):
        # Generate ingredient ratios that sum to 1
        ingredients = np.random.dirichlet(np.ones(len(INGREDIENT_COLS)))
        temp = random.uniform(self.bounds['Temp'][0], self.bounds['Temp'][1])
        time = random.uniform(self.bounds['Time'][0], self.bounds['Time'][1])
        return np.concatenate((ingredients, [temp, time]))

ai_agent = AIAgent(bounds={col: (data[col].min(), data[col].max()) for col in ['Temp', 'Time']})

# Reward function (inverse MSE to real distribution)
def reward_function(generated_row):
    diff = ((generated_row - real_stats.loc['mean']) / real_stats.loc['std']) ** 2
    return -diff.mean()

@app.route("/", methods=["GET", "POST"])
def index():
    message = ""
    player_score = None
    ai_score = None
    ai_values = {}

    if request.method == "POST":
        try:
            # Collect and validate human input
            human_values = [float(request.form[col]) for col in INGREDIENT_COLS]
            total = sum(human_values)

            if not all(0 <= val <= 1 for val in human_values):
                raise ValueError("Each ingredient must be between 0 and 1.")
            if abs(total - 1.0) > 1e-4:
                raise ValueError("Ingredient ratios must sum to 1.")

            temp = float(request.form['Temp'])
            time = float(request.form['Time'])

            human_row = np.array(human_values + [temp, time])
            player_score = reward_function(pd.Series(human_row, index=FEATURE_COLS))

            if player_score >= THRESHOLD:
                message = f"🎉 You win! Your score: {player_score:.4f}"
            else:
                # AI turn
                ai_row = ai_agent.act()
                ai_score = reward_function(pd.Series(ai_row, index=FEATURE_COLS))
                ai_values = dict(zip(FEATURE_COLS, ai_row))

                if ai_score >= THRESHOLD:
                    message = f"🤖 AI wins! Its score: {ai_score:.4f}"
                else:
                    message = f"Keep trying! Your score: {player_score:.4f}, AI score: {ai_score:.4f}"

        except ValueError as e:
            message = f"⚠️ Error: {e}"

    return render_template("index.html", message=message, player_score=player_score,
                           ai_score=ai_score, ai_values=ai_values)

if __name__ == "__main__":
    app.run(debug=True)
