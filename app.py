from flask import Flask, render_template, request, jsonify, url_for
import torch
import torch.nn as nn
import numpy as np
from our_model import OurModel
from sklearn.preprocessing import StandardScaler
from nba_data import get_player_stats,all_players
from jinja2 import Environment, FileSystemLoader


app = Flask(__name__)


model = OurModel()
model.load_state_dict(torch.load('best_model.pth'))
model.eval()
players = all_players()


def create_jinja_environment():
    env = Environment(loader=FileSystemLoader('templates'))
    env.globals.update(zip=zip, url_for=url_for)
    return env

app.jinja_env = create_jinja_environment()



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET'])
def predict():
    player_name = request.args.get('player_name')
    
    if not player_name:
        return render_template('index.html', error="Please enter a player name.")
    

    input_tensor,info = get_player_stats(player_name,"2023-24")
    
    with torch.no_grad():
        prediction = model(input_tensor)
    
    prediction_unnormalized = prediction.numpy()
    
    print(prediction_unnormalized)

    stat_names = ['Rebounds', 'Assists', 'Points']
    
    return render_template('player_stats.html', player_name=player_name, predicted_stats=prediction_unnormalized, stat_names=stat_names,previous_year_stats=info, enumerate = enumerate)


@app.route('/autocomplete', methods=['GET'])
def autocomplete():
    term = request.args.get('term')
    suggestions = [player['full_name'] for player in players if term.lower() in player['full_name'].lower()]
    return jsonify(suggestions)


if __name__ == '__main__':
    app.run(debug=True)
