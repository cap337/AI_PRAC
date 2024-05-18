from nba_api.stats.endpoints import playercareerstats,commonplayerinfo
from nba_api.stats.static import players
import torch


def get_player_id(name):
    player = players.find_players_by_full_name(name)
    player_id = player[0]["id"]
    #print(player)
    return player_id


def height_to_inches(height_str):
        if isinstance(height_str, str):
            feet, inches = map(int, height_str.split('-'))
            return feet * 12 + inches 
        else: 
            return float('nan')


def dict_to_tensor(info):
    age = float(info['age'])
    experience = float(info['experience'])
    games = float(info['g'])
    fg_percent = float(info['fg_percent'])
    trb = float(info['trb'])
    ast = float(info['ast'])
    pts = float(info['pts'])
    
    # Assuming height is in the format "feet-inches" (e.g., "6-5")
   
    height = height_to_inches(info["height"])
    weight = float(info['weight'])
    
    # Assuming position is a string that needs to be converted to a numeric value
    # You can define a mapping for positions
    position_mapping = {
        "Guard": 4.0,
        "Forward": 1.0,
        "Center": 0.0,
        # Add other positions as needed
    }
    position = position_mapping.get(info['position'], 0.0)  # Default to 0.0 if position is not found

    # Creating a list of the values
    values = [age, experience, games, fg_percent, trb, ast, pts, height, weight, position]
    
    # Converting the list to a tensor
    tensor = torch.tensor(values, dtype=torch.float32)
    
    return tensor

# Example usage




def get_player_stats(name, year):

# Nikola Jokić
    pid = get_player_id(name)
    career = playercareerstats.PlayerCareerStats(player_id=pid, per_mode36= "PerGame") 
    career = career.get_dict()
    common = commonplayerinfo.CommonPlayerInfo(player_id=pid)
    common = common.get_dict()
    year = "2023-24"
    row = None
    for rows in career["resultSets"][0]["rowSet"]:
        if rows[1] == year:
            row = rows
    headers = career["resultSets"][0]["headers"]
# print(headers)
# print(headers.index("PLAYER_AGE"))
# print(career)


    common_headers = common["resultSets"][0]["headers"]
    common_row = common["resultSets"][0]["rowSet"][0]


    info ={
 'age': row[headers.index('PLAYER_AGE')],
            'experience': str(int(row[headers.index('SEASON_ID')].split("-")[0]) - int(common_row[common_headers.index("DRAFT_YEAR")]) + 1),
            'g': row[headers.index('GP')],
            'fg_percent': row[headers.index('FG_PCT')],
            'trb': row[headers.index('REB')],  # Per game
            'ast': row[headers.index('AST')],  # Per game
            'pts': row[headers.index('PTS')],   # Per game
            'height': common_row[common_headers.index("HEIGHT")],  # Placeholder, adjust if you have height data
            'weight': common_row[common_headers.index("WEIGHT")],  # Placeholder, adjust if you have weight data
            'position': common_row[common_headers.index("POSITION")]  # Placeholder, adjust if you have position data
        }

    return dict_to_tensor(info)



def all_players():
    return players.get_players()





    




# pandas data frames (optional: pip install pandas)
#career.get_data_frames()[0]


# json
#career.get_json()

# dictionary

#print(career)
#stats = get_player_stats("lebron james", "2023-24")
