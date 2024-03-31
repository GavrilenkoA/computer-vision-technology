import pickle


with open('tracks/track_9', 'rb') as f:
    some_dict = pickle.load(f)
    amnt = some_dict['country_balls_amount']
    track_data_test = some_dict['track_data']

print(f'country_balls_amount = {amnt}')
print(f'track_data = {track_data_test}')
