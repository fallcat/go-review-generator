import pickle

with open('data_splits/train.pkl', 'rb') as input_file:
    data = pickle.load(input_file)
    new_data = {}
    new_data['boards'] = data['boards']
    new_data['steps'] = data['steps']
    with open('data_splits_final/train.pkl', 'wb') as output_file:
        pickle.dump(new_data, output_file)

with open('data_splits/val.pkl', 'rb') as input_file:
    data = pickle.load(input_file)
    new_data = {}
    new_data['boards'] = data['boards']
    new_data['steps'] = data['steps']
    with open('data_splits_final/val.pkl', 'wb') as output_file:
        pickle.dump(new_data, output_file)

with open('data_splits/test.pkl', 'rb') as input_file:
    data = pickle.load(input_file)
    new_data = {}
    new_data['boards'] = data['boards']
    new_data['steps'] = data['steps']
    with open('data_splits_final/test.pkl', 'wb') as output_file:
        pickle.dump(new_data, output_file)
