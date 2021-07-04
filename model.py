import dataset

netflix = dataset.get_data('NFLX', '2021-01-01', '2021-04-01')
dataset.change_index(netflix)
netflix_train, netflix_test = dataset.create_test_train_dataset(netflix)
x_train, y_train = dataset.create_xy_dataset(netflix_train, 1)
x_test, y_test = dataset.create_xy_dataset(netflix_test, 1)

