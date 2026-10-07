import pandas as pd
from sklearn.tree import DecisionTreeRegressor

melbourne_file_path = "dataset/melb_data.csv"
melbourne_data = pd.read_csv(melbourne_file_path)


def main():
    melbourne_file_path = "dataset/melb_data.csv"
    melbourne_data = pd.read_csv(melbourne_file_path)
    describe = melbourne_data.describe()
    melbourne_data = melbourne_data.dropna(axis=0)
    y = melbourne_data.Price

    melbourne_features = ["Rooms", "Bathroom", "Landsize"]
    x = melbourne_data[melbourne_features]
    x.describe()

    melbourne_model = DecisionTreeRegressor(random_state=1)

    # Fit model
    melbourne_model.fit(x, y)
    output = melbourne_model.predict(x.head())
    print(output)


if __name__ == "__main__":
    main()