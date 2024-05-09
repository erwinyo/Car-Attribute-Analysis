if __name__ == '__main__':
    data_str = "Color: (golden, prob: 0.8583461046218872), Type: (sedan)"

    # Split the string by commas
    data_list = data_str.split(",")

    # Create an empty dictionary
    data_dict = {}

    for item in data_list:
        # Remove extra spaces and split by colon
        key, value = item.strip().split(":")

        # Split the value (handle case with or without probability)
        split_value = value.strip()[1:-1].split(",")

        if len(split_value) == 2:  # Check if there are two values
            value_key, value_prob = split_value
            data_dict[key.strip()] = {"value": value_key.strip(), "probability": float(value_prob.strip())}
        else:  # Case with only value
            data_dict[key.strip()] = {"value": split_value[0].strip(),
                                      "probability": None}  # Set probability to None

    print(data_dict)