import csv
import json


def csv_to_json(csv_file, json_file):
    # Open the CSV file for reading
    with open(csv_file, 'r') as file:
        # Create a CSV reader object
        csv_reader = csv.DictReader(file)

        # Convert CSV data to a list of dictionaries
        data = []
        for row in csv_reader:
            data.append(row)

    # Write the data to a JSON file
    with open(json_file, 'w') as file:
        # Convert data to JSON format
        json_data = json.dumps(data, indent=4)

        # Write JSON data to the file
        file.write(json_data)

    print("Conversion complete.")


# Specify the input CSV file and output JSON file
csv_file = 'People_figures.csv'
json_file = 'People_figures.json'

# Call the function to perform the conversion
csv_to_json(csv_file, json_file)
