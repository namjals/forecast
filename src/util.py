import csv


def read_data(file):
    data = {}
    with open(file) as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key, val in row.items():
                if key in data:
                    data[key].append(val)
                else:
                    data[key] = [val]
    return data