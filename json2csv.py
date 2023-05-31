import json
import csv
import ijson
import pandas as pd

# f = open('WikidataNE_20161205_NECKAR_1_0.json')
# objects = ijson.items(f, 'id.item', multiple_values=True)
# cities = [o for o in objects][:10]
# for city in cities:
#     print(city)
reader = pd.read_json('WikidataNE_20161205_NECKAR_1_0.json', lines=True, chunksize=1000)

df = pd.DataFrame()
count = 0
for chunk in reader:
    df = df.append(pd.DataFrame(chunk[['norm_name', 'id', 'neClass']]))
    count += 1000
    print(count)

df.to_csv('ne.csv')

# data_file = open('data_file.csv', 'w')
# csv_writer = csv.writer(data_file)
# count = 0
# for obj in df:
#     csv_writer.writerow()