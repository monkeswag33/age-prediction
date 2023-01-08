import pandas as pd, os

base_dir = "UTKFace/"
filenames, genders, ages, races = [], [], [], []
for file in os.listdir(base_dir):
    temp = file.split("_")
    if len(temp) < 4:
        continue
    filenames.append(file)
    genders.append(temp[0])
    ages.append(temp[1])
    races.append(temp[2])

df = pd.DataFrame(list(zip(filenames, ages)), columns=["filename", "age"])
df.to_csv("data.csv", index=False, header=True)
