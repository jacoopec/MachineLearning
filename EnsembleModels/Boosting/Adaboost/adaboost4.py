import pandas as pd
import matplotlib.pyplot as plt



data = {
    "Weather":     ["Sunny", "Sunny",  "Sunny", "Windy", "Rainy",  "Rainy",  "Rainy", "Windy",  "Sunny",   "Windy",    "Sunny"],
    "Temperature": ["Cold",   "Normal", "Hot",   "Hot",   "Cold",   "Hot",   "Normal", "Cold",    "Cold",   "Normal",   "Hot"],
    "Play?":         ["No",   "Yes",    "Yes",    "Yes",   "No",    "No",     "No",     "No",     "No",      "No",      "Yes"],
}

# "NumberPlayers": [3, 2, 7, 4, 3, 2, 6, 9, 4],

df = pd.DataFrame(data)

def is_sunny():
    print(df["Weather"] == "Sunny")
    return

df["sample_weight"] = 1/9

df["Cat1_code"] = df["Weather"].astype("category").cat.codes
df["Cat2_code"] = df["Temperature"].astype("category").cat.codes


color_map = {"Yes": "green", "No": "red"}
colors = df["Play?"].map(color_map)

plt.scatter(df["Cat1_code"], df["Cat2_code"], c=colors)
plt.xticks(
    df["Cat1_code"].unique(),
    df["Weather"].astype("category").cat.categories
)
plt.yticks(
    df["Cat2_code"].unique(),
    df["Temperature"].astype("category").cat.categories
)

plt.xlabel("Weather")
plt.ylabel("Temperature")
plt.show()
is_sunny()


# new_sample_weight = sample_weight * e**(amount_of_say)