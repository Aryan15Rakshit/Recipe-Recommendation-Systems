import pandas as pd

data = pd.read_csv("recipe_final.csv")

data['image_url'] = "https://via.placeholder.com/300"

data.to_csv("recipe_fixed.csv", index=False)

print("Done ✅")