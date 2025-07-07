import numpy as np
from sklearn.decomposition import NMF
import pandas as pd

# داده‌های اولیه
ratings = np.array([
    [5, 3, 0, 1, 0],
    [4, 0, 0, 1, 2],
    [1, 1, 0, 5, 0],
    [0, 0, 5, 4, 0],
])

# ساخت مدل NMF با 2 تا latent feature
model = NMF(n_components=2, init='random', random_state=42)
user_features = model.fit_transform(ratings)
item_features = model.components_

# بازسازی ماتریس امتیازها
reconstructed_ratings = np.dot(user_features, item_features)

# نمایش
print(pd.DataFrame(np.round(reconstructed_ratings, 2), 
                   columns=["M1", "M2", "M3", "M4", "M5"],
                   index=["U1", "U2", "U3", "U4"]))

new_user = np.array([[5, 3, 0, 0, 0]])  # فقط امتیاز به M1 و M2

new_user_features = model.transform(new_user)
new_user_ratings = np.dot(new_user_features, item_features)

print("Predicted ratings for new user:", np.round(new_user_ratings, 2))


import matplotlib.pyplot as plt
import seaborn as sns

# اصل
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
sns.heatmap(ratings, annot=True, cmap="Blues", cbar=False)
plt.title("Original Ratings")

# بازسازی‌شده
plt.subplot(1,2,2)
sns.heatmap(reconstructed_ratings, annot=True, cmap="Greens", cbar=False)
plt.title("Reconstructed Ratings")
plt.show()



from sklearn.metrics import mean_squared_error

# فقط مقادیر غیر صفر را برای ارزیابی در نظر بگیر
mask = ratings > 0

mse = mean_squared_error(ratings[mask], reconstructed_ratings[mask])
print(f"Mean Squared Error (on known ratings): {mse:.4f}")
