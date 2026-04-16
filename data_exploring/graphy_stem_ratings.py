import matplotlib.pyplot as plt
import re

# Read data from the file (Assuming 'ratings.txt')
with open('users_stem_books_with_rating.txt', 'r') as f:
    data = f.read()

# Use regex to extract the integers following "rating: "
ratings = [int(r) for r in re.findall(r'rating: (\d+)', data)]

# Create the histogram
plt.hist(ratings, bins=range(min(ratings), max(ratings) + 2), align='left', rwidth=0.8, color='skyblue', edgecolor='black')
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.title('Distribution of Ratings')
plt.xticks(range(min(ratings), max(ratings) + 1))
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()