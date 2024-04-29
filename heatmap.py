from astropy.io import fits
import numpy as np
from Crop import croppen
import matplotlib.pyplot as plt

fits_file = ['NRS1_EXP1.fits', 'NRS2_EXP1.fits', 'NRS1_EXP2.fits', 'NRS2_EXP2.fits']


hdul = fits.open('NRS2_EXP1_2024.fits')  # Open up the fits file for the selected observation
data_All = hdul[1]  # Extracts scientific data


data_spec = data_All.data[2500]
data_spec[np.isnan(data_spec)] = 0

# Normalize data to range between 0 and 1
min_val = min(min(row) for row in data_spec)
max_val = max(max(row) for row in data_spec)
normalized_data = [[(val - min_val) / (max_val - min_val) for val in row] for row in data_spec]

# Define characters to represent different density levels
#characters = ['.', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
characters = ['.', ',', '-', '~', '+', '*', 'x', 'X', '#', '&', '%', '@']
#characters = ['|', '_', '~', '+', '*', 'x', 'X', '#', '&', '%', '@']
# Map normalized values to characters
heatmap = [[' ' for _ in range(len(normalized_data[0]))] for _ in range(len(normalized_data))]
print("längd")
print(len(normalized_data[0]))
print(len(normalized_data))
for i in range(len(normalized_data)):
    for j in range(len(normalized_data[i])):
        print("Ny pixel")
        print(i)
        print(j)
        index = int(normalized_data[i][j] * (len(characters) - 1))
        heatmap[i][j] = characters[index]


def print_heatmap(heatmap):
    rows = len(heatmap)
    cols = len(heatmap[0])
    
    result = "    " + " ".join(f"{i:2}" for i in range(cols)) + "\n"
    result += "  " + "=" * (cols * 3 + 2) + "\n"  # Horizontal line
    for i, rows in enumerate(heatmap):
        result += f"{i:2}||"
        result += " ".join(f"{val:2}" for val in rows) + "\n"
    return result


# Print the ASCII heatmap
space_image = print_heatmap(heatmap)
print(space_image)

def find_planet_center(normalized_data):
    total_weight = 0
    x_weighted_sum = 0
    y_weighted_sum = 0
    
    rows = len(normalized_data)
    cols = len(normalized_data[0])
    
    for i in range(rows):
        for j in range(cols):
            # Use the normalized value as the weight
            weight = normalized_data[i][j]
            total_weight += weight
            x_weighted_sum += j * weight
            y_weighted_sum += i * weight
            
    if total_weight == 0:
        return None  # Prevent division by zero
    
    center_x = x_weighted_sum / total_weight
    center_y = y_weighted_sum / total_weight
    
    return (center_x, center_y)
def find_planet_radius(normalized_data, center_x, center_y):
    edge_values = []  # List to hold distances of edge points from center
    
    rows, cols = normalized_data.shape
    threshold = 0.5  # Assuming normalized_data, adjust threshold as needed

    # Iterate over the array to find edge points
    for i in range(rows):
        for j in range(cols):
            # Consider a point as part of the edge if its value crosses the threshold
            # This threshold should be adjusted based on how you define the 'edge' in your normalized data
            if normalized_data[i, j] > threshold:
                distance = np.sqrt((center_x - j) ** 2 + (center_y - i) ** 2)
                edge_values.append(distance)

    # Calculate the average distance to estimate the radius
    if edge_values:
        radius = np.mean(edge_values)
        return radius
    else:
        return None

# Assuming you've calculated center_x and center_y previously
center_x, center_y = find_planet_center(normalized_data)

# Convert your list of lists normalized_data to a NumPy array if it's not already
normalized_data_np = np.array(normalized_data)

# Calculate the radius
planet_radius = find_planet_radius(normalized_data_np, center_x, center_y)

if planet_radius:
    print(f"Estimated radius of the planet: {planet_radius:.2f}")
else:
    print("Unable to estimate the planet's radius.")

# Calculate the center of the planet
planet_center = find_planet_center(normalized_data)

if planet_center:
    print(f"Center of the planet is approximately at coordinates: x={planet_center[0]:.2f}, y={planet_center[1]:.2f}")
else:
    print("Unable to determine the center of the planet.")

print("längd")
print(len(normalized_data[0]))
print(len(normalized_data))