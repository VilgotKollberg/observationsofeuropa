# Ett program so låter användaren skapa en matris som går att använda som en stensil för att plocka ut spesifika värden ur en annan matris
# Det går att rita en stencil som en cirkel med utifrån en medelpunkt och radie
import numpy as np
import math

class ZeroMatrix:
    def __init__(self, rows, cols, item=0):
        self.rows = rows
        self.cols = cols
        self.item = item
        self.matrix = self.create_zero_matrix()

    def create_zero_matrix(self):
        matrix = []
        for _ in range(self.rows):
            row = [self.item] * self.cols
            matrix.append(row)
        return matrix

    def __str__(self):
        result = "    " + " ".join(f"{i:2}" for i in range(self.cols)) + "\n"
        result += "  " + "=" * (self.cols * 3 + 2) + "\n"  # Horizontal line
        for i, row in enumerate(self.matrix):
            result += f"{i:2}||"
            result += " ".join(f"{val:2}" for val in row) + "\n"
        return result
    
    def fill_circle(self, center_x, center_y, radius, item=1):
        for i in range(self.rows):
            for j in range(self.cols):
                if math.sqrt((i - center_y) ** 2 + (j - center_x) ** 2) <= radius:
                    self.matrix[i][j] = item

    def extract_values(self, random_matrix):
            if not isinstance(random_matrix, list) or not all(isinstance(row, list) for row in random_matrix):
                raise TypeError("The input matrix must be a list of lists.")

            if len(random_matrix) != self.rows or any(len(row) != self.cols for row in random_matrix):
                raise ValueError("The size of the input matrix does not match the size of the stencil matrix.")

            values = []
            for i in range(self.rows):
                for j in range(self.cols):
                    if self.matrix[i][j] != self.item:
                        values.append(random_matrix[i][j])
            return values

# Example usage:
M = 10  # Number of rows
N = 10  # Number of columns
zero_matrix = ZeroMatrix(M, N, 0)
zero_matrix.fill_circle(5, 5, 3.5, 1)  # Fill a circle with center at (5, 5) and radius 3 with ones
print(zero_matrix)
random_matrix = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                 [11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                 [21, 22, 23, 24, 25, 26, 27, 28, 29, 30],
                 [31, 32, 33, 34, 35, 36, 37, 38, 39, 40],
                 [41, 42, 43, 44, 45, 46, 47, 48, 49, 50],
                 [51, 52, 53, 54, 55, 56, 57, 58, 59, 60],
                 [61, 62, 63, 64, 65, 66, 67, 68, 69, 70],
                 [71, 72, 73, 74, 75, 76, 77, 78, 79, 80],
                 [81, 82, 83, 84, 85, 86, 87, 88, 89, 90],
                 [91, 92, 93, 94, 95, 96, 97, 98, 99, 100]]

try:
    extracted_values = zero_matrix.extract_values(random_matrix)
    print("Extracted Values:", extracted_values)
except TypeError as e:
    print("Error:", e)
except ValueError as e:
    print("Error:", e)

                    
# Example usage:
M = 10  # Number of rows
N = 10  # Number of columns
zero_matrix = ZeroMatrix(M, N, ".")
new_var = zero_matrix
print(new_var)
zero_matrix.fill_circle(5, 5, 3, "#")  # Fill a circle with center at (5, 5) and radius 3
print(zero_matrix)

# Example usage:
M = 49  # Number of rows
N = 49  # Number of columns
zero_matrix = ZeroMatrix(M, N, ".")
print(zero_matrix)
#zero_matrix.fill_circle(23.53, 25.25, 4.8754, "#")  # Fill a circle with center at (5, 5) and radius 3
zero_matrix.fill_circle(23.91, 24.47, 2.91, "#")  # Fill a circle with center at (5, 5) and radius 3
print("With X=23.91, Y=24.47, R=2.91")
print(zero_matrix)


M = int(input("antal rader\n"))
N = int(input("antal kolumner\n"))
fill_item = input("Item\n")
zero_matrix = ZeroMatrix(M, N, fill_item)
print(zero_matrix)
x = float(input("x cord\n"))
y = float(input("y cord\n"))
r = float(input("Radie\n"))
circe_litem = input("Item\n")
zero_matrix.fill_circle(x,y,r,circe_litem)
print(zero_matrix)




#########################
# wavelength prosesing kod
test_Fil = [
    [[11, 12, 13], [14, 15, 16], [17, 18, 19]],
    [[21, 22, 23], [24, 25, 26], [27, 28, 29]],
    [[31, 32, 33], [34, 35, 36], [37, 38, 39]]
]

stencil_1 = [
    [0, 1, 0],
    [1, 1, 1],
    [0, 1, 0]
]  # +

stencil_2 = [
    [0, 1, 0],
    [1, 0, 1],
    [0, 1, 0]
]  # O

stencil_3 = [
    [0, 0, 0],
    [0, 1, 0],
    [0, 0, 0]
]  # .

stenciler = []
stenciler.append(stencil_1)
stenciler.append(stencil_2)
stenciler.append(stencil_3)

# In fil, 3d matris[W1[ Y1[X1, X2, X3] 
#                       Y2[X1, X2, X3] 
#                       Y3[X1, X2, X3]] W2[[---][---][---]] W3[[---][---][---]]]
# In stencil, en stencil med samma x,y som 3d matrisen: [ Y1[X1, X2, X3] 
#                                                         Y2[X1, X2, X3] 
#                                                         Y3[X1, X2, X3]]
# Return pruned, en lista med de sparade intenciteterna från varje våglängd [[alla värden för W1][alla värden för W2][alla värden för W3]...]
def prune(fil, stencil):

    wavelengths = len(fil)
    y_depth = len(fil[0])
    x_depth = len(fil[0][0])
    # För varje våglängd
    pruned=[]
    for w in range(wavelengths):
        # För varje rad (y)
        reflex=[]
        for y in range(y_depth):
            # För varje värde påraden (x)
            for x in range(x_depth):
                # Beskär enligt stencil
                if stencil[y][x]:
                    reflex.append(fil[w][y][x])
        # Spara våglängdens beskurna intenciteter 
        pruned.append(reflex)
    return pruned

def handle_file_and_stencil(fil,stencil):
    proceced=[]
    for i in range(len(stenciler)):
        proceced.append(prune(fil,stencil[i]))
    return proceced

def printMatrix(matrix):
    for i in range(len(matrix)):
        print(matrix[i])
    return

beskurna = handle_file_and_stencil(test_Fil, stenciler)
print("\nTestfil:")
for i in range(len(test_Fil)):
    print("\n", "wavelength ", i+1)
    printMatrix(test_Fil[i])
print("\nstenciler:")
for i in range(len(stenciler)):
    print("\n", "stencil ", i+1)
    printMatrix(stenciler[i])
print("\nbeskurna:")
for i in range(len(beskurna)):
    print("\nFrån stencil ", i+1)
    print(beskurna[i])

    #Skapa stencil
rows = 49
cols = 49
center_y = 22
center_x = 24
radius = 6
matrix = []
for _ in range(rows):
    row = [0] * cols
    matrix.append(row)
for i in range(rows):
    for j in range(cols):
        if math.sqrt((i - center_y) ** 2 + (j - center_x) ** 2) <= radius:
            matrix[i][j] = 1
result = "    " + " ".join(f"{i:2}" for i in range(cols)) + "\n"
result += "  " + "=" * (cols * 3 + 2) + "\n"  # Horizontal line
for i, row in enumerate(matrix):
    result += f"{i:2}||"
    result += " ".join(f"{val:2}" for val in row) + "\n"
print(result)