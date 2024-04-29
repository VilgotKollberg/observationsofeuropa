from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
import math
import random
from scipy.interpolate import interp1d

###################### STENCIL
def make_stencil(rows, cols, center_y, center_x, radius):
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
    #print(result)
    return matrix

###################### MOVING AVERAGE
def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

###################### FILE READ
def openPSG():
    l = []  # Lambda
    n = []  # Nucleus
    h = []  # H2O

    with open("psg_rad.txt", "r") as file:
        for line in file:
            if not line.startswith("#"):
                parts = line.split()  # Directly assign the split result to a variable
                # Ensure parts has enough elements to avoid IndexError
                if len(parts) > 6:  # Checking there are at least 7 elements (indices 0 through 6)
                    l.append(float(parts[0]))  # Convert the first element to float and append
                    n.append(float(parts[4]))  # Convert the fifth element to float and append
                    h.append(float(parts[6]))  # Convert the seventh element to float and append

    return l, n, h


# Fits files
fits_file = ['NRS1_EXP1_2024.fits', 'NRS2_EXP1_2024.fits', 'NRS1_EXP1_2024.fits', 'NRS2_EXP2_2024.fits']

# Total flux and Iambda
total_flux = []
total_lambda = []
total_Error = []

# Välj fil
for index in range(len(fits_file)):
    print("Observing: " + fits_file[index]) 
    hdul = fits.open(fits_file[index])  # Öppna upp filen vi vill gå igenom
        
    # Extrahera SCI och Error data
    SCI_data = hdul[1] 
    ERR_data = hdul[2]

    hdr = SCI_data.header
    data = SCI_data.data
    error = ERR_data.data
    
    #Skapa stencil 
    # Size Y, X, Cirkel y, x, R
    stencil_moon = make_stencil(49,49,22,24,6)

    # beskär fill och städa bort konstiga värden
    data_from_moon = []
    flux_sum = []
    data_Error_sum = []
    wavelengths = len(data)
    y_depth = len(data[0])
    x_depth = len(data[0][0])
    # För varje våglängd
    pruned=[]
    for w in range(wavelengths):
        # För varje rad (y)
        reflex=[]
        data_Error=[]

        for y in range(y_depth):
            # För varje värde påraden (x)
            for x in range(x_depth):
                # Beskär enligt stencil
                if stencil_moon[y][x]:
                    if data[w][y][x] > 0 and data[w][y][x] < 250000:
                        #print(data[w][y][x])
                        #print(error[w][x][y])
                        #print("Approved!!!")

                        reflex.append(data[w][y][x])
                        data_Error.append(error[w][y][x]**2)
                        #random_number = random.randint(1, 100)
                        #if w > 2300 and w < 2700:
                            #if error[w][y][x] > 5000:
                                #print("5000+", " ", w, " ", y, " ", x, )
                                #print(error[w][y][x])
                            #if random_number == 5:
                                #print(error[w][y][x], " Error w=",w," Error y=",y," Error x=",x,)                        
            # Spara våglängdens beskurna intensiteter 
            
        

        data_from_moon.append(reflex)
                
        #print(reflex)
        if len(reflex) != 0:
            flux_sum.append(sum(reflex)/len(reflex))
        else:
            flux_sum.append(0)
            print("No ")
        if len(data_Error) != 0:
            data_Error_sum.append(math.sqrt(sum(data_Error))/len(data_Error))
        else:
            data_Error_sum.append(0)
            print("Data error does not exist for wawelength")


        
    #print(data_from_moon)
    #print(flux_sum)
    total_flux.append(flux_sum)
    total_lambda.append(np.linspace(hdr['CRVAL3'], (hdr['CRVAL3'] + hdr['CDELT3'] * hdr['NAXIS3']), len(data)))
    total_Error.append(data_Error_sum)
    #print(total_lambda)
    print(data_Error_sum)
    print("Observation of " + fits_file[index] + " is done!")


###################### AVERAGE FLUX
# Gör en average på NRS1 EXP 1 och NRS1 EXP 2 och sätter det som total_flux[1] osv
total_flux[0] = [(f0 + f2) / 2 for f0, f2 in zip(total_flux[0], total_flux[2])]
total_flux[1] = [(f1 + f3) / 2 for f1, f3 in zip(total_flux[1], total_flux[3])]

#total_Error[0] = [math.sqrt((f0**2 + f2**2)) / 2 for f0, f2 in zip(total_flux[0], total_flux[2])]
#total_Error[1] = [math.sqrt((f1**2 + f3**2)) / 2 for f1, f3 in zip(total_flux[1], total_flux[3])]

###################### PSG'S
psg_lambda, psg_Nucleus, psg_H2O = openPSG()


###################### MOVING AVERAGE
window_size = 5 # Example window size, adjust as needed
total_flux_ma = [moving_average(flux, window_size) for flux in total_flux]
psg_Nucleus_ma = moving_average(psg_Nucleus, window_size)
psg_H2O_ma = moving_average(psg_H2O, window_size)


lambda_total_ma = []
for i in range(len(total_lambda[1]) - (window_size-1)):
    centered_index = i + int((window_size - 1) / 2)
    lambda_total_ma.append(total_lambda[1][centered_index])
    

psg_lambda_ma = []
for i in range(len(psg_lambda) - (window_size-1)):
    centered_index = i + int((window_size - 1) / 2)
    psg_lambda_ma.append(psg_lambda[centered_index])
    
print("abow knas")
print(psg_Nucleus)


################### Interpolera för att matcha diskret frekvens på JWSP-data

# Sample data 
x1 = np.array(total_lambda[1])  # x values for the first set
y1 = np.array(total_flux[1])  # y values for the first set

x2 = np.array(psg_lambda)  # x values for the second set
y2 = np.array(psg_Nucleus)  # y values for the second set
y2_2 = np.array(psg_H2O)





print("Längder, lambda, Nucleus, H2O")
print(len(psg_lambda))
print(len(psg_Nucleus))
print(len(psg_H2O))
print(len(psg_lambda_ma))
print(len(psg_Nucleus_ma))
print(len(psg_H2O_ma))


# Interpolate the second set onto the x values of the first set
interp_func = interp1d(x2, y2, kind='linear', fill_value='extrapolate')
interp_func_H2O = interp1d(x2, y2_2, kind='linear', fill_value='extrapolate')
y2_interp_PSG_Nucleus = interp_func(x1)
y2_2_interp_PSG_H2O = interp_func_H2O(x1)

# Add the interpolated values to the first set
#y_combined = y1 + y2_interp

# Print the combined y values
#print("Combined y values:", y_combined)
######################



# ta bort solen
flux_comp = []
flux_psg = []

flux_wow = []
psg_H2O_ma_ma_big = []
flux_H2O = []


#for i in range(len(total_flux[1]) - (window_size-1)):
for i in range(len(total_flux[1])):
    flux_comp.append(total_flux[1][i]) #/(y2_interp_PSG_Nucleus[i]*(*10**6)))
    
    ####### olika skalning på nucleus
    flux_psg.append(y2_interp_PSG_Nucleus[i]*(3.886*10**6))
    flux_wow.append(total_flux[1][i]-y2_interp_PSG_Nucleus[i]*(3.886*10**6))
    flux_H2O.append(y2_2_interp_PSG_H2O[i]*(3.886*10**6))
    ###
    #flux_psg.append(y2_interp_PSG_Nucleus[i]*(3.886*0.922*10**6))
    #flux_wow.append(total_flux[1][i]-y2_interp_PSG_Nucleus[i]*(3.886*0.922*10**6))
    #flux_H2O.append(y2_2_interp_PSG_H2O[i]*(3.886*0.922*10**6))
    
    #psg_H2O_ma_ma_big.append(psg_H2O_ma[i+2]/6*10**5)

#print(total_flux[1][i]-total_flux_ma[1][i])




window_normalizer = 5

flux_wow_ma = moving_average(flux_wow, window_size)
flux_wow_ma_ma = moving_average(flux_wow_ma, window_normalizer)

lambda_wow_ma = []
final = []

for i in range(len(flux_wow_ma)):#-(window_normalizer-1)):
    final.append(flux_wow[i+2]-flux_wow_ma[i]) #/(y2_interp_PSG_Nucleus[i]*(*10**6)))
    lambda_wow_ma.append(total_lambda[1][i+2])

final_ma = moving_average(final, window_size)
lambda_final_ma = []
for i in range(len(final_ma)):
    lambda_final_ma.append(lambda_wow_ma[i+2])

     
    
    
print("sizes")
print(len(psg_H2O_ma_ma_big))
print(len(psg_H2O_ma))
print(len(lambda_total_ma))
print(len(psg_lambda_ma))
print("sizes interpol")
print(len(y2_interp_PSG_Nucleus))
print(len(y2_2_interp_PSG_H2O))    
print(len(total_flux[1]))

#fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 10))  # 2 rows, 1 column

plt.plot(total_lambda[1], y2_2_interp_PSG_H2O, label='PSG H20')
plt.legend() 
plt.xlabel('Wavelength (µm)')
plt.ylabel('Intensity (Jy)') 
plt.title('PSG H2O')
plt.show()

# Show the figure with subplots
plt.show()

plt.plot(lambda_final_ma, final_ma, label='JWST Data')
plt.legend() 
plt.xlabel('Wavelength (µm)')
plt.ylabel('Intensity (Jy/sr)') 
plt.title('JWST Data After Proccesing')
plt.show()

plt.plot(psg_lambda, psg_Nucleus, label='Nucleus')
plt.legend() 
plt.xlabel('Wavelength (µm)')
plt.ylabel('Intensity (Jy)') 
plt.title('Nucleus')
plt.show()

plt.plot(lambda_final_ma, final_ma, label='JWST Data')
plt.plot(total_lambda[1], flux_H2O, label='H2O PSG')
#plt.plot(psg_lambda_ma, psg_H2O_ma, label='H2O PSG')
plt.legend() 
plt.xlabel('Wavelength (µm)')
plt.ylabel('Intensity (µJy/sr)') 
plt.title('Comparison, smooth JWST data vs H2O PSG')
plt.show()

plt.plot(lambda_wow_ma, final, label = 'JWST Data before moving average')
plt.plot(total_lambda[1], flux_H2O, label='H2O PSG')
plt.legend() 
plt.xlabel('Wavelength (µm)')
plt.ylabel('Intensity (µJy/sr)')  
plt.title('Comparison, JWST data vs H2O PSG')
plt.show()

plt.plot(total_lambda[1], flux_wow, label = 'JWST Data')
plt.plot(lambda_wow_ma, flux_wow_ma, label = 'JWST Data with moving average')
plt.legend() 
plt.xlabel('Wavelength (µm)')
plt.ylabel('Intensity (µJy/sr)')  
plt.title('Plot to illustrate centering around 0')
plt.show()


print(len(lambda_total_ma))
print(len(flux_comp))
print("längd på interpol OSG Nucleus")
print(y2_interp_PSG_Nucleus)
###################### PLOT



print(lambda_total_ma)


plt.plot(total_lambda[1], flux_comp, label = 'JWST Data')
plt.plot(total_lambda[1], flux_psg, label = 'Nucles PSG')
plt.legend() 
plt.xlabel('Wavelength (µm)')
plt.ylabel('Intensity (µJy/sr)')  
plt.title('Plot to illustrate removal of Nucleus')
plt.show()
#plt.plot(lambda_total_ma, y2_interp_PSG_Nucleus)
#plt.show()

# Since we've applied a moving average, the length of each total_flux_ma list
# is reduced compared to the original total_flux lists. We must adjust total_lambda
# accordingly to match the lengths for plotting.
total_lambda_ma = [lambdas[window_size-1:] for lambdas in total_lambda]

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(total_lambda_ma[1], total_flux_ma[1], label='NRS2 MA')
ax.plot(total_lambda_ma[0], total_flux_ma[0], label='NRS1 MA')
ax.set_title("Wavelength spectrum with Moving Average")
ax.set_xlabel('Wavelength [µm]')
ax.set_ylabel('Flux [Jy/sr]')
ax.legend()

# Show the plot
plt.show()

plt.plot(total_lambda[1], total_flux[1], label = 'NRS2')
plt.plot(total_lambda[0], total_flux[0], label = 'NRS1')
plt.legend() 
plt.xlabel('Wavelength (µm)')
plt.ylabel('Intensity (Jy/sr)')  
plt.title('Wavelength spectrum')
plt.show()

plt.plot(total_lambda[1], total_Error[1], label = 'NRS2 Total Error')
plt.plot(lambda_final_ma, final_ma, label='JWST Data')
plt.plot(total_lambda[1], flux_H2O, label='H2O PSG')
plt.legend() 
plt.xlabel('Wavelength (µm)')
plt.ylabel('Intensity (µJy/sr)')  
plt.title('Total Error')
plt.show()

print("TOTAL ERROR")
print(total_Error[0][1000])