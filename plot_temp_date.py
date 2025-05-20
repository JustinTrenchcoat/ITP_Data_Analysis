from itp.itp_query import ItpQuery
import matplotlib.pyplot as plt
from itp.profile import Profile



global dates
dates = []

global temp_values
temp_values = []
 
global miss_data
miss_data=[]

# Sample data
path = r'D:\EOAS\ITP_package_try\itp_data\itp_final_2025_05_07.db'
for i in range(1,132):
    print("Processing ITP#", i)
    query = ItpQuery(path, system=[i])
    query.set_max_results(10000)
    results = query.fetch()
    
    longitude = [p.longitude for p in results]
    
    latitude = [p.latitude for p in results]
    
    # print out size of the ITP system set
    # print(len(results))
    if (len(results)>0):
        for p in results:
            dates.append(Profile.python_datetime(p))

        for p in results:
            avg_temp = sum(Profile.conservative_temperature(p))/len(Profile.conservative_temperature(p))
            temp_values.append(avg_temp)
    if(len(results)==0):
        print("ITP#",i," has no data input")
        miss_data.append(i)
    print("Processing Complete, Proceed to next...")

# Plotting
plt.scatter(dates, temp_values)
plt.xlabel("Date")
plt.ylabel("Temperature")
plt.title("Time vs. Avg temperature for 2025_05_07.db file")
plt.grid(True)

# Optional: Rotate date labels for clarity
plt.xticks(rotation=45)

plt.savefig("plots/time_vs_temp_05_07.png")
plt.show()
print("There are total", len(miss_data)," systems without data.")
print("Here is the list for ITP# without data input:")
for i in range(len(miss_data)):
    print(miss_data[i])