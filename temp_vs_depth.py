from itp.itp_query import ItpQuery
import matplotlib.pyplot as plt
from itp.profile import Profile

# Sample data
path = r'D:\EOAS\ITP_package_try\itp_data\itp_final_2025_05_07.db'
query = ItpQuery(path, system=[1])
results = query.fetch()

longitude = [p.longitude for p in results]
    
latitude = [p.latitude for p in results]
    
sample = results[0]

depth = Profile.depth(sample)
temp = Profile.conservative_temperature(sample)
print(sample.date_time)

# for i in range(1,132):
#     print("Processing ITP#", i)
#     query = ItpQuery(path, system=[i])
#     query.set_max_results(10000)
#     results = query.fetch()
    
#     longitude = [p.longitude for p in results]
    
#     latitude = [p.latitude for p in results]
    
#     # print out size of the ITP system set
#     # print(len(results))
#     if (len(results)>0):
#         for p in results:
#             dates.append(Profile.python_datetime(p))

#         for p in results:
#             avg_temp = sum(Profile.conservative_temperature(p))/len(Profile.conservative_temperature(p))
#             temp_values.append(avg_temp)

#     print("Processing Complete, Proceed to next...")

# Plotting
plt.plot(temp, depth, marker='o',linestyle='dashed',
     linewidth=2, markersize=12)
plt.xlabel("Temperature")
plt.ylabel("Depth")
plt.title("Temperature vs. Depth for 2025_05_07.db file system 001")
plt.grid(True)
plt.gca().invert_yaxis()

# Optional: Rotate date labels for clarity
plt.xticks(rotation=45)
plt.show()
