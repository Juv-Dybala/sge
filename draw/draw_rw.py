import matplotlib.pyplot as plt

# Given data
sequence_length = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
random_walk_time = [0.1865, 0.3647, 0.5457, 0.7233, 0.9280, 1.1055, 1.2484, 1.5232, 1.6533, 1.8521]

# Plotting the data
plt.plot(sequence_length, random_walk_time, marker='o', linestyle='-')

# Adding labels and title
plt.xlabel('Sequence Length (x10^4)')
plt.ylabel('Random Walk Time (s)')
plt.title('Random Walk Time on Sequence')
plt.grid(True,linewidth=0.75)
plt.xticks([0] + sequence_length)
plt.yticks([0,0.25,0.5,0.75,1,1.25,1.5,1.75,2])
# Display the plot
plt.show()
plt.savefig('./random_walk.png')