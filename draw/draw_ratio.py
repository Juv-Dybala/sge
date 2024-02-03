import matplotlib.pyplot as plt
import numpy as np

# Given data
sequence_length = [3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000, 13000]
transformer_time = [0.4223, 0.4190, 0.4302, 0.4353, 0.4718, 0.4964, 0.5893, 0.6396, 0.6553, 0.6748, 0.7308]
sge_time = [0.3863, 0.3952, 0.4044, 0.3780, 0.4006, 0.3728, 0.3942, 0.3944, 0.3946, 0.3871, 0.3924]
acceleration_ratio = [1.0932, 1.0622, 1.0638, 1.1516, 1.1777, 1.3125, 1.4949, 1.6217, 1.6607, 1.7432, 1.8624]

# Plotting the data
fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('Sequence Length')
ax1.set_ylabel('Time (s)', color=color)
ax1.plot(sequence_length, transformer_time, marker='o', linestyle='-', color=color, label='Transformer')
ax1.plot(sequence_length, sge_time, marker='o', linestyle='-', color='tab:blue', label='SGE')
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()
color = 'tab:green'
ax2.set_ylabel('Acceleration Ratio', color=color)
ax2.plot(sequence_length, acceleration_ratio, marker='o', linestyle='-', color=color, label='Acceleration Ratio')
ax2.tick_params(axis='y', labelcolor=color)

# Adding labels and title
plt.title('Transformer and SGE Performance')

# Display the legend
fig.tight_layout()
plt.legend(loc='upper left')

# Display the plot
plt.show()
plt.savefig('./ratio.png')
