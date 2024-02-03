import matplotlib.pyplot as plt
import numpy as np

# Given data
sequence_length = [3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000, 13000]
transformer_time = [0.4223, 0.4190, 0.4302, 0.4353, 0.4718, 0.4964, 0.5893, 0.6396, 0.6553, 0.6748, 0.7308]
sge_time = [0.3863, 0.3952, 0.4044, 0.3780, 0.4006, 0.3728, 0.3942, 0.3944, 0.3946, 0.3871, 0.3924]

# Plotting the data
plt.plot(sequence_length, transformer_time, marker='o', linestyle='-', label='Transformer')
plt.plot(sequence_length, sge_time, marker='o', linestyle='-', label='SGE')

# Mark OOM with 'X' and connect with dashed line
for i, val in enumerate(transformer_time):
    if val is None:
        plt.scatter([sequence_length[i]], [0.75], marker='X', color='red', s=100) 

# Adding labels and title
plt.xlabel('Sequence Length')
plt.ylabel('Time (s)')
plt.title('Training Time Comparison on Sequence (Short)')

# Adding legend
plt.legend()

# Adding grid
plt.grid(True)
plt.yticks([0.35+x*0.05 for x in range(8) ])
# Display the plot
plt.show()


plt.savefig('./short.png')
