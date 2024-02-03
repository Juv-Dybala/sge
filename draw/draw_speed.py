import matplotlib.pyplot as plt
import numpy as np

# Given data
sequence_length = [0.5, 1, 1.5, 3, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55]
transformer_time = [0.4302, 0.6396, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
sge_time = [0.4044, 0.3944, 0.3964, 0.4032, 0.4186, 0.4290, 0.4499, 0.4766, 0.5023, 0.5208, 0.5468, 0.5567, 0.5837, 0.6124, 0.6364]

# Make sure both lists have the same length
min_length = min(len(sequence_length), len(transformer_time), len(sge_time))
sequence_length = sequence_length[:min_length]
transformer_time = transformer_time[:min_length]
sge_time = sge_time[:min_length]

# Plotting the data
plt.plot(sequence_length, transformer_time, marker='o', linestyle='-', label='Transformer')
plt.plot(sequence_length, sge_time, marker='o', linestyle='-', label='SGE')

# Mark OOM with 'x'
for i, val in enumerate(transformer_time):
    if np.isnan(val):
        plt.scatter([sequence_length[i]], [val], marker='x', color='red', s=100)

for i, val in enumerate(sge_time):
    if np.isnan(val):
        plt.scatter([sequence_length[i]], [val], marker='x', color='red', s=100)

# Adding labels and title
plt.xlabel('Sequence Length (x10^4)')
plt.ylabel('Time (s)')
plt.title('Training Time Comparison on Sequence (Long)')

# Adding legend
plt.legend()

# Adding grid
plt.grid(True)

# Display the plot
plt.show()

plt.savefig('./long.png')