# Utility for finding the mean and variance reported in paper

import numpy as np

def calculate_mean_std(file_path):
    try:
        # Read data from file
        with open(file_path, 'r') as file:
            data = file.readlines()

        # Convert data to a list of numbers
        numbers = [float(line.strip()) for line in data]

        # Calculate mean and standard deviation
        mean = np.mean(numbers)
        std_dev = np.std(numbers)

        return mean, 2 * std_dev/np.sqrt(len(numbers))
    
    except FileNotFoundError:
        return "File not found. Please check the file path."
    except ValueError:
        return "File contains non-numeric data."
    except Exception as e:
        return f"An error occurred: {e}"

# Example usage
file_path = "examples/comparison_methods/gp_comparison/msep/regression_spectroscopy.txt"
mean, std_dev = calculate_mean_std(file_path)

print(mean, std_dev)
