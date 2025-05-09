import csv
import numpy as np

# Define a function to convert action to words
def action_to_words(action):

    def _separate_array_and_integer(input_array):
        # Find the non-zero element and its index
        non_zero_index = np.nonzero(input_array)[0]
        
        if len(non_zero_index) == 0:
            raise ValueError("Input array must contain at least one non-zero element.")
        
        # Get the first non-zero element
        non_zero_value = input_array[non_zero_index[0]]
        
        # Create the new array with the same direction but normalized
        new_array = input_array / np.abs(non_zero_value)
        
        # The integer is the absolute value of the non-zero element
        integer_value = non_zero_value
        
        return new_array, integer_value
    
    direction_map = {
        (1, 0): "down",
        (-1, 0): "up",
        (0, 1): "right",
        (0, -1): "left"
    }
    
    # Convert the action string to a list of integers
    action_list = eval(action)  # Use eval carefully; ensure input is trusted
    action_array = np.array(action_list)
    direction, step = _separate_array_and_integer(action_array)

    # Get the direction word
    direction_word = direction_map.get(tuple(direction.tolist()), "unknown")
    
    # Return the formatted string
    return f"{direction_word} {abs(action_list[0]) if action_list[0] != 0 else abs(action_list[1])}"

# Read the CSV file and convert actions
with open('tests/test1/solution_manual.csv', mode='r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        src = row['src']
        action = row['action']
        words = action_to_words(action)
        print(f"{src} -> {words}")
