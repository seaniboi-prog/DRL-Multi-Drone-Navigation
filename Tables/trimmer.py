import os
import re

def round_floats_in_file(input_file_path, output_file_path, decimal_places):
    # Read the content of the input file
    with open(input_file_path, 'r') as file:
        content = file.read()

    # Regular expression pattern to find all floats in the file
    float_pattern = r"\b\d+\.\d+\b"

    # Function to round floats to the specified decimal places
    def round_match(match):
        float_value = float(match.group())
        return f"{float_value:.{decimal_places}f}"

    # Replace all floats in the content with the rounded values
    new_content = re.sub(float_pattern, round_match, content)

    # Write the modified content to the output file
    with open(output_file_path, 'w') as file:
        file.write(new_content)

def process_directory(input_directory, output_directory, decimal_places):
    # Ensure the output directory exists
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Iterate over all files in the input directory
    for filename in os.listdir(input_directory):
        input_file_path = os.path.join(input_directory, filename)
        
        # Check if it's a file (not a directory or other types)
        if os.path.isfile(input_file_path):
            output_file_path = os.path.join(output_directory, filename)
            try:
                round_floats_in_file(input_file_path, output_file_path, decimal_places)
                print(f"Processed file: {filename} -> Saved to {output_file_path}")
            except Exception as e:
                print(f"Error processing file {filename}: {e}")

if __name__ == "__main__":
    # Ask the user for the number of decimal places
    n = int(input("Enter the number of decimal places to round floats to: "))
    
    # Define the input and output directory paths
    input_directory = 'raw_data'
    output_directory = 'trimmed_data'
    
    # Process all files in the input directory and save them to the output directory
    process_directory(input_directory, output_directory, n)
