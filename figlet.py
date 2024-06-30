import pyfiglet
import argparse

# Parse Arguments
parser = argparse.ArgumentParser()
parser.add_argument("-t", "--text", help="text to show", default="Hello World", type=str)

args = parser.parse_args()

# Generate ASCII art using pyfiglet
text = args.text
ascii_art = pyfiglet.figlet_format(text, font='slant')

# Print colored ASCII art using termcolor
print(ascii_art)
