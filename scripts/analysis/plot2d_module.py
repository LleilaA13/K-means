import matplotlib.pyplot as plt
import sys

def read_coordinates(file_path):
    """Read coordinates from a file and return them as a list of tuples."""
    coordinates = []
    try:
        with open(file_path, 'r') as file:
            for line in file:
                # Split the line into x and y values
                x, y = map(float, line.strip().split())
                coordinates.append((x, y))
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return []
    except ValueError as e:
        print(f"Error: Could not parse line. {e}")
        return []
    return coordinates

def read_classes(file_path):
    """Read class information from a file and return it as a list of integers."""
    classes = []
    try:
        with open(file_path, 'r') as file:
            for line in file:
                cls = int(line.strip())
                classes.append(cls)
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return []
    except ValueError as e:
        print(f"Error: Could not parse line. {e}")
        return []
    return classes

def plot_coordinates_by_class(coordinates, classes, k):
    """Plot coordinates and color points according to their class."""
    if not coordinates or not classes:
        print("No valid data to plot.")
        return

    # Generate a set of unique colors for each class (use a colormap or random colors)
    colors = plt.cm.get_cmap('tab10', k)  # Using a color map with 'k' different colors

    plt.figure(figsize=(8, 6))

    # Plot each point, colored by its class
    for i in range(len(coordinates)):
        x, y = coordinates[i]
        cls = classes[i] - 1  # Adjust class to zero-indexing
        plt.scatter(x, y, color=colors(cls), label=f"Class {cls+1}" if i == 0 else "")

    plt.title("Point Coordinates Colored by Class")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1))
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # VISUALIZATION OF KMEANS CLUSTERING ON 2D POINTCLOUDS
    coord_path, class_path, k= sys.argv[1], sys.argv[2], int(sys.argv[3])
    coordinates = read_coordinates(coord_path)
    classes = read_classes(class_path)
    plot_coordinates_by_class(coordinates, classes, k)

# python plot_module.py test_files/input2D2.inp ../../results/results_seq.txt 5