import sys

def compare_files(file1, file2):
    try:
        with open(file1, 'r') as f1, open(file2, 'r') as f2:
            lines1 = f1.readlines()
            lines2 = f2.readlines()

            if lines1 == lines2:
                print("The files are identical.")
                return True
            else:
                print("The files are different.")
                for i, (line1, line2) in enumerate(zip(lines1, lines2), start=1):
                    if line1 != line2:
                        print(f"Difference found on line {i}:")
                        print(f"File 1: {line1.strip()}")
                        print(f"File 2: {line2.strip()}")
                        break
                if len(lines1) != len(lines2):
                    print("One file has extra lines.")
                return False
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python compare_files.py <file1> <file2>")
        sys.exit(1)

    file1 = sys.argv[1]
    file2 = sys.argv[2]
    compare_files(file1, file2)

# python compare_module.py file1.txt file2.txt
