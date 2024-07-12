import os

def ping_directory_os(path):
    if os.path.exists(path) and os.path.isdir(path):
        print(f"The directory '{path}' exists and is accessible.")
    else:
        print(f"The directory '{path}' does not exist or is not accessible.")

# Example usage
ping_directory_os("/media/gaston/gaston1/DEV/ACTIA/workstation/Models")
