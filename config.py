import os

if os.name == 'posix':  # POSIX-compliant systems (Linux/Mac)
    BASE_PATH = "/hpc/za64617/projects/AutoEA_dev/"
elif os.name == 'nt':  # Windows systems
    BASE_PATH = "C:/Users/s143036/PycharmProjects/AutoEA_development/"
else:
    raise OSError("Unsupported operating system. Please configure the BASE_PATH manually.")

