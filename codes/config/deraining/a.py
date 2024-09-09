import subprocess

args = [
        "--test_path /Users/nshaff3r/Downloads/image-restoration-sde/codes/data/datasets/nypl/testH/LQ", 
        "--output_dir /Users/nshaff3r/Downloads/image-restoration-sde/codes/config/deraining/results/masks",
        "--input_size full_size"
        ]
try:
    command = ["python3", "Bringing-Old-Photos-Back-to-Life/Global/detection.py"] + args
    result = subprocess.run(command, check=True, text=True, capture_output=True)
    print("Mask output:", result.stdout)
except subprocess.CalledProcessError as e:
    print("Errors:", e)
    print("ERROR: ", subprocess.CalledProcessError)