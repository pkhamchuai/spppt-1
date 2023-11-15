import subprocess

dataset = [0, 1, 2, 3]
for i in dataset:
    # Run the SP batch file
    subprocess.run(['python', 'run_RANSAC.py', '--model', 'RANSAC', '--sup', '1', '--dataset', str(i)])
