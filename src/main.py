import subprocess
import random

def run_trials(num_trials):
    # List of configurations
    configurations = [
        ("HERA", "AOFlagger", "50"),
        ("LOFAR", "AOFlagger", "50"),
        ("HERA", "Expert20", "100"),
        ("LOFAR", "Expert20", "100"),
        ("HERA", "Transfer", "100"),
        ("LOFAR", "Transfer", "100")
    ]

    for _ in range(num_trials):
        seed = str(random.randint(0, 99999))  # Generates a random seed between 0 and 99999

        for dataset, labels, epochs in configurations:
            subprocess.call(["./run_trial.sh", seed, dataset, labels, epochs, str(num_trials)])

# Usage example:
run_trials(10)  # This will generate 10 random seeds and run the script 60 times (6 configs x 10 seeds) with those seeds
