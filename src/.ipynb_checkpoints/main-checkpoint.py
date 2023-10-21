import subprocess
import random
import argparse

def run_trials(num_trials):
    # List of configurations
    configurations = [
        ("HERA", "A", "50"),
        ("LOFAR", "A", "50"),
        ("HERA", "B", "100"),
        ("LOFAR", "B", "100"),
        ("HERA", "C", "100"),
        ("LOFAR", "C", "100")
    ]

    for _ in range(num_trials):
        seed = str(random.randint(0, 99999))  # Generates a random seed between 0 and 99999

        for dataset, labels, epochs in configurations:
            subprocess.call(["./run_trial.sh", seed, dataset, labels, epochs, str(num_trials)])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run experiments for a specified number of trials.")
    parser.add_argument("--num_trials", type=int, default=3, help="Number of trials to run each configuration.")
    
    args = parser.parse_args()

    run_trials(args.num_trials)
