#!/usr/bin/env python

import argparse
import subprocess
import os

def build_conda_envs_from_venvs(venvs_folder="venvs"):
    """
    For each subfolder in venvs_folder, check for requirements.txt and build a conda env.
    The env name will be the subfolder name. If the env exists, prompt the user before reinstalling.
    """
    print("Attempting to build conda environments from the specified virtual environments folder...")
    # Get list of existing conda envs
    result = subprocess.run(["conda", "env", "list"], capture_output=True, text=True)
    existing_envs = set()
    for line in result.stdout.splitlines():
        if line and not line.startswith("#") and "/" in line:
            env_name = line.split("/")[-1]
            existing_envs.add(env_name)

    for subdir in os.listdir(venvs_folder):
        env_path = os.path.join(venvs_folder, subdir)
        req_file = os.path.join(env_path, "environment.yml")
        if os.path.isfile(req_file):
            env_name = subdir
            print(env_name)
            if env_name in existing_envs:
                
                print(f"Skipping {env_name} as it already exists.")
                continue
            else:
                print(f"Creating conda environment '{env_name}' from {req_file}...")
                #conda create --name <env> --file <this file>
                subprocess.run(["conda", "env", "create", "--file", req_file], check=True)
                print(f"Environment '{env_name}' created and requirements installed.")


def run_in_conda_env(env_name, script_path, script_args=None):
    """
    Run a Python script in the specified conda environment.
    """
    cmd = ["conda", "run", "-n", env_name, "-v", "python", script_path]
    if script_args:
        cmd.extend(script_args)
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Run the full reaction classification pipeline.")
    parser.add_argument("-i", "--input", required=True, help="Path to the initial input CSV file")
    parser.add_argument("-o", "--output", required=True, help="Path to the final output CSV file")
    args = parser.parse_args()

    # Build conda environments from the specified virtual environments folder
    build_conda_envs_from_venvs("venvs")

    # File paths
    input_file = args.input
    intermediate1 = "output/single_product_one_step_reactions.csv"
    intermediate2 = "output/single_product_one_step_reactions_with_templates.csv"
    output_file = args.output

    # Run the ProcessReactions.py script in the specified conda environment
    run_in_conda_env(
        "rxnmapper-env",
        "src/ProcessReactions.py",
        ["-i", input_file, "-o", intermediate1]
    )

    # Run the ReactionTemplates.py script in the specified conda environment
    run_in_conda_env(
        "rxn-utils-env",
        "src/ReactionTemplates.py",
        ["-i", intermediate1, "-o", intermediate2]
    )

    # Run the ClassifyReactions.py script in the specified conda environment
    run_in_conda_env(
        "rxnmapper-env",
        "src/ClassifyReactions.py",
        ["-i", intermediate2, "-o", output_file]
    )