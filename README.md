# 🛰️ Remove-First-Detect-Later
> Code Repository for the paper "Remove First, Detect Later: A Counter-intuitive Approach For Detecting Radio Frequency Interference in Radio Sky Imagery."

The experiments from the study can be easily replicated with the code housed in this repository.

## 🚀 Getting Started

1. **Set Up Your Environment**:
   Navigate to the `src` directory:
```bash
   cd src
```

2. **Install the necessary packages**:

```bash
    pip install -e .
```

3. **Run the Experiments**:
Simply execute:

```bash
    python3 main.py
```

For multiple trials of each experiment, leverage the --num_trials argument:
```bash
    python3 main.py --num_trials <NUMBER>
```

ℹ️ There are 18 distinct experiments. Please be aware that executing all trials might require considerable computational resources.

4. **Review the Results**:
All trial outcomes are stored within the report directory. The provided Jupyter notebooks offer insights into the violin plots and tables.

📁 Directory Structure
Ensure the datasets LOFAR_Full_RFI_dataset and HERA-SIM_Full_RFI_dataset are appropriately placed:

Remove-First-Detect-Later  
|-- data  
|-- src  
|----|-- architectures  
|----|-- experiment  
|----|-- models  
|----|-- report  
|----|-- simulation  
|----|-- utils  

