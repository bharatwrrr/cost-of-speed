# Cost of Speed: Research Repository

This repository contains the code and analysis notebooks for the paper: **"Speeding incurs substantial environmental and economic costs for negligible travel time savings nationwide"**. Link and DOI will be updated here as they are available.

## Repository Structure

The repository is organized as follows:

* **src/**: Core source code directory.
    * **__init__.py**: Package initialization.
    * **access.py**: Contains core classes for preprocessing vehicle trajectories and running energy consumption simulations via FASTSim.
    * **speed_transform.py**: Contains the mathematical models used to the synthetic drive cycles from the original drive cycles from the dataset according to descriptions of the two interventions (SC1 and SC2) described in the paper.
    * **utils.py**: Provides spatial analysis tools and plotting wrappers for mapping geohashed trip data to US state geometries.
* **config.py**: Configuration file for data mount points.
* **table_scripts/**: Scripts to generate data for the tables in the manuscript.
---

## Getting Started

### Prerequisites
* [Anaconda](https://www.anaconda.com/products/distribution) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed on your system.

### Installation & Environment Setup
1. **Clone the repository**:
   ```bash
   git clone https://github.com/bharatwrrr/cost-of-speed.git
   cd cost-of-speed
   ```
2. **Create and activate the conda environment**:
    ```bash
    conda env create -f environment.yaml -n cost-of-speed
    ```
3. **Activate the environment**:
   ```bash
   conda activate cost-of-speed
   ```
4. **Data Configuration**: 
   Open `config.py` and update the data mount paths to point to your local data directory where the vehicle datasets are stored. NOTE: the raw dataset that this points to is not publicly available due to legal reasons. This is file is meant to be seen as a placeholder for import statements in `access.py`.


## Citation

If you use this code or these findings in your research, please cite the corresponding journal paper.
```bash
@article{
  jayaprakash2026speeding,
  title={Speeding incurs substantial environmental and economic costs nationwide for negligible travel time savings},
  author={Jayaprakash, Bharat and Northrop, William F},
  journal={Communications Sustainability},
  volume={1},
  number={1},
  pages={107},
  year={2026},
  publisher={Nature Publishing Group UK, London}
}
```
