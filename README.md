# Skeleton Analysis

## Overview

The **Skeleton Analysis** package provides tools to process spatial graphs exported from Amira. It allows for the correction of collapsed vessels, extraction of topological metrics, and visualization of the processed networks.

![tree example](assets/images/tree_example.png)

## Features

The package enables:

- **Correction of collapsed vessels** in spatial graphs
- **Extraction of vascular and network metrics**, including:
  - Topological Generation
  - Strahler Order
  - K-means Cluster Order
  - Tortuosity
  - Mean Radius
  - Murray’s Law
  - Branching Angle (between child vessels and parent-child connections)
  - Intervessel Length-Diameter Ratio
  - Branching Ratio
- **Visualization** of computed metrics in Amira
- **Graph merging** for combined analysis of multiple spatial graphs
- **Detection and correction of bad edges**
- **Outlier correction** for accurate metric computation

## Installation

```sh
# Clone the repository from GitLab
git clone https://github.com/JosephBrunet/skeleton-analysis.git
cd skeleton-analysis

# Install skeleton-analysis
pip install .
```

⚠️ **Warning:** To avoid dependency conflicts, it's recommended to use a virtual environment

## Usage

### Step 1: Prepare Input Files

- Export your Amira spatial graph as an ASCII file.
- Identify the root node IDs using Amira.
- Save the spatial graph in `.am` format.

### Step 2: Run the Ordering Script

```python
from skeleton_analysis import run_ordering

input_file = "path/to/spatial_graph.am"
output_file = "path/to/output_graph.am"
run_ordering(input_file, output_file)
```

This process corrects the spatial graph and computes the **Strahler Order** and **Topological Generation**.

### Step 3: Visualize and Smooth the Network

- Load the output graph in Amira.
- Perform multi-scale smoothing (optional, can be automated in the future).
- Save the smoothed graph.

### Step 4: Merge Networks (if applicable)

If your dataset consists of multiple spatial graphs, use:

```python
from skeleton_analysis import add_spatial_graphs

input_files = ["graph1.am", "graph2.am", "graph3.am"]
output_file = "merged_graph.am"
add_spatial_graphs(input_files, output_file)
```

After merging, re-run `run_ordering` to check for and correct bad edges.

### Step 5: Correct Outliers

Use the **Outliers Spatial Graph** function to correct any remaining outliers in the processed graph.

### Step 6: Compute and Visualize Metrics

Once the spatial graph is finalized, compute and visualize all the extracted metrics:

```python
from skeleton_analysis import compute_metrics

metrics = compute_metrics("final_graph.am")
print(metrics)
```

This step generates various plots and outputs the computed values.

## Example Datasets

Three test datasets representing a kidney network are available:

- [Component 1](https://www.dropbox.com/scl/fi/umhuiabl7p9hvcab2al5m/50um-LADAF-2021-17_labels_finalised_2_connected_component1.Spatial-Graph.am?rlkey=644v5me9sewmd4y1mg4r52e3h&dl=0)
- [Component 4](https://www.dropbox.com/scl/fi/vsqrnn9gh73xbtxochbiz/50um-LADAF-2021-17_labels_finalised_2_connected_component4.Spatial-Graph.am?rlkey=1z5q77hnvvw8738ilzx6ast87&dl=0)
- [Component 9](https://www.dropbox.com/scl/fi/eym4xrbqb8w1q17ofov3d/50um-LADAF-2021-17_labels_finalised_2connected_component9.Spatial-Graph.am?rlkey=j8hpzrzmnnihx45cb4cu5uh49&dl=0)

The corresponding raw segmentations:

- [Component 1 TIFF](https://www.dropbox.com/scl/fi/rd74fn79yrhnzoywss6n5/50um-LADAF-2021-17_labels_finalised_2_cc1.tif?rlkey=0vv3tzbltgmah07fzr5dri7fp&dl=0)
- [Component 4 TIFF](https://www.dropbox.com/scl/fi/fy1srgl9mv4to8331a3kc/50um-LADAF-2021-17_labels_finalised_2_cc4.tif?rlkey=ef538sm9d6a0tnomag3weyoym&dl=0)
- [Component 9 TIFF](https://www.dropbox.com/scl/fi/ij5xk7be7dcjwghdf0zhi/50um-LADAF-2021-17_labels_finalised_2_cc9.tif?rlkey=w1evnurimlygyprtos4fn5e0j&dl=0)

## Contributing

Contributions are welcome! To contribute:

1. **Clone the repository** from GitHub.
   ```sh
   git clone https://github.com/JosephBrunet/skeleton-analysis.git
   ```
2. **Create a feature branch:**
   ```sh
   git checkout -b feature-branch
   ```
3. **Commit your changes:**
   ```sh
   git commit -m "Add new feature"
   ```
4. **Push to GitHub:**
   ```sh
   git push origin feature-branch
   ```
5. **Create a Pull Request (PR)!** 🎉

---

## Issues & Support

If you encounter any issues or have feature requests, please open an issue on GitLab.

## Future Improvements

- Automate **multi-scale smoothing**
- Improve **graph merging** to automatically correct bad edges
- Enhance **outlier correction** workflow

## License

This project is licensed under the **MIT License**.

## Contact

For questions or support, feel free to reach out to:

- **Claire Walsh** – [c.walsh.11@ucl.ac.uk](mailto:c.walsh.11@ucl.ac.uk)
- **Joseph Brunet** – [j.brunet@ucl.ac.uk](mailto:j.brunet@ucl.ac.uk)

Thank you for using **skeleton-analysis**! --🚌
