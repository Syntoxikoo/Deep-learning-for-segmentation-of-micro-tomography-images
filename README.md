# Deep-learning-for-segmentation-of-micro-tomography-images

## Description
High resolution X-ray tomography images are used for detecting porosity, cracks and fails in materials. They support the monitoring and modeling of material deformation under chemical reactions, thermal processes and mechanical conditions. A sequence of these images taken in regular times may show the deformation of how microstructures evolve under a specific treatment.

We dispose of a large dataset forming a 3D volume, with a resolution in all directions of 22 nm. The images were acquired during a heat treatment process, with scans taken at regular intervals over the same object. They give spatial-temporal information of the material’s response under heat treatment. We aim to map the structural deformation by comparing consecutive images. One approach involves segmenting pores in the 2D slices of the volume and tracking the specific pores in subsequent images taken during the heating.

By combining deep learning techniques with semi-supervised methods, the project aims to build a robust segmentation framework capable of identifying porosities across the entire dataset, ultimately supporting downstream tasks such as deformation tracking and material characterization.

## Principles of good practice
Important: Report results and methodological decisions since the beginning.

The  code and experimental setup should be fully replicable, and all experiments must be repeatable under the same conditions to ensure the reliability and validity of the results. 

Note: We are supervising more than 50 students, so please do not come to our offices without a prior agreement via email. Kindly contact us in advance to schedule a meeting. 

## Tasks
Experimental design:  Make a diagram of the full experimental design, before start to code. Each step should include a clear description of its pros and cons. Please also include the reference(s) you used to support your decisions.  For example, if you decide to work in the frequency domain versus the spatial domain (RGB), then explain why and also explicitly define what information maybe you loose for it.

Report of literature/tools overview: please write in a report the reference object: 

authors, source tiltle (e.g., article name), proposed method, short description, why is relevant to your work, dataset used, BibTex (properly, is not enough copy/paste), link, 

A reference method:  define a baseline method and test in a public dataset to be sure is properly working.

---

## Environment Setup

### Prerequisites
- Python >= 3.13

### Installation Steps

1. **Install uv** (fast Python package installer):
   ```zsh
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Create a virtual environment**:
   ```zsh
   uv venv
   ```

3. **Activate the virtual environment**:
   ```zsh
   source .venv/bin/activate
   ```

4. **Install dependencies**:
   ```zsh
   uv sync
   ```
