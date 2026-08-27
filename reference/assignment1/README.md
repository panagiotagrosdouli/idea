# ISCAI PC-FMCW Laser Headlamp

## Overview

This repository contains the implementation and study of a **Phase-Coded FMCW Laser Headlamp for Integrated Sensing, Communication, and Illumination (ISCAI)**.

The project was developed as part of the course **Principles of Telecommunication Systems** during the academic year **2025–2026**.

The main objective is to reproduce and analyze the results of the paper:

> S. Liu, T. Sun, X. Shu, J. Song and Y. Dong,  
> “Phase-coded FMCW Laser Headlamp for Integrated Sensing, Communication, and Illumination,”  
> IEEE Photonics Technology Letters, 2025.

## Project Description

Modern intelligent connected vehicles require the combination of sensing, communication, and illumination in a unified and efficient platform. Instead of using separate hardware for radar sensing, optical communication, and adaptive headlights, the studied system integrates all three functions into a single laser-based headlamp architecture.

The proposed system uses a **phase-coded frequency modulated continuous wave (PC-FMCW)** signal. The FMCW waveform enables high-resolution sensing and ranging, while phase coding allows high-speed communication without destroying the sensing capability of the signal. At the same time, the laser source contributes to adaptive vehicle illumination.

## Objectives

The goals of this project are:

- Reproduce the main results of the reference paper.
- Simulate the PC-FMCW signal model.
- Study the sensing and communication subsystems.
- Analyze range-Doppler processing for target detection.
- Examine the adaptive driving beam illumination concept.
- Investigate the multidimensional Hough Transform tracking method.
- Propose a possible improvement to the original system or algorithm.

## System Architecture

The ISCAI system combines three main functionalities:

### 1. Sensing

The reflected PC-FMCW signal is received by the ego vehicle and processed to estimate target distance and motion. After coherent detection and signal processing, a range-Doppler map is generated using FFT-based processing.

### 2. Communication

Information bits are embedded into the FMCW waveform using phase coding, specifically differential phase shift keying (DPSK). This enables high-data-rate optical communication while maintaining compatibility with the sensing function.

### 3. Illumination

The laser headlamp also supports adaptive driving beam control. The system can adjust the illumination pattern to improve visibility while reducing glare for other vehicles.

## Methodology

The project follows the general processing chain below:

```text
PC-FMCW signal generation
        ↓
Phase coding / DPSK modulation
        ↓
Optical transmission
        ↓
Echo reception and coherent detection
        ↓
Group delay filtering
        ↓
Range-Doppler processing
        ↓
Target detection
        ↓
Trajectory estimation using Hough Transform
        ↓
Adaptive illumination control
````

## Tracking Algorithm

A key part of the studied paper is the use of a **Multidimensional Hough Transform (MHT)** for track-before-detect target tracking.

Instead of detecting targets frame by frame, the algorithm accumulates information over multiple observations. This improves robustness in noisy environments and helps detect weak or maneuvering targets.

The method includes:

* Projection of 3D spatio-temporal point clouds into 2D planes.
* Independent Hough Transform voting on each projection.
* AND-logic fusion to reject clutter.
* Rolling-window reconstruction for non-linear trajectories.

## Repository Structure

```text
ISCAI_pc_fmcw/
├── notebooks/
│   └── ISCAI_PC_FMCW.ipynb
├── docs/
│   └── ISCAI_Technical_Report.pdf
├── README.md
└── requirements.txt


```

## Technologies Used

* Python
* Jupyter Notebook
* NumPy
* Matplotlib
* SciPy
* Signal processing techniques
* FMCW radar principles
* Hough Transform based tracking

## How to Run

### Option 1: Google Colab

Open the notebook directly in Google Colab and run all cells.

### Option 2: Local Execution

Clone the repository:

```bash
git clone https://github.com/PanagiotaGr/ISCAI_pc_fmcw.git
cd ISCAI_pc_fmcw
```

Install the required Python libraries:

```bash
pip install numpy matplotlib scipy jupyter
```

Open the notebook:

```bash
jupyter notebook notebooks/ISCAI_PC_FMCW.ipynb
```



## Authors

* **58523 –  Grosdouli Panagiota**
* **58767 –  Kokorotsikou Agni Ioanna**

## Supervisors

* **Boulogeorgos A.Alexandros-Apostolos**
* **Chrysologou Athanasios**

## Course Information

**Course:** Principles of Telecommunication Systems
**Type:** Research Project
**Academic Year:** 2025–2026




