# Analyzing BC’s Hiking Trails

## Overview
This project, conducted for CMPT 353 at Simon Fraser University, focuses on applying smoothing techniques to hiking trail data obtained from OpenStreetMap to enhance its accuracy and usability. By addressing the challenge of data noise and inconsistencies, we aimed to improve the precision of GPS trail data, making it more reliable for hikers and outdoor enthusiasts.
For more information refer to the report pdf file.

## Group Members
- Yang Liu
- Ronney Lok
- Sina MohammadiNiyaki

## Datasets
We analyzed two well-known hiking trails in British Columbia:
- Garibaldi Trail
- Upper Falls Trail in Golden Ears Park

## Techniques Employed
To refine the data, we explored various smoothing techniques, including:
- Kalman Smoothing
- Loess Smoothing
- Simple Moving Average (SMA)
- Gaussian Smoothing
- Savitzky-Golay Filter

## Instructions for Running the Project
1. Ensure `main.py` and the dataset files are in the same directory.
2. To analyze the Garibaldi dataset, execute: `python3 main.py ./data/garibaldi.gpx`
3. For the Golden Ears dataset, run: `python3 main.py ./data/upperFalls.gpx`

The results, including refined GPX files and analysis outcomes, will be saved in designated folders and can also be viewed on [mygpsfiles](https://www.mygpsfiles.com) for graphical representation.

## Contributions
- **Sina MohammadiNiyaki**: Focused on implementing and comparing multiple smoothing algorithms to determine the most effective approach for hiking trail data refinement.

## Acknowledgments
We extend our gratitude to Professor Greg Baker and the Simon Fraser University Department of Computing Science for their support and guidance throughout this project.