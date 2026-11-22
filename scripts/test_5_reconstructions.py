import os
import subprocess
import pandas as pd

samples_to_test = [
    {'sample': '1_Cheek_cells_1', 'image_path': 'data/SW-Dataset/1_Cheek_cells/z1/1.bmp', 'wavelength': 0.000633, 'L_value': 18.81},
    {'sample': '1_Cheek_cells_2', 'image_path': 'data/SW-Dataset/1_Cheek_cells/z1/2.bmp', 'wavelength': 0.000633, 'L_value': 18.81},
    {'sample': '1_Cheek_cells_3', 'image_path': 'data/SW-Dataset/1_Cheek_cells/z1/3.bmp', 'wavelength': 0.000633, 'L_value': 18.81},
    {'sample': '8_Spirogyra_1', 'image_path': 'data/SW-Dataset/8_Spirogyra/z1/1.bmp', 'wavelength': 0.000633, 'L_value': 13.92},
    {'sample': '8_Spirogyra_2', 'image_path': 'data/SW-Dataset/8_Spirogyra/z1/2.bmp', 'wavelength': 0.000633, 'L_value': 13.92},
]

print("--- RUNNING 5 RECONSTRUCTIONS ---")
for row in samples_to_test:
    sample = row['sample']
    img_path = row['image_path']
    wave = row['wavelength']
    L = row['L_value']
    
    out_name = 'reconstruction_' + sample.replace(' ', '_') + '.png'
    
    cmd = [
        "python", "scripts/reconstruct_image.py",
        "--image", img_path,
        "--wavelength", str(wave),
        "--L", str(L),
        "--out", out_name
    ]
    
    print(f"\nProcessing {sample}...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Parse output to find z and check success
    z_pred = None
    for line in result.stdout.splitlines():
        if "Predicted Reconstruction Distance" in line:
            z_pred = line.split(":")[-1].strip()
            
    success = "Done!" in result.stdout
    
    print(f"- Predicted z: {z_pred}")
    print(f"- Output path: {out_name}")
    print(f"- Completed successfully: {success}")
    if not success:
        print(f"Error output: {result.stderr}")
