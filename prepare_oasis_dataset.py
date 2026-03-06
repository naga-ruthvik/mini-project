"""
Prepare OASIS-1 cross-sectional data for AXIAL framework training.

This script:
1. Scans OASIS disc folders for subjects
2. Reads CDR scores from each subject's .txt file
3. Finds the preprocessed atlas-registered, skull-stripped MRI (T88_111/*_masked_gfc)
4. Converts Analyze (.hdr/.img) to NIfTI (.nii.gz) format
5. Creates dataset.csv with columns: subject, mri_path, diagnosis

Usage:
    python prepare_oasis_dataset.py

CDR mapping:
    CDR = 0     -> CN (Cognitively Normal)
    CDR = 0.5   -> MCI (Mild Cognitive Impairment)
    CDR >= 1    -> AD (Alzheimer's Disease)
"""

import os
import re
import glob
import nibabel as nib
import pandas as pd


# === CONFIGURATION ===
# Add paths to all your OASIS disc folders here
OASIS_ROOT = r"C:\Users\konde\main-projects\datasets\OASIS\OASIS_1"
DISC_DIRS = [os.path.join(OASIS_ROOT, f"disc{i}") for i in range(1, 13)]

# Output directory for converted NIfTI files
OUTPUT_DIR = os.path.join("data", "oasis1_nifti")

# Output CSV path (this is what you'll point config.yaml to)
OUTPUT_CSV = os.path.join("data", "oasis1_nifti", "dataset.csv")

# Which classes to include. Set to ['CN', 'AD'] to match default config.
# Set to ['CN', 'MCI', 'AD'] if you want all three classes.
INCLUDE_CLASSES = ["CN", "AD"]


def cdr_to_diagnosis(cdr_value):
    """Convert CDR score to diagnosis label."""
    cdr = float(cdr_value)
    if cdr == 0:
        return "CN"
    else:  # CDR > 0 (includes 0.5 MCI and >=1 AD)
        return "AD"


def parse_subject_info(txt_path):
    """Parse the subject .txt file to extract CDR score."""
    info = {}
    with open(txt_path, "r") as f:
        for line in f:
            line = line.strip()
            if ":" in line:
                key, value = line.split(":", 1)
                info[key.strip()] = value.strip()
    return info


def find_masked_gfc_image(subject_dir):
    """
    Find the atlas-registered, skull-stripped brain image in the subject directory.
    Looks for: PROCESSED/MPRAGE/T88_*/  *_t88_masked_gfc.hdr
    """
    # Look in T88_* subdirectories for the masked_gfc image
    pattern = os.path.join(
        subject_dir, "PROCESSED", "MPRAGE", "T88_*", "*_t88_masked_gfc.hdr"
    )
    matches = glob.glob(pattern)
    if matches:
        return matches[0]

    # Fallback: look for any masked_gfc file
    pattern = os.path.join(subject_dir, "PROCESSED", "**", "*masked_gfc.hdr")
    matches = glob.glob(pattern, recursive=True)
    if matches:
        return matches[0]

    return None


def convert_analyze_to_nifti(hdr_path, output_path):
    """Convert Analyze 7.5 (.hdr/.img) to NIfTI (.nii.gz)."""
    img = nib.load(hdr_path)
    nib.save(img, output_path)
    return output_path


def main():
    project_root = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(project_root, OUTPUT_DIR), exist_ok=True)

    records = []
    skipped = []

    for disc_dir in DISC_DIRS:
        if os.path.isabs(disc_dir):
            disc_path = disc_dir
        else:
            disc_path = os.path.join(project_root, disc_dir)

        if not os.path.exists(disc_path):
            print(f"WARNING: Disc directory not found: {disc_path}")
            continue

        # Get all subject directories
        subject_dirs = sorted(
            [
                d
                for d in os.listdir(disc_path)
                if os.path.isdir(os.path.join(disc_path, d)) and d.startswith("OAS1_")
            ]
        )

        print(f"\nProcessing {disc_dir} ({len(subject_dirs)} subjects)...")

        for subject_name in subject_dirs:
            subject_dir = os.path.join(disc_path, subject_name)

            # 1. Read the .txt file for CDR score
            txt_file = os.path.join(subject_dir, f"{subject_name}.txt")
            if not os.path.exists(txt_file):
                print(f"  SKIP {subject_name}: No .txt file found")
                skipped.append((subject_name, "no .txt file"))
                continue

            info = parse_subject_info(txt_file)
            cdr = info.get("CDR", "").strip()

            if not cdr or cdr == "N/A":
                # Empty CDR = healthy young volunteers with no clinical assessment -> CN
                diagnosis = "CN"
                print(f"  INFO {subject_name}: No CDR score (healthy control assumed)")
            else:
                diagnosis = cdr_to_diagnosis(cdr)

            # Filter by desired classes
            if diagnosis not in INCLUDE_CLASSES:
                print(
                    f"  SKIP {subject_name}: diagnosis={diagnosis} not in {INCLUDE_CLASSES}"
                )
                skipped.append((subject_name, f"diagnosis {diagnosis} filtered out"))
                continue

            # 2. Find the preprocessed brain image
            hdr_path = find_masked_gfc_image(subject_dir)
            if hdr_path is None:
                print(f"  SKIP {subject_name}: No masked_gfc image found")
                skipped.append((subject_name, "no masked_gfc image"))
                continue

            # 3. Convert to NIfTI
            # Extract subject ID (e.g., OAS1_0001 from OAS1_0001_MR1)
            subject_id = "_".join(subject_name.split("_")[:2])
            nifti_filename = f"{subject_name}_brain.nii.gz"
            nifti_path = os.path.join(project_root, OUTPUT_DIR, nifti_filename)

            try:
                convert_analyze_to_nifti(hdr_path, nifti_path)
                print(
                    f"  OK   {subject_name}: CDR={cdr} -> {diagnosis}, saved {nifti_filename}"
                )
            except Exception as e:
                print(f"  ERROR {subject_name}: Failed to convert - {e}")
                skipped.append((subject_name, f"conversion error: {e}"))
                continue

            # 4. Add record
            records.append(
                {"subject": subject_id, "mri_path": nifti_path, "diagnosis": diagnosis}
            )

    # 5. Create and save the dataset CSV
    df = pd.DataFrame(records)
    df.to_csv(os.path.join(project_root, OUTPUT_CSV), index=False)

    # Print summary
    print("\n" + "=" * 60)
    print("DATASET PREPARATION COMPLETE")
    print("=" * 60)
    print(f"\nTotal subjects processed: {len(records)}")
    print(f"Total subjects skipped:  {len(skipped)}")
    print(f"\nClass distribution:")
    for diagnosis in INCLUDE_CLASSES:
        count = len(df[df["diagnosis"] == diagnosis])
        print(f"  {diagnosis}: {count}")
    print(f"\nDataset CSV saved to: {OUTPUT_CSV}")
    print(f"NIfTI files saved to: {OUTPUT_DIR}/")

    if skipped:
        print(f"\nSkipped subjects:")
        for name, reason in skipped:
            print(f"  {name}: {reason}")

    print(f"\n--- Next Steps ---")
    print(f"1. Update config.yaml:")
    print(f"   dataset_csv: '{OUTPUT_CSV}'")
    print(f"   dataset_name: 'OASIS1'")
    print(f"2. Run: python train.py")


if __name__ == "__main__":
    main()
