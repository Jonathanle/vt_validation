import pydicom
import matplotlib.pyplot as plt
import os
import json
from pathlib import Path
import shutil
import zipfile
import argparse

# Default series descriptions
SeriesDescriptions = [
    "MDE FLASH PSIR_PSIR",
    "MDE FLASH PSIR_MAG",
    "LATE GAD PSIR SERIES",
    "LATE GAD MAG SERIES"
]


# Considerations ---> How od i know if an image is LGE? shoudl i use PSIR or MAG Image sany distinguishiung qualities
# whether to select SA TI 320 Delayed - seems to be a mag image

def filter_LGE_images(path, destination, SeriesDescriptions=SeriesDescriptions):
    """
    Filters DICOM files based on SeriesDescription and maintains directory structure.
    
    Args:
        path: Source directory containing patient folders
        destination: Destination directory for filtered files
        SeriesDescriptions: List of series descriptions to filter for
    """
    dest_root = Path(destination)
    dest_root.mkdir(parents=True, exist_ok=True)
    created_dirs = set()
    patient_dir = Path(path).name
    new_patient_dir = f"{patient_dir}_new"
    dest_path = dest_root / new_patient_dir
    dest_path.mkdir(parents=True, exist_ok=True)

    for root, dirs, files in os.walk(path):
        for file in files:
            try:
                file_path = Path(root) / file
                try:
                    ds = pydicom.dcmread(file_path, force=True)
                    if ds.SeriesDescription not in allSeriesDescriptions:
                        allSeriesDescriptions.append(str(ds.SeriesDescription))

                    if hasattr(ds, 'SeriesDescription') and ds.SeriesDescription in SeriesDescriptions:
                        rel_path = Path(root).relative_to(path)
                        dest_path = dest_root / new_patient_dir / rel_path
                        dest_path.mkdir(parents=True, exist_ok=True)
                        created_dirs.add(dest_path)
                        shutil.copy2(file_path, dest_path / file)
                        print(f"Copied: {file} to {dest_path}")
                except Exception as e:
                    print(f"Error reading {file}: {str(e)}")
                    continue
            except Exception as e:
                print(f"Error processing {file}: {str(e)}")
                continue

    print("\nCleaning up empty directories...")
    for dir_path in sorted(created_dirs, reverse=True):
        try:
            if not any(dir_path.iterdir()):
                dir_path.rmdir()
                print(f"Removed empty directory: {dir_path}")
        except Exception as e:
            print(f"Error removing directory {dir_path}: {str(e)}")

    return dest_root / new_patient_dir

def explore_dicomdir(dicomdir_path):
    """Explore contents of a DICOMDIR file"""
    dicom = pydicom.dcmread(dicomdir_path, force=True)
    print(dir(dicom))
    print(dicom.Modality)
    print(dicom.SeriesDescription)
    print(dicom.PatientSex)
    print(dicom.InstitutionName)
    print(dicom.LargestImagePixelValue)
    print(dicom.pixel_array)
    plt.imshow(dicom.pixel_array)
    print(type(dicom.pixel_array))
    plt.show()

def unzip_to_folder(zip_path):
    """Unzip file to folder"""
    parent_dir = os.path.dirname(zip_path)
    base_name = os.path.splitext(os.path.basename(zip_path))[0]
    extract_path = os.path.join(parent_dir, base_name)
    
    if not os.path.exists(extract_path):
        os.makedirs(extract_path)
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        file_names = zip_ref.namelist()
        common_prefix = os.path.commonprefix(file_names)
        if common_prefix == base_name + '/':
            extract_path = parent_dir
        zip_ref.extractall(extract_path)
    
    print(f"Unzipped to: {extract_path}")
    return extract_path

def zip_directory(directory_path, output_path=None):
    """Zip a directory"""
    directory_path = Path(directory_path)
    
    if output_path is not None:
        output_path = Path(output_path)
        if output_path.is_dir() or str(output_path).endswith('/'):
            output_path = output_path / f"{directory_path.name}.zip"
    else:
        output_path = directory_path.parent / f"{directory_path.name}.zip"
    
    print(f"Creating zip file: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zip_ref:
        for file_path in directory_path.rglob('*'):
            relative_path = file_path.relative_to(directory_path)
            if file_path.is_file():
                print(f"Adding: {relative_path}")
                zip_ref.write(file_path, relative_path)
    
    print(f"Zip file created at: {output_path}")
    return output_path

def get_inner_directory(extract_path):
    """Get inner directory from extracted path"""
    contents = os.listdir(extract_path)
    directories = [d for d in contents if os.path.isdir(os.path.join(extract_path, d))]
    if len(directories) == 0: 
        raise Exception(f"Error 0 directories found in path {extract_path}")
    if len(directories) == 1:
        inner_dir_path = os.path.join(extract_path, directories[0])
        return inner_dir_path
    elif len(directories) == 2:
        if directories[0] == "__MACOSX":
            return os.path.join(extract_path, directories[1])
        elif directories[1] == "__MACOSX":
            return os.path.join(extract_path, directories[0])
        else:
            raise Exception(f"Expected Mac OS X file for 2 directories {extract_path}")
    else:
        raise Exception(f"Error: 3 directories found: extract path: {extract_path} \n directories: {directories}")

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Process DICOM files from zip archives and filter by series description.'
    )
    parser.add_argument(
        '--input-dir',
        default='./patients/',
        help='Input directory containing zip files (default: ./patients/)'
    )
    parser.add_argument(
        '--no-zip',
        action='store_true',
        help='Skip creating zip files for processed patient cases'
    )
    parser.add_argument(
        '--output-dir',
        default='dest/',
        help='Output directory for processed files (default: dest/)'
    )
    parser.add_argument(
        '--series-desc-file',
        default="target_series_descriptions.json",
        help='List of series descriptions to filter (default: predefined list)'
    )
    parser.add_argument(
        '--keep-temp',
        action='store_true',
        help='Keep temporary directories after processing'
    )
    parser.add_argument(
        '--explore',
        help='Explore a single DICOMDIR file instead of processing'
    )
    return parser.parse_args()

def main():
    args = parse_args()
    global allSeriesDescriptions
    allSeriesDescriptions = []


    with open(args.series_desc_file) as file:
        SeriesDescriptions = json.load(file)

    if args.explore:
        explore_dicomdir(args.explore)
        return

    parent_path = Path(args.input_dir)
    zip_files = [item for item in parent_path.iterdir() if item.suffix.lower() == '.zip']

    for patient_zip in zip_files:
        print(f"Working on {patient_zip}")


        extract_path = unzip_to_folder(patient_zip)
        patient_directory = get_inner_directory(extract_path)
        print(f"Processing patient directory: {extract_path} /{patient_directory}")

        new_directory_path = filter_LGE_images(
            patient_directory,
            args.output_dir,
            SeriesDescriptions
        )
        
        if not args.no_zip:
            # Create zip and clean up directories
            zip_directory(new_directory_path, args.output_dir)
            if not args.keep_temp:
                print("Cleaning up temporary directories...")
                shutil.rmtree(extract_path)
                shutil.rmtree(new_directory_path)
            else:
                print("Keeping temporary directories as requested")
        else:
            # Only clean up the extract path, keep the processed directory
            print(f"Skipping zip file creation - keeping processed directory: {new_directory_path}")
            if not args.keep_temp:
                print("Cleaning up original extract directory...")
                shutil.rmtree(extract_path)
            else:
                print("Keeping all temporary directories as requested")

    print("\nAll Series Descriptions found:", allSeriesDescriptions)
    with open("SeriesDescriptions.json", "w") as file:
        json.dump(allSeriesDescriptions, file)

if __name__ == '__main__':
    allSeriesDescriptions = []
    main()