import argparse
import pdb
import os

from pathlib import Path
import shutil

def parse_args():
    parser = argparse.ArgumentParser(description = "Program for transforming mat files to correct directory structure for preprocessing")

        
    parser.add_argument(
        '-i', '--input_dir',
        type=str,
        required=False,
        default='C:/Users/jonathanle/downloads/lge_data/data',
        help='Path to directory containing .mat files'
    )

    parser.add_argument(
        "-o",
        "--output-dir", 
        type = str,
        required = False, 
        help='Output directory to create the directory structure',
        default = "./data/Matlab"
    )
    
    args = parser.parse_args()
    
    # Validate that the input directory exists
    if not os.path.isdir(args.input_dir):
        parser.error(f'Input directory {args.input_dir} does not exist')
    
    return args

def reorganize_patient_files(input_dir, output_dir):
    """
    Reorganizes patient files from:
        input_dir/patientid_PSIR.mat
    to:
        output_dir/patientid/patientid_PSIR.mat

    Args:
        input_dir (str or Path): Input directory containing .mat files
        output_dir (str or Path): Output directory for reorganized structure
    
    Returns:
        tuple: (success_count, error_count, list of errors)
    """
    # Convert to Path objects
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Track operations
    success_count = 0
    error_count = 0
    errors = []
    
    try:
        # Create output directory if it doesn't exist
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Find all .mat files matching the pattern
        mat_files = list(input_path.glob("*_PSIR.mat"))
        
        if not mat_files:
            raise ValueError(f"No *_PSIR.mat files found in {input_dir}")
            
        for mat_file in mat_files:
            try:
                # Extract patient ID (everything before _PSIR.mat)
                patient_id = mat_file.stem.replace('_PSIR', '')
                
                # Create patient directory
                patient_dir = output_path / patient_id
                patient_dir.mkdir(exist_ok=True)
                
                # Copy file to new location
                destination = patient_dir / mat_file.name
                shutil.copy2(mat_file, destination)
                
                success_count += 1
                print(f"Successfully processed: {mat_file.name}")
                
            except Exception as e:
                error_count += 1
                error_msg = f"Error processing {mat_file.name}: {str(e)}"
                errors.append(error_msg)
                print(error_msg)
        
        return success_count, error_count, errors
        
    except Exception as e:
        return 0, 1, [f"Fatal error: {str(e)}"]

def verify_mat_files(root_dir):
    errors = []
    for folder in Path(root_dir).iterdir():
        if folder.is_dir():
            mat_files = list(folder.glob('*.mat'))
            if len(mat_files) != 1:
                errors.append(f"{folder.name}: {len(mat_files)} .mat files found")
    return errors
# Example usage
if __name__ == "__main__":
    # Example directory paths
    input_directory = "raw_data"
    output_directory = "organized_data"
    
    successes, failures, error_list = reorganize_patient_files(input_directory, output_directory)
    
    print(f"\nReorganization complete!")
    print(f"Successfully processed: {successes} files")
    print(f"Failed to process: {failures} files")
    
    if error_list:
        print("\nErrors encountered:")
        for error in error_list:
            print(f"- {error}")

def main():
    args = parse_args()

    reorganize_patient_files(args.input_dir, args.output_dir)    


    pdb.set_trace()
    return
    
if __name__ == "__main__":
    main() 