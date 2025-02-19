import os
import sys
from pathlib import Path

s

def analyze_directory(path):
   total_files = 0
   empty_dirs = 0
   first_level_dirs = 0
   
   # Count first level directories
   first_level_dirs = len([d for d in os.listdir(path) 
                         if os.path.isdir(os.path.join(path, d))])
   
   # Walk through all directories for files and empty dirs
   for root, dirs, files in os.walk(path):
       total_files += len(files)
       if not dirs and not files:
           empty_dirs += 1

   return total_files, empty_dirs, first_level_dirs

def main():
   if len(sys.argv) != 2:
       print("Usage: python script.py <directory_path>")
       sys.exit(1)
   
   target_dir = Path(sys.argv[1])
   
   if not target_dir.exists() or not target_dir.is_dir():
       print("Error: Invalid directory path")
       sys.exit(1)
   
   files, empty_dirs, first_level = analyze_directory(target_dir)
   print(f"Total files: {files}")
   print(f"Empty directories: {empty_dirs}")
   print(f"First-level directories: {first_level}")

if __name__ == "__main__":
   main()