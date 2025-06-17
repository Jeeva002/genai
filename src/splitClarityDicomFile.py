import os
import pydicom
import shutil
from collections import defaultdict
import logging
import re

def setup_logging():
    """Setup logging to track processing and errors."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('dicom_processing.log'),
            logging.StreamHandler()
        ]
    )

def is_dicom_file(filepath):
    """Check if a file is a DICOM file."""
    try:
        # Check file extension first
        if not filepath.lower().endswith(('.dcm', '.dicom', '.ima')):
            # Try to read as DICOM anyway (some DICOM files have no extension)
            try:
                pydicom.dcmread(filepath, stop_before_pixels=True)
                return True
            except:
                return False
        return True
    except:
        return False

def clean_folder_name(name):
    """Clean a string to be safe for folder names."""
    if not name or name.strip() == "":
        return "Unknown"
    
    # Remove or replace invalid characters for folder names
    name = str(name).strip()
    # Replace problematic characters
    invalid_chars = r'[<>:"/\\|?*\n\r\t]'
    name = re.sub(invalid_chars, '_', name)
    # Remove multiple consecutive underscores
    name = re.sub(r'_+', '_', name)
    # Remove leading/trailing underscores and dots
    name = name.strip('_.')
    # Limit length
    if len(name) > 100:
        name = name[:100]
    
    return name if name else "Unknown"

def get_dicom_metadata(dicom_file_path):
    """Extract study type from DICOM file and folder name from file path."""
    try:
        # Read DICOM file metadata only (faster)
        ds = pydicom.dcmread(dicom_file_path, stop_before_pixels=True)
        
        # Get study type (scan type)
        study_type = "Unknown_Study"
        
        # Primary method: Series Description
        if hasattr(ds, 'SeriesDescription') and ds.SeriesDescription:
            study_type = str(ds.SeriesDescription).strip()
        
        # Secondary method: Protocol Name
        elif hasattr(ds, 'ProtocolName') and ds.ProtocolName:
            study_type = str(ds.ProtocolName).strip()
        
        # Tertiary method: Study Description
        elif hasattr(ds, 'StudyDescription') and ds.StudyDescription:
            study_type = str(ds.StudyDescription).strip()
        
        # Quaternary method: Modality + Body Part
        elif hasattr(ds, 'Modality'):
            modality = str(ds.Modality) if ds.Modality else "Unknown"
            body_part = ""
            if hasattr(ds, 'BodyPartExamined') and ds.BodyPartExamined:
                body_part = f"_{str(ds.BodyPartExamined)}"
            study_type = f"{modality}{body_part}"
        
        # Clean up the study type string
        study_type = study_type.replace('\n', ' ').replace('\r', ' ')
        study_type = ' '.join(study_type.split())  # Remove extra whitespace
        study_type = clean_folder_name(study_type)
        
        # Get folder name (parent directory of the DICOM file)
        folder_name = os.path.basename(os.path.dirname(dicom_file_path))
        if not folder_name or folder_name == "":
            folder_name = "Root_Folder"
        
        folder_name = clean_folder_name(folder_name)
        
        # Get additional metadata for better organization
        study_date = ""
        if hasattr(ds, 'StudyDate') and ds.StudyDate:
            study_date = str(ds.StudyDate)
        
        study_time = ""
        if hasattr(ds, 'StudyTime') and ds.StudyTime:
            study_time = str(ds.StudyTime)[:6]  # HHMMSS format, take first 6 chars
        
        return {
            'study_type': study_type,
            'folder_name': folder_name,
            'study_date': study_date,
            'study_time': study_time
        }
        
    except Exception as e:
        logging.warning(f"Error reading DICOM file {dicom_file_path}: {str(e)}")
        return {
            'study_type': "Error_Reading_File",
            'folder_name': "Unknown_Folder",
            'study_date': "",
            'study_time': ""
        }

def create_folder_structure(base_path, study_type, folder_name):
    """Create folder structure: base_path/study_type/folder_name/"""
    try:
        study_folder = os.path.join(base_path, study_type)
        target_folder = os.path.join(study_folder, folder_name)
        
        # Create directories if they don't exist
        os.makedirs(target_folder, exist_ok=True)
        
        return target_folder
    except Exception as e:
        logging.error(f"Error creating folder structure: {str(e)}")
        return None

def find_dicom_files(root_folder):
    """Recursively find all DICOM files in the folder structure."""
    dicom_files = []
    
    for root, dirs, files in os.walk(root_folder):
        for file in files:
            filepath = os.path.join(root, file)
            if is_dicom_file(filepath):
                dicom_files.append(filepath)
    
    return dicom_files

def organize_dicom_files(source_folder, destination_folder):
    """Organize DICOM files by study type and folder name."""
    setup_logging()
    
    if not os.path.exists(source_folder):
        logging.error(f"Source folder does not exist: {source_folder}")
        return None
    
    # Create destination folder if it doesn't exist
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder, exist_ok=True)
        logging.info(f"Created destination folder: {destination_folder}")
    
    logging.info(f"Starting DICOM organization from: {source_folder}")
    logging.info(f"Destination folder: {destination_folder}")
    
    # Statistics tracking
    organization_stats = {
        'total_files': 0,
        'organized_files': 0,
        'error_files': 0,
        'study_types': defaultdict(int),
        'folders_per_study': defaultdict(set)
    }
    
    # Find all DICOM files
    logging.info("Searching for DICOM files...")
    dicom_files = find_dicom_files(source_folder)
    
    if not dicom_files:
        logging.warning("No DICOM files found!")
        return None
    
    organization_stats['total_files'] = len(dicom_files)
    logging.info(f"Found {len(dicom_files)} DICOM files")
    
    # Process each DICOM file
    processed = 0
    for dicom_file in dicom_files:
        try:
            # Get metadata
            metadata = get_dicom_metadata(dicom_file)
            study_type = metadata['study_type']
            folder_name = metadata['folder_name']
            
            # Create folder structure
            destination_path = create_folder_structure(destination_folder, study_type, folder_name)
            
            if destination_path:
                # Generate unique filename to avoid conflicts
                original_filename = os.path.basename(dicom_file)
                base_name, ext = os.path.splitext(original_filename)
                
                # Add study date/time to filename if available for uniqueness
                if metadata['study_date'] and metadata['study_time']:
                    unique_filename = f"{base_name}_{metadata['study_date']}_{metadata['study_time']}{ext}"
                else:
                    unique_filename = original_filename
                
                destination_file = os.path.join(destination_path, unique_filename)
                
                # Handle filename conflicts
                counter = 1
                while os.path.exists(destination_file):
                    name_without_ext, ext = os.path.splitext(unique_filename)
                    destination_file = os.path.join(destination_path, f"{name_without_ext}_{counter:03d}{ext}")
                    counter += 1
                
                # Copy the file
                shutil.copy2(dicom_file, destination_file)
                
                # Update statistics
                organization_stats['organized_files'] += 1
                organization_stats['study_types'][study_type] += 1
                organization_stats['folders_per_study'][study_type].add(folder_name)
                
                logging.debug(f"Organized: {dicom_file} -> {destination_file}")
            else:
                organization_stats['error_files'] += 1
                logging.error(f"Failed to create destination path for: {dicom_file}")
            
            processed += 1
            
            if processed % 100 == 0:
                logging.info(f"Processed {processed}/{len(dicom_files)} files...")
                
        except Exception as e:
            organization_stats['error_files'] += 1
            logging.error(f"Error processing file {dicom_file}: {str(e)}")
    
    logging.info(f"Completed processing {processed} files")
    return organization_stats

def write_organization_report(stats, output_file="dicom_organization_report.txt"):
    """Write organization results to a report file."""
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("DICOM FILES ORGANIZATION REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            # Summary statistics
            f.write(f"SUMMARY:\n")
            f.write(f"Total DICOM files found: {stats['total_files']}\n")
            f.write(f"Successfully organized: {stats['organized_files']}\n")
            f.write(f"Files with errors: {stats['error_files']}\n")
            f.write(f"Total study types: {len(stats['study_types'])}\n\n")
            
            # Study types breakdown
            f.write("STUDY TYPES BREAKDOWN:\n")
            f.write("-" * 40 + "\n")
            
            sorted_studies = sorted(stats['study_types'].items(), key=lambda x: x[1], reverse=True)
            
            for i, (study_type, file_count) in enumerate(sorted_studies, 1):
                folder_count = len(stats['folders_per_study'][study_type])
                percentage = (file_count / stats['total_files']) * 100 if stats['total_files'] > 0 else 0
                f.write(f"{i:2d}. {study_type:<40} : {file_count:4d} files, {folder_count:3d} folders ({percentage:5.1f}%)\n")
            
            f.write("\n" + "=" * 50 + "\n\n")
            
            # Detailed breakdown by folders per study type
            f.write("FOLDERS PER STUDY TYPE:\n")
            f.write("=" * 50 + "\n\n")
            
            for study_type, file_count in sorted_studies:
                folders = sorted(list(stats['folders_per_study'][study_type]))
                f.write(f"\nSTUDY TYPE: {study_type}\n")
                f.write(f"Total Files: {file_count}, Total Folders: {len(folders)}\n")
                f.write("-" * 30 + "\n")
                
                for j, folder in enumerate(folders, 1):
                    f.write(f"  {j:2d}. {folder}\n")
                f.write("\n")
        
        logging.info(f"Organization report written to: {output_file}")
        return True
        
    except Exception as e:
        logging.error(f"Error writing organization report: {str(e)}")
        return False

def main():
    """Main function to run the DICOM organization."""
    # Configuration
    source_folder = 'c:/Users/Welcome/Documents/clarity'  # Source folder with DICOM files
    destination_folder = 'c:/Users/Welcome/Documents/organized_dicom'  # Where to organize files
    
    if not source_folder:
        print("No source folder path provided. Exiting.")
        return
    
    # Remove quotes if present
    source_folder = source_folder.strip('"\'')
    destination_folder = destination_folder.strip('"\'')
    
    print(f"Source folder: {source_folder}")
    print(f"Destination folder: {destination_folder}")
    print(f"Starting DICOM file organization...\n")
    
    # Organize DICOM files
    stats = organize_dicom_files(source_folder, destination_folder)
    
    if stats is None:
        print("No DICOM files found or error occurred.")
        return
    
    # Write organization report
    report_file = "dicom_organization_report.txt"
    if write_organization_report(stats, report_file):
        print(f"\nOrganization complete!")
        print(f"Report saved to: {report_file}")
        print(f"Log file saved to: dicom_processing.log")
        
        # Print summary to console
        print(f"\nSUMMARY:")
        print(f"Total files found: {stats['total_files']}")
        print(f"Successfully organized: {stats['organized_files']}")
        print(f"Files with errors: {stats['error_files']}")
        print(f"Study types found: {len(stats['study_types'])}")
        
        print(f"\nTop 5 study types:")
        sorted_studies = sorted(stats['study_types'].items(), key=lambda x: x[1], reverse=True)
        for i, (study_type, count) in enumerate(sorted_studies[:5], 1):
            folder_count = len(stats['folders_per_study'][study_type])
            print(f"  {i}. {study_type}: {count} files, {folder_count} folders")
        
        print(f"\nFiles organized in: {destination_folder}")
        print("Folder structure: Study_Type/Source_Folder_Name/dicom_files")
    else:
        print("Error writing organization report.")

if __name__ == "__main__":
    main()