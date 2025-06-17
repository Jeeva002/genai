import os
import pydicom
from collections import defaultdict
import logging

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

def get_scan_type(dicom_file_path):
    """Extract scan type from DICOM file."""
    try:
        # Read DICOM file metadata only (faster)
        ds = pydicom.dcmread(dicom_file_path, stop_before_pixels=True)
        
        # Try different DICOM tags to determine scan type
        scan_type = "Unknown"
        
        # Primary method: Series Description
        if hasattr(ds, 'SeriesDescription') and ds.SeriesDescription:
            scan_type = str(ds.SeriesDescription).strip()
        
        # Secondary method: Protocol Name
        elif hasattr(ds, 'ProtocolName') and ds.ProtocolName:
            scan_type = str(ds.ProtocolName).strip()
        
        # Tertiary method: Study Description
        elif hasattr(ds, 'StudyDescription') and ds.StudyDescription:
            scan_type = str(ds.StudyDescription).strip()
        
        # Quaternary method: Modality + Body Part
        elif hasattr(ds, 'Modality'):
            modality = str(ds.Modality) if ds.Modality else "Unknown"
            body_part = ""
            if hasattr(ds, 'BodyPartExamined') and ds.BodyPartExamined:
                body_part = f" - {str(ds.BodyPartExamined)}"
            scan_type = f"{modality}{body_part}"
        
        # Clean up the scan type string
        scan_type = scan_type.replace('\n', ' ').replace('\r', ' ')
        scan_type = ' '.join(scan_type.split())  # Remove extra whitespace
        
        return scan_type if scan_type else "Unknown"
        
    except Exception as e:
        logging.warning(f"Error reading DICOM file {dicom_file_path}: {str(e)}")
        return "Error_Reading_File"

def find_dicom_files(root_folder):
    """Recursively find all DICOM files in the folder structure."""
    dicom_files = []
    
    for root, dirs, files in os.walk(root_folder):
        for file in files:
            filepath = os.path.join(root, file)
            if is_dicom_file(filepath):
                dicom_files.append(filepath)
    
    return dicom_files

def categorize_dicom_files(main_folder):


    
    """Categorize DICOM files by scan type."""
    setup_logging()
    
    if not os.path.exists(main_folder):
        logging.error(f"Main folder does not exist: {main_folder}")
        return None
    
    logging.info(f"Starting DICOM categorization in: {main_folder}")
    
    # Dictionary to store scan types and their counts
    scan_categories = defaultdict(int)
    scan_files = defaultdict(list)  # Store file paths for each category
    
    # Find all DICOM files
    logging.info("Searching for DICOM files...")
    dicom_files = find_dicom_files(main_folder)
    
    if not dicom_files:
        logging.warning("No DICOM files found!")
        return None
    
    logging.info(f"Found {len(dicom_files)} DICOM files")
    
    # Process each DICOM file
    processed = 0
    for dicom_file in dicom_files:
        try:
            scan_type = get_scan_type(dicom_file)
            if scan_type == 'TRIMESTER III/US\
                ':
                print("trimester",dicom_file)
            scan_categories[scan_type] += 1
            scan_files[scan_type].append(dicom_file)
            processed += 1
            
            if processed % 100 == 0:
                logging.info(f"Processed {processed}/{len(dicom_files)} files...")
                
        except Exception as e:
            logging.error(f"Error processing file {dicom_file}: {str(e)}")
    
    logging.info(f"Completed processing {processed} files")
    return scan_categories, scan_files

def write_results_to_file(scan_categories, scan_files, output_file="dicom_scan_categories.txt"):
    """Write categorization results to a text file."""
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("DICOM SCAN CATEGORIZATION RESULTS\n")
            f.write("=" * 50 + "\n\n")
            
            # Summary statistics
            total_files = sum(scan_categories.values())
            total_categories = len(scan_categories)
            
            f.write(f"SUMMARY:\n")
            f.write(f"Total DICOM files processed: {total_files}\n")
            f.write(f"Total scan categories found: {total_categories}\n\n")
            
            # Sort categories by count (descending)
            sorted_categories = sorted(scan_categories.items(), key=lambda x: x[1], reverse=True)
            
            f.write("SCAN CATEGORIES (sorted by count):\n")
            f.write("-" * 40 + "\n")
            
            for i, (scan_type, count) in enumerate(sorted_categories, 1):
                percentage = (count / total_files) * 100
                f.write(f"{i:2d}. {scan_type:<40} : {count:4d} files ({percentage:5.1f}%)\n")
            
            f.write("\n" + "=" * 50 + "\n\n")
            
            # Detailed file listing for each category
            f.write("DETAILED FILE LISTING BY CATEGORY:\n")
            f.write("=" * 50 + "\n\n")
            
            for scan_type, count in sorted_categories:
                f.write(f"\nCATEGORY: {scan_type}\n")
                f.write(f"COUNT: {count} files\n")
                f.write("-" * 30 + "\n")
                
                for file_path in scan_files[scan_type]:
                    f.write(f"  {file_path}\n")
                f.write("\n")
        
        logging.info(f"Results written to: {output_file}")
        return True
        
    except Exception as e:
        logging.error(f"Error writing results to file: {str(e)}")
        return False

def main():
    """Main function to run the DICOM categorization."""
    # Change this to your main folder path
    main_folder = 'c:/Users/Welcome/Documents/clarity'
    
    if not main_folder:
        print("No folder path provided. Exiting.")
        return
    
    # Remove quotes if present
    main_folder = main_folder.strip('"\'')
    
    print(f"Processing DICOM files in: {main_folder}")
    
    # Categorize DICOM files
    result = categorize_dicom_files(main_folder)
    
    if result is None:
        print("No DICOM files found or error occurred.")
        return
    
    scan_categories, scan_files = result
    
    # Write results to file
    output_file = "dicom_scan_categories.txt"
    if write_results_to_file(scan_categories, scan_files, output_file):
        print(f"\nCategorization complete!")
        print(f"Results saved to: {output_file}")
        print(f"Log file saved to: dicom_processing.log")
        
        # Print summary to console
        print(f"\nSUMMARY:")
        print(f"Total files processed: {sum(scan_categories.values())}")
        print(f"Total categories found: {len(scan_categories)}")
        print(f"\nTop 5 scan types:")
        sorted_cats = sorted(scan_categories.items(), key=lambda x: x[1], reverse=True)
        for i, (scan_type, count) in enumerate(sorted_cats[:5], 1):
            print(f"  {i}. {scan_type}: {count} files")
    else:
        print("Error writing results to file.")

if __name__ == "__main__":
    main()