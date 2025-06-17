import pydicom

class USRegionLocation():
    def __init__(self, dicomData):
        self.dicomPath = dicomData
        # Initialize the region coordinates as instance variables
        self.x0, self.y0, self.x1, self.y1 = self.getMeasurementRegions()
        # Don't return anything from __init__
             
    def getMeasurementRegions(self):
        self.dicomData = pydicom.dcmread(self.dicomPath)  
        self.studyInfo = self.dicomData[0x0008, 0x1030]
        self.imageType = self.dicomData[0x0008, 0x0008]
        print(self.imageType)
        print(self.studyInfo)
        return self.getAllAvailableRegions()

    def getAllAvailableRegions(self):
        print("DICOM file loaded successfully", self.dicomData)
        print(self.dicomData)
        regions_sequence = self.dicomData[(0x0018, 0x6011)]  # Common tag for ultrasound regions
        print(f"Found ultrasound regions:")
        min_x0 = []
        min_y0 = []
        max_x1 = []
        max_y1 = []
        
        for i, region in enumerate(regions_sequence):
            print(f"\nRegion {i+1}:")
            
            # Access Region Location Min X0
            if (0x0018, 0x6018) in region:
                min_x0.append(region[(0x0018, 0x6018)].value)
                print(f"  Region Location Min X0: {min_x0}")
            
            # Access other region parameters
            if (0x0018, 0x601a) in region:
                min_y0.append(region[(0x0018, 0x601a)].value)
                print(f"  Region Location Min Y0: {min_y0}")
            
            if (0x0018, 0x601c) in region:
                max_x1.append(region[(0x0018, 0x601c)].value)
                print(f"  Region Location Max X1: {max_x1}")
            
            if (0x0018, 0x601e) in region:
                max_y1.append(region[(0x0018, 0x601e)].value)
                print(f"  Region Location Max Y1: {max_y1}")
        
        print(f"Image size: {self.dicomData.Rows} x {self.dicomData.Columns}")
        return min_x0, min_y0, max_x1, max_y1
    
    def get_coordinates(self):
        """Method to get the coordinates after object creation"""
        return self.x0, self.y0, self.x1, self.y1

# Usage example:
# dicom_file = "c:\\Users\\Welcome\\Desktop\\IMG2DICOM\\I0000010"
# us_region = USRegionLocation(dicom_file)
# minx0, miny0, maxx1, maxy1 = us_region.get_coordinates()