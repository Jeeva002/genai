import os
import cv2
from paddleocr import PPStructure,draw_structure_result,save_structure_res

table_engine = PPStructure(show_log=True, image_orientation=True)

save_folder = 'c:\\Users\\Welcome\\Desktop\\IMG2DICOM\\output.jpg'
img_path = 'c:\\Users\\Welcome\\Desktop\\IMG2DICOM\\adding_image_to_table.jpg'
img = cv2.imread(img_path)
result = table_engine(img)
save_structure_res(result, save_folder,os.path.basename(img_path).split('.')[0])

for line in result:
    line.pop('img')
    print(line)

from PIL import Image

font_path = 'c:\\Users\\Welcome\\Downloads\\simfang.ttf' # PaddleOCR下提供字体包
image = Image.open(img_path).convert('RGB')
im_show = draw_structure_result(image, result,font_path=font_path)
im_show = Image.fromarray(im_show)
im_show.save('result.jpg')

#Table Recognition

# import os
# import cv2
# from paddleocr import PPStructure, save_structure_res

# table_engine = PPStructure(layout=False, show_log=True)

# save_folder = 'c:\\Users\\Welcome\\Desktop\\IMG2DICOM\\output22.jpg'
# img_path = 'c:\\Users\\Welcome\\Pictures\\Screenshots\\Screenshot 2025-06-06 170020.png'
# img = cv2.imread(img_path)
# result = table_engine(img)
# save_structure_res(result, save_folder, os.path.basename(img_path).split('.')[0])

# for line in result:
#     line.pop('img', None)  # Safely remove 'img' if present

#     # Print the full dictionary for inspection
#     print(line)

#     # Get bounding box if available
#     if 'bbox' in line:
#         x_min, y_min, x_max, y_max = line['bbox']
#         print(f"Bounding Box: Min X: {x_min}, Min Y: {y_min}, Max X: {x_max}, Max Y: {y_max}")
#     else:
#         print("No bounding box found for this item.")
# cv2.imshow('image',img)
# cv2.waitKey(0)



#layout analysis

# import os
# import cv2
# from paddleocr import PPStructure,save_structure_res

# table_engine = PPStructure(table=False, ocr=False, show_log=True)

# save_folder = 'c:\\Users\\Welcome\\Desktop\\IMG2DICOM'
# img_path = 'c:\\Users\\Welcome\\Desktop\\IMG2DICOM\\adding_image_to_table.jpg'
# img = cv2.imread(img_path)
# result = table_engine(img)
# save_structure_res(result, save_folder, os.path.basename(img_path).split('.')[0])

# for line in result:
#     line.pop('img')
#     print(line)

#layout analysis + table recognition

# import os
# import cv2
# from paddleocr import PPStructure,draw_structure_result,save_structure_res

# table_engine = PPStructure(show_log=True)

# save_folder = 'c:\\Users\\Welcome\\Desktop\\IMG2DICOM\out'
# img_path = 'c:\\Users\\Welcome\\Desktop\\IMG2DICOM\\adding_image_to_table.jpg'
# img = cv2.imread(img_path)
# result = table_engine(img)
# save_structure_res(result, save_folder,os.path.basename(img_path).split('.')[0])

# for line in result:
#     line.pop('img')
#     print(line)

# from PIL import Image
# font_path = 'c:\\Users\\Welcome\\Downloads\\simfang.ttf'  # font provieded in PaddleOCR
# image = Image.open(img_path).convert('RGB')
# im_show = draw_structure_result(image, result,font_path=font_path)
# im_show = Image.fromarray(im_show)
# im_show.save('result.jpg')
