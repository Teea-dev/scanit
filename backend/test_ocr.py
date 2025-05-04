# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# import pytesseract
# from PIL import Image

# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# def detect_table_and_cells(image_path):
#     """
#     Detects table structure and cells in the image.
#     Returns the original image with lines drawn on it and coordinates of cells.
#     """
#     # Read the image
#     img = cv2.imread(image_path)
#     if img is None:
#         print(f"Error: Could not read image at {image_path}")
#         return None, []

#     # Convert to grayscale
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
#     # Apply threshold to get image with only black and white
#     _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
#     # Detect horizontal lines
#     horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
#     horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    
#     # Detect vertical lines
#     vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
#     vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
    
#     # Combine horizontal and vertical lines
#     table_skeleton = cv2.add(horizontal_lines, vertical_lines)
    
#     # Find contours for the cells
#     contours, _ = cv2.findContours(table_skeleton, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
#     # Draw all contours on a copy of the original image
#     img_with_contours = img.copy()
#     cv2.drawContours(img_with_contours, contours, -1, (0, 255, 0), 2)
    
#     # Find the bounding rectangles for each contour (cell)
#     cells = []
#     for contour in contours:
#         x, y, w, h = cv2.boundingRect(contour)
#         # Filter out very small contours that might be noise
#         if w > 30 and h > 20:
#             cells.append((x, y, w, h))
    
#     return img_with_contours, cells

# def extract_text_from_cells(img, cells):
#     """
#     Extract text from each detected cell in the table.
#     """
#     cell_texts = []
    
#     for i, (x, y, w, h) in enumerate(cells):
#         # Extract the cell from the image
#         cell_img = img[y:y+h, x:x+w]
        
#         # Convert to grayscale
#         gray_cell = cv2.cvtColor(cell_img, cv2.COLOR_BGR2GRAY)
        
#         # Apply adaptive threshold
#         thresh_cell = cv2.adaptiveThreshold(gray_cell, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
#                                            cv2.THRESH_BINARY, 11, 2)
        
#         # Convert to PIL Image
#         pil_img = Image.fromarray(thresh_cell)
        
#         # Extract text
#         custom_config = r'--oem 3 --psm 6'
#         text = pytesseract.image_to_string(pil_img, config=custom_config)
        
#         # Clean up the text
#         text = text.strip()
        
#         if text:  # Only append non-empty cells
#             cell_texts.append({"cell_id": i, "position": (x, y, w, h), "text": text})
    
#     return cell_texts

# def extract_table_from_image(image_path):
#     """
#     Main function to extract table structure and content from an image.
#     """
#     # Read the original image
#     original_img = cv2.imread(image_path)
#     if original_img is None:
#         print(f"Error: Could not read image at {image_path}")
#         return
    
#     # Detect table structure
#     img_with_contours, cells = detect_table_and_cells(image_path)
    
#     if not cells:
#         print("No table cells detected. Trying alternative approach...")
#         # TODO: Implement alternative approach if needed
#         return
    
#     # Extract text from each cell
#     cell_texts = extract_text_from_cells(original_img, cells)
    
#     # Display the original and processed images
#     plt.figure(figsize=(15, 10))
    
#     plt.subplot(1, 2, 1)
#     plt.title("Original Image")
#     plt.imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
#     plt.axis('off')
    
#     plt.subplot(1, 2, 2)
#     plt.title("Detected Table Structure")
#     plt.imshow(cv2.cvtColor(img_with_contours, cv2.COLOR_BGR2RGB))
#     plt.axis('off')
    
#     plt.tight_layout()
#     plt.show()
    
#     # Print extracted cell texts
#     print(f"Extracted {len(cell_texts)} table cells with text:")
#     for cell in cell_texts:
#         print(f"Cell {cell['cell_id']} at position {cell['position']}:")
#         print(f"    {cell['text']}")
    
#     # Save the table structure image
#     cv2.imwrite("table_structure.jpg", img_with_contours)
#     print("\nSaved table structure image as 'table_structure.jpg'")
    
#     # Try to reconstruct the table structure
#     print("\nAttempting to reconstruct table structure...")
#     # Sort cells by their y-coordinate (row) and then by x-coordinate (column)
#     sorted_cells = sorted(cell_texts, key=lambda c: (c['position'][1], c['position'][0]))
    
#     # Group cells by rows (cells with similar y-coordinate)
#     rows = []
#     current_row = []
#     current_y = None
#     tolerance = 20  # Tolerance for considering cells to be in the same row
    
#     for cell in sorted_cells:
#         y = cell['position'][1]
#         if current_y is None or abs(y - current_y) <= tolerance:
#             current_row.append(cell)
#             if current_y is None:
#                 current_y = y
#         else:
#             # Sort the current row by x-coordinate
#             current_row.sort(key=lambda c: c['position'][0])
#             rows.append(current_row)
#             current_row = [cell]
#             current_y = y
    
#     # Add the last row if it exists
#     if current_row:
#         current_row.sort(key=lambda c: c['position'][0])
#         rows.append(current_row)
    
#     # Print the reconstructed table
#     print("\nReconstructed Table:")
#     for i, row in enumerate(rows):
#         row_text = " | ".join([cell['text'] for cell in row])
#         print(f"Row {i}: {row_text}")
    
#     return rows

# if __name__ == "__main__":
#     # Use a raw string for the file path
#     image_path = r"C:\Users\Admin\OneDrive\Pictures\test_timetable.png"
#     extract_table_from_image(image_path)

import cv2
import pytesseract
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import time

# Set Tesseract path - adjust if needed
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def enhance_image_for_table_ocr(image_path):
    """
    Enhanced preprocessing specifically for table structures.
    """
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image at {image_path}")
        return None
        
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY, 11, 2)
    
    # Denoise
    denoised = cv2.fastNlMeansDenoising(thresh, None, 10, 7, 21)
    
    # Dilation - helps connect components and make text clearer
    kernel = np.ones((1, 1), np.uint8)
    dilated = cv2.dilate(denoised, kernel, iterations=1)
    
    return dilated

def detect_table_grid(image):
    """
    Detect table grid lines and draw them on the image.
    Returns image with detected lines.
    """
    gray = image.copy()
    
    # Apply edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    # Apply Hough Line Transform
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
    
    # Create color image to draw lines on
    lined_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(lined_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    return lined_image

def extract_table_text(image_path, debug=True, timeout=60):
    """
    Main function to extract text from tables with timeout protection
    """
    start_time = time.time()
    
    try:
        # Read the image
        original_img = cv2.imread(image_path)
        if original_img is None:
            print(f"Error: Could not read image at {image_path}")
            return None
        
        # Enhanced preprocessing for tables
        processed_img = enhance_image_for_table_ocr(image_path)
        if processed_img is None:
            return None
            
        # Detect and visualize table structure
        table_structure_img = detect_table_grid(processed_img)
        
        # Convert processed image to PIL Image for OCR
        pil_img = Image.fromarray(processed_img)
        
        # Show debug images if requested
        if debug:
            plt.figure(figsize=(15, 10))
            
            plt.subplot(1, 3, 1)
            plt.title("Original Image")
            plt.imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            
            plt.subplot(1, 3, 2)
            plt.title("Processed Image")
            plt.imshow(processed_img, cmap='gray')
            plt.axis('off')
            
            plt.subplot(1, 3, 3)
            plt.title("Detected Table Structure")
            plt.imshow(cv2.cvtColor(table_structure_img, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            
            plt.tight_layout()
            plt.show()
        
        # Save intermediate results
        output_dir = "ocr_results"
        os.makedirs(output_dir, exist_ok=True)
        cv2.imwrite(os.path.join(output_dir, "preprocessed_table.jpg"), processed_img)
        cv2.imwrite(os.path.join(output_dir, "table_structure.jpg"), table_structure_img)
        
        print("Processing with OCR - this may take a few seconds...")
        
        # Try different OCR configurations specifically for tables
        results = {}
        
        # Standard configuration with timeout check
        if time.time() - start_time < timeout:
            print("\n=== STANDARD CONFIGURATION ===")
            text_standard = pytesseract.image_to_string(pil_img)
            print(text_standard)
            results["standard"] = text_standard
        
        # Table configuration with timeout check
        if time.time() - start_time < timeout:
            print("\n=== TABLE CONFIGURATION ===")
            # PSM 6: Assume a single uniform block of text
            custom_config = r'--oem 3 --psm 6'
            text_custom = pytesseract.image_to_string(pil_img, config=custom_config)
            print(text_custom)
            results["table"] = text_custom
        
        # Table structure detection with timeout check
        if time.time() - start_time < timeout:
            print("\n=== TABLE STRUCTURED OUTPUT ===")
            try:
                # Extract structured data from the table
                # Limit to confidence above 30% to reduce noise
                table_data = pytesseract.image_to_data(pil_img, output_type=pytesseract.Output.DICT)
                print(f"Detected {len(table_data['text'])} text elements")
                
                # Filter out empty strings and low confidence results
                filtered_data = []
                for i in range(len(table_data['text'])):
                    if table_data['text'][i].strip() != '' and float(table_data['conf'][i]) > 30:
                        item = {
                            'text': table_data['text'][i],
                            'conf': table_data['conf'][i],
                            'left': table_data['left'][i],
                            'top': table_data['top'][i],
                            'width': table_data['width'][i],
                            'height': table_data['height'][i]
                        }
                        filtered_data.append(item)
                        print(f"Text: {item['text']}, Conf: {item['conf']}, Position: ({item['left']}, {item['top']})")
                
                results["structured"] = filtered_data
                
                # Try to save structured data as CSV
                with open(os.path.join(output_dir, "table_data.csv"), "w") as f:
                    f.write("text,confidence,left,top,width,height\n")
                    for item in filtered_data:
                        f.write(f"{item['text']},{item['conf']},{item['left']},{item['top']},{item['width']},{item['height']}\n")
                
            except Exception as e:
                print(f"Error in structured output: {e}")
        
        # Try specialized table extraction with timeout check
        if time.time() - start_time < timeout:
            print("\n=== SPECIALIZED TABLE EXTRACTION ===")
            try:
                # TSV output format specifically for tables
                tsv_config = r'--oem 3 --psm 6 -c preserve_interword_spaces=1 outputbase tsv'
                tsv_output = pytesseract.image_to_string(pil_img, config=tsv_config)
                
                # Save TSV output
                with open(os.path.join(output_dir, "table_output.tsv"), "w") as f:
                    f.write(tsv_output)
                    
                print("TSV output saved to table_output.tsv")
                results["tsv"] = tsv_output
                
            except Exception as e:
                print(f"Error in TSV extraction: {e}")
        
        print(f"\nProcessing completed in {time.time() - start_time:.2f} seconds")
        print(f"Results saved to '{output_dir}' directory")
        
        return results
        
    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"Error occurred after {elapsed_time:.2f} seconds: {e}")
        return None

if __name__ == "__main__":
    # Use a raw string for the file path
    image_path = r"C:\Users\Admin\OneDrive\Pictures\test_timetable.png"
    
    print("Starting table text extraction...")
    results = extract_table_text(image_path, debug=True)
    
    if results:
        print("\nSUMMARY OF EXTRACTED TEXT:")
        print("-------------------------")
        if "table" in results:
            print(results["table"])
    else:
        print("Failed to extract text from the table.")