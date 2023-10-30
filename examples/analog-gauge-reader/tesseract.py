import cv2
import pytesseract
import sys
import os 

# Load the image
image = cv2.imread(sys.argv[1])

# Perform OCR to extract the text
custom_config = r'--oem 3 --psm 6 outputbase digits'  # Config to only recognize digits
extracted_text = pytesseract.image_to_string(image, config=custom_config)

# Clean up the extracted text (remove non-digit characters and spaces)
cleaned_text = "".join(filter(str.isdigit, extracted_text))

# Convert the cleaned text to numeric value
meter_value = int(cleaned_text)

# Display the result
print("Meter Reading:", meter_value)