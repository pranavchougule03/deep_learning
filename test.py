import requests

image_path = 'data/2b827.png'
api_url = 'http://localhost:5000/ocr'

files = {'image': open(image_path, 'rb')}  # Open image in binary mode
response = requests.post(api_url, files=files)  # Use 'files' parameter

# Handle response as before
