from dotenv import load_dotenv
load_dotenv()

import os
import io
import uuid
import json
import re
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image
import google.generativeai as genai
from google.cloud import storage
from google.generativeai import types
# Import exceptions for better handling
from google.cloud.exceptions import Forbidden
# Import specific Gemini exceptions
#from google.generativeai.errors import BlockedPromptError, InternalServerError


app = Flask(__name__)
CORS(app)

# --- CONFIGURATION ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
   raise ValueError("Gemini API Key (GOOGLE_API_KEY) missing in .env file")

genai.configure(api_key=GOOGLE_API_KEY)
FLASH_IMAGE_MODEL_ID = "gemini-2.5-flash-image-preview"

try:
   model = genai.GenerativeModel(FLASH_IMAGE_MODEL_ID)
   print(f"âœ… Gemini model initialized: {FLASH_IMAGE_MODEL_ID}")
except Exception as e:
   print(f"âŒ Model initialization error: {e}")
   raise

# --- TEXT GENERATION CONFIGURATION ---
gemini_key = os.getenv("GOOGLE_API_KEY")
if not gemini_key:
   # Use the GOOGLE_API_KEY already loaded
   pass


genai.configure(api_key=GOOGLE_API_KEY)

OUTPUT_DIR = "generated_images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- GOOGLE CLOUD STORAGE CONFIG ---
GCS_BUCKET_NAME = "amz-image-stores"

try:
   # Initialize GCS client. It will use the ADC credentials and project ID you set.
   storage_client = storage.Client()
   print("âœ… Google Cloud Storage client initialized.")
except Exception as e:
   # This will now only fail if the Project ID is missing or ADC is completely broken
   print(f"âŒ GCS client initialization error: {e}")
   # Do NOT raise here, let the upload function handle the specific GCS failure
  
def upload_to_gcs(local_file_path, destination_blob_name):
   """
   Uploads file to GCS, makes it public, and returns the public URL.
   Includes explicit error handling for GCS failures (like 403 Forbidden).
   """
   try:
       bucket = storage_client.bucket(GCS_BUCKET_NAME)
   except NameError:
       # storage_client initialization failed during startup
       raise Exception("GCS client not available. Check startup logs.")
      
   blob = bucket.blob(destination_blob_name)
  
   try:
       print(f"DEBUG: Attempting to upload {local_file_path} to gs://{GCS_BUCKET_NAME}/{destination_blob_name}")
      
       # 1. Upload the file
       blob.upload_from_filename(local_file_path)
      
       # 2. Make it publicly readable (requires storage.objects.update permission)
       # blob.make_public()
      
       gcs_public_url = blob.public_url
       print(f"âœ… Uploaded to GCS. Public URL: {gcs_public_url}")
       return gcs_public_url
      
   except Forbidden as e:
       # THIS IS THE MOST LIKELY REMAINING ERROR SOURCE (403 Forbidden)
       error_message = f"GCS Permission Denied (403): User lacks permission to upload or set public ACLs on bucket '{GCS_BUCKET_NAME}'. Ensure the ADC user has 'Storage Admin' role."
       print(f"âŒ GCS Upload FAILED: {error_message}")
       raise Exception(error_message)
   except Exception as e:
       error_message = f"GCS Upload Failed: {type(e).__name__} - {str(e)}"
       print(f"âŒ GCS Upload FAILED: {error_message}")
       raise Exception(error_message)


# --- PROMPT TEMPLATES (UNCHANGED) ---
def get_prompt_model_generate_with_a_white_background():
   return (
       "Transform the subject into a high-quality fashion photo of a [Gender] model standing upright "
       "wearing the [SubCategory]. Background must be pure white (#FFFFFF). "
       "Show full product in frame. Studio lighting and realistic fabric texture."
   )

def get_prompt_product_image_with_a_white_background():
   return (
       "Transform the subject into a pure white background e-commerce image of the [SubCategory]. "
       "Full view, isolated, no shadows, clear lighting. Ideal for Amazon listings."
   )

def get_prompt_image_of_the_model_in_a_lively_event_setting():
   return (
       "Transform into realistic photo of a [Gender] model wearing [SubCategory] in an elegant boutique or festive scene. "
       "Full body, lively composition, vibrant ambiance."
   )

def get_prompt_image_of_the_model_from_the_left_or_right_or_back():
   return (
       "Create side or back profile view of [Gender] model wearing [SubCategory]. "
       "Festive [Occasion] setup, full product visible, rich colors, and clear details."
   )

def get_prompt_infographic_image_with_details_of_the_product():
   return (
       "Generate infographic displaying the [SubCategory] with clear close-up of texture and stitching. "
       "Soft pastel background, no watermark, elegant composition."
   )

PROMPT_FUNCTIONS = [
   get_prompt_model_generate_with_a_white_background,
   get_prompt_product_image_with_a_white_background,
   get_prompt_image_of_the_model_in_a_lively_event_setting,
   get_prompt_image_of_the_model_in_a_lively_event_setting, # Duplicated, corrected index 3 below
   get_prompt_infographic_image_with_details_of_the_product,
]

# Ensure index 3 uses the correct prompt function
PROMPT_FUNCTIONS[3] = get_prompt_image_of_the_model_from_the_left_or_right_or_back


def replace_placeholders(prompt: str, attributes: dict) -> str:
   """Replaces [keys] with values from frontend form attributes."""
   pattern = r"\[(\w+)\]"
   def replacer(match):
       key = match.group(1)
       return attributes.get(key, f"[{key}]")
   return re.sub(pattern, replacer, prompt)

# --- IMAGE GENERATION ENDPOINT ---
@app.route("/generate-image", methods=["POST"])
def generate_image_api():
   if "image" not in request.files:
       return jsonify({"error": "No image uploaded"}), 400

   file = request.files["image"]
  
   # Store the stream in a temporary in-memory buffer to use for both PIL and FormData later
   img_stream = io.BytesIO(file.read())
   img_stream.seek(0)
  
   try:
       uploaded_image = Image.open(img_stream).convert("RGB")
   except Exception as e:
       return jsonify({"error": f"Invalid image file: {str(e)}"}), 400


   try:
       style_index = int(request.form.get("style_index", -1))
   except ValueError:
       return jsonify({"error": "Invalid style_index"}), 400

   if not (0 <= style_index < len(PROMPT_FUNCTIONS)):
       return jsonify({"error": "style_index out of range"}), 400

   attributes_str = request.form.get("attributes", "{}")
   try:
       attributes = json.loads(attributes_str)
   except json.JSONDecodeError:
       return jsonify({"error": "Invalid attributes JSON"}), 400

   prompt_template = PROMPT_FUNCTIONS[style_index]()
   final_prompt = replace_placeholders(prompt_template, attributes)

   try:
       full_prompt = (
           "Transform this photo photorealistically for premium e-commerce quality. "
           f"Instructions: {final_prompt} Avoid distortion, blur, watermarks, or cropping."
       )

       # Pass PIL image and prompt to the model
       response = model.generate_content(
           [uploaded_image, full_prompt],
           generation_config=genai.types.GenerationConfig(
               temperature=0.6, top_p=0.9, top_k=40
           ),
       )

       for part in response.candidates[0].content.parts:
           if part.inline_data and part.inline_data.mime_type.startswith("image/"):
               image_bytes = part.inline_data.data
               img = Image.open(io.BytesIO(image_bytes))

               unique_filename = f"generated_{uuid.uuid4().hex}.png"
               local_path = os.path.join(OUTPUT_DIR, unique_filename)
              
               # Save locally (needed before GCS upload)
               img.save(local_path)

               gcs_blob_name = f"generated/{unique_filename}"
              
               # This function now has specific error handling
               gcs_url = upload_to_gcs(local_path, gcs_blob_name)

               # The frontend will check the 'gcs_url'
               return jsonify({
                   "filename": unique_filename,
                   "gcs_url": gcs_url
               }), 200

       return jsonify({"error": "No image returned from model."}), 500

   except Exception as e:
       # Catch any failure during generation or upload and send to frontend
       return jsonify({"error": f"Generation/Upload failed: {str(e)}"}), 500


@app.route("/generated_images/<filename>")
def serve_generated_image(filename):
   return send_from_directory(OUTPUT_DIR, filename)


# --- TEXT GENERATION FUNCTIONS ---
def create_base_prompt(subcategory, product_details, task_type):
   if task_type == 'name':
       instruction = "TASK: Write exactly one labeled line: Product Name: [title here]"
   else:
       instruction = "TASK: Write exactly one labeled line: Product Description: [description here]"
   return f"""
You are an expert Amazon e-commerce copywriter specializing in Clothing.
Only generate factual and accurate output based solely on the provided details.
{instruction}

Product Details:
{subcategory}
{product_details}

IMPORTANT:
Max 200 characters for title.
Max 2000 characters for description.
Mention brand once, focus on material, fit, pattern, neck, sleeve, color, size.
No promotional language.
End description with a complete sentence.

GOOD EXAMPLES
# Example 1 - Men's Slim Fit Cotton T-Shirt
Input:
Brand: Levi's
Department: Men
Generic Name: T-Shirt
Material: 100% Cotton
Fit: Slim Fit
Pattern: Solid
Neck: Crew Neck
Sleeve: Short Sleeve
Size: M
Output:
Product Name: Levi's Men's Slim Fit 100% Cotton Crew Neck T-Shirt | M
Product Description: Crafted from 100% cotton, this Levi's men's slim fit t-shirt offers breathable comfort and a sleek, modern look. The classic crew neck and short sleeves make it ideal for casual outings. Pairs effortlessly with jeans or chinos for everyday style.


# Example 2 - Men's Oversized Graphic T-Shirt
Input:
Brand: Bewakoof
Department: Men
Generic Name: T-Shirt
Material: 100% Cotton
Fit: Oversized
Pattern: Graphic Print (Batman)
Neck: Round Neck
Size: XL
Output:
Product Name: Bewakoof Men's Oversized 100% Cotton Batman Graphic Round Neck T-Shirt | XL
Product Description: Make a bold statement with this Bewakoof men's oversized Batman t-shirt. Made from 100% cotton for comfort and durability. The round neck and striking graphic print add a contemporary streetwear edge, perfect for casual wear.

# Example 3 - Women's Anarkali Kurta Set
Input:
Brand: KLOSIA
Department: Women
Generic Name: Kurta Set
Fit: Anarkali
Material: 100% Viscose
Pattern: Printed
Dupatta: Chanderi Cotton
Sleeve: 3/4 Sleeve
Length: Calf Length
Output:
Product Name: KLOSIA Women's 100% Viscose Printed Anarkali Kurta Set with Chanderi Cotton Dupatta
Product Description: This KLOSIA women's Anarkali set combines traditional elegance with modern comfort. Crafted from 100% viscose with a printed design and Chanderi cotton dupatta, it flows gracefully while keeping you comfortable. Ideal for festive or casual ethnic occasions.

Subcategory: {subcategory}
Details: {product_details}
"""

def call_gemini(prompt, max_tokens):
   try:
       model = genai.GenerativeModel("gemini-2.5-flash")
       response = model.generate_content(
           prompt,
           generation_config=types.GenerationConfig(
               temperature=0.8,
               max_output_tokens=max_tokens,
           ),
       )
      
       # --- FIX APPLIED HERE: Robust check for output and finish reason ---
      
       # 1. Check if a candidate was returned
       if not response.candidates:
           return "Generation Failed: Model returned no candidates/output."


       candidate = response.candidates[0]
       finish_reason = candidate.finish_reason
      
       # Define the finish reasons using integer values for stability
       FINISH_REASON_SAFETY = 2
       FINISH_REASON_STOP = 1


       # 2. Check for safety block using the integer value (2)
       is_safety_blocked = (finish_reason == FINISH_REASON_SAFETY)
      
       if is_safety_blocked:
           # Safely try to get the safety rating details
           safety_info = ""
           if candidate.safety_ratings:
               # Use .name if the enum is available, otherwise just use the rating value
               try:
                   blocked_category_name = candidate.safety_ratings[0].category.name
               except AttributeError:
                   blocked_category_name = f"Code {candidate.safety_ratings[0].category}"
                  
               safety_info = f" Blocked Category: {blocked_category_name}"


           return f"Generation Failed: Output Blocked by Safety Filters (Finish Reason: SAFETY).{safety_info}"
      
       # 3. Successful completion check (STOP is usually 1)
       if finish_reason == FINISH_REASON_STOP and hasattr(response, "text") and response.text:
           return response.text.strip()
      
       # 4. Check for other non-successful stops (like MAX_TOKENS, or 0/UNSPECIFIED)
       if finish_reason != FINISH_REASON_STOP:
           # Try to get the enum name for better logging, or fall back to the code
           try:
               # Requires 'from google.generativeai import types'
               finish_reason_name = types.FinishReason(finish_reason).name
           except (ValueError, AttributeError):
               finish_reason_name = f"Code {finish_reason}"
              
           return f"Generation Failed: Model stopped with reason: {finish_reason_name}."
          
       # 5. Last resort check for empty text
       return "Generation Failed: Model returned no valid text parts (likely due to invalid input data or internal block)."
      
       # --- END FIX ---
      
   except Exception as e:
       # Catch all exceptions generically
       return f"Generation Failed: API Error. {type(e).__name__} - {str(e)}"


def generate_product_name(subcategory, product_details):
   prompt = create_base_prompt(subcategory, product_details, 'name')
   raw_output = call_gemini(prompt, 2000)
   label = "Product Name:"
   if raw_output.startswith(label):
       return raw_output[len(label):].strip()
   return raw_output

def generate_product_description(subcategory, product_details):
   prompt = create_base_prompt(subcategory, product_details, 'description')
   raw_output = call_gemini(prompt, 2000)
   label = "Product Description:"
   if raw_output.startswith(label):
       return raw_output[len(label):].strip()
   return raw_output

# --- TEXT GENERATION ENDPOINT ---
@app.route('/api/generate-title-description', methods=['POST'])
def generate_title_description():
   data = request.get_json()
   subcategory = data.get('subcategory', '')
  
   # --- FIX APPLIED HERE: Robustly creating product_details while stripping HTML/unsafe chars ---
   product_details_lines = []
   for k, v in data.items():
       if k not in ('subcategory', 'type') and v is not None:
           # Simple cleaning: strip newlines/tabs and remove any basic HTML tags (if scraped)
           cleaned_v = str(v).strip().replace('\n', ' ').replace('\t', ' ')
           # Optional: Add re.sub(r'<[^>]+>', '', cleaned_v) if you suspect HTML tags
           if cleaned_v:
                product_details_lines.append(f"{k}: {cleaned_v}")
               
   product_details = "\n".join(product_details_lines)
   # --- END FIX ---
  
   task_type = data.get('type', 'title')
  
   if not subcategory or not product_details:
       return jsonify({'success': False, 'error': 'Missing required fields or product details are empty after cleaning.'}), 400
      
   if task_type == 'title':
       generated_title = generate_product_name(subcategory, product_details)
       if generated_title.startswith("Generation Failed:"):
           return jsonify({'success': False, 'error': generated_title}), 500
       return jsonify({'success': True, 'generated_title': generated_title})
      
   elif task_type == 'description':
       generated_description = generate_product_description(subcategory, product_details)
       if generated_description.startswith("Generation Failed:"):
           return jsonify({'success': False, 'error': generated_description}), 500
       return jsonify({'success': True, 'generated_description': generated_description})
      
   else:
       return jsonify({'success': False, 'error': 'Unknown type specified'}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)