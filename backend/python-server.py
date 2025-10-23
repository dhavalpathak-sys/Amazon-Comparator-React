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
from google.cloud.exceptions import Forbidden 
from dotenv import load_dotenv

# --- FLASK AND CORS SETUP ---
app = Flask(__name__)
load_dotenv()

# Configure CORS to explicitly allow multiple origins (FIXED CORS ISSUE)
CORS(app, resources={r"/api/*": {"origins": ["http://localhost:5173", "http://localhost:3000", "http://localhost:5000"]}})

# --- API KEY & MODEL CONFIG ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("Gemini API Key (GOOGLE_API_KEY/GEMINI_KEY) missing in .env file")

genai.configure(api_key=GOOGLE_API_KEY)
FLASH_IMAGE_MODEL_ID = "gemini-2.5-flash-image-preview"

try:
    # Renamed to image_model to prevent conflict with text_model in call_gemini
    image_model = genai.GenerativeModel(FLASH_IMAGE_MODEL_ID)
    print(f"✅ Gemini Image model initialized: {FLASH_IMAGE_MODEL_ID}")
except Exception as e:
    print(f"❌ Model initialization error: {e}")
    # Raise only if the image part is strictly required, otherwise allow text to function
    # raise

OUTPUT_DIR = "generated_images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- GOOGLE CLOUD STORAGE CONFIG (Retained) ---
GCS_BUCKET_NAME = "amz-image-store"

try:
    storage_client = storage.Client()
    print("✅ Google Cloud Storage client initialized.")
except Exception as e:
    print(f"❌ GCS client initialization error: {e}")
    
def upload_to_gcs(local_file_path, destination_blob_name):
    """
    Uploads file to GCS, makes it public, and returns the public URL.
    """
    try:
        bucket = storage_client.bucket(GCS_BUCKET_NAME)
    except NameError:
        raise Exception("GCS client not available. Check startup logs.")
        
    blob = bucket.blob(destination_blob_name)
    
    try:
        print(f"DEBUG: Attempting to upload {local_file_path} to gs://{GCS_BUCKET_NAME}/{destination_blob_name}")
        blob.upload_from_filename(local_file_path)
        gcs_public_url = blob.public_url 
        print(f"✅ Uploaded to GCS. Public URL: {gcs_public_url}")
        return gcs_public_url
        
    except Forbidden as e:
        error_message = f"GCS Permission Denied (403): User lacks permission to upload or set public ACLs on bucket '{GCS_BUCKET_NAME}'. Ensure the ADC user has 'Storage Admin' role."
        print(f"❌ GCS Upload FAILED: {error_message}")
        raise Exception(error_message)
    except Exception as e:
        error_message = f"GCS Upload Failed: {type(e).__name__} - {str(e)}"
        print(f"❌ GCS Upload FAILED: {error_message}")
        raise Exception(error_message)

@app.route("/generated_images/<filename>")
def serve_generated_image(filename):
    return send_from_directory(OUTPUT_DIR, filename)

# --- UTILITY FUNCTIONS ---
def basic_clean(value):
    """Strips newlines/tabs and surrounding whitespace from a value."""
    # Ensure value is treated as a string before cleaning
    return str(value).strip().replace('\n', ' ').replace('\t', ' ')

# --- TEXT GENERATION API CALL FUNCTION (FIXED ERROR HANDLING) ---
def call_gemini(prompt, max_tokens):
    """
    Calls the Gemini API and handles safety blocks (Finish Reason 2) and 
    missing response text defensively (FIXED response.text error).
    """
    try:
        text_model = genai.GenerativeModel("gemini-2.5-flash")
        response = text_model.generate_content(
            prompt,
            generation_config=types.GenerationConfig(
                temperature=0.9,
                max_output_tokens=max_tokens,
            ),
        )
        
        # 1. Check if a candidate was returned
        if not response.candidates:
            return "Generation Failed: Model returned no candidates/output."

        candidate = response.candidates[0]
        finish_reason = candidate.finish_reason
        
        # Define the finish reasons using integer values for stability
        # 2 is SAFETY, 1 is STOP
        FINISH_REASON_SAFETY = 2
        FINISH_REASON_STOP = 1

        # 2. Check for safety block (Finish Reason 2)
        is_safety_blocked = (finish_reason == FINISH_REASON_SAFETY)
        
        if is_safety_blocked:
            safety_info = ""
            if candidate.safety_ratings:
                try:
                    # Safely get the name of the blocked category
                    blocked_category_name = candidate.safety_ratings[0].category.name
                except AttributeError:
                    blocked_category_name = f"Code {candidate.safety_ratings[0].category}"
                    
                safety_info = f" Blocked Category: {blocked_category_name}"

            return f"Generation Failed: Output Blocked by Safety Filters (Finish Reason: SAFETY).{safety_info}"
        
        # 3. Successful completion check (STOP is 1)
        if finish_reason == FINISH_REASON_STOP and hasattr(response, "text") and response.text:
            return response.text.strip()
        
        # 4. Check for other non-successful stops (like MAX_TOKENS, or 0/UNSPECIFIED)
        if finish_reason != FINISH_REASON_STOP:
            try:
                # Safely attempt to get the enum name
                finish_reason_name = types.FinishReason(finish_reason).name
            except (ValueError, AttributeError):
                finish_reason_name = f"Code {finish_reason}"
                
            return f"Generation Failed: Model stopped with reason: {finish_reason_name}."
            
        # 5. Last resort check for empty text
        return "Generation Failed: Model returned no valid text parts."
        
    except Exception as e:
        # Catch all exceptions generically
        return f"Generation Failed: API Error. {type(e).__name__} - {str(e)}"


# --- PROMPT TEMPLATES (Retained) ---

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
    get_prompt_image_of_the_model_from_the_left_or_right_or_back,
    get_prompt_infographic_image_with_details_of_the_product,
]


def replace_placeholders(prompt: str, attributes: dict) -> str:
    """Replaces [keys] with values from frontend form attributes."""
    pattern = r"\[(\w+)\]"
    def replacer(match):
        key = match.group(1)
        return attributes.get(key, f"[{key}]")
    return re.sub(pattern, replacer, prompt)

# --- IMAGE GENERATION ENDPOINT (Retained) ---
@app.route("/generate-image", methods=["POST"])
def generate_image_api():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
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

        response = image_model.generate_content(
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
                
                img.save(local_path)

                gcs_blob_name = f"generated/{unique_filename}"
                gcs_url = upload_to_gcs(local_path, gcs_blob_name)

                return jsonify({
                    "filename": unique_filename,
                    "gcs_url": gcs_url
                }), 200

        return jsonify({"error": "No image returned from model, possibly blocked."}), 500

    except Exception as e:
        return jsonify({"error": f"Generation/Upload failed: {str(e)}"}), 500


# --- TEXT PROMPT TEMPLATES (Retained) ---

def create_base_prompt(subcategory, product_details, task_type):
    # Choose label based on task type
    item_type_label = "Product Title" if task_type == 'name' else "Product Description"
    output_label = "Product Name:" if task_type == 'name' else "Product Description:"

    instruction = (
        f"TASK: Using all of the following Product Details—including any existing product name or description—generate a new, unique, high-quality Amazon {item_type_label}. "
        f"Do NOT repeat or closely paraphrase any existing title or description text verbatim. Instead, rewrite and improve them: be clear, factual, concise, and use Amazon best practices. "
        "Base your answer ONLY on the given details. Avoid promotional language. Mention brand once; focus on material, fit, pattern, neck, sleeve, color, and size. End with a complete sentence."
        "\n\n"
        f"Return your answer as exactly one labeled line for this task:\n{output_label} [output here]"
    )

    return f"""
You are an expert Amazon e-commerce copywriter specializing in Clothing.
{instruction}

Product Details (including all existing scraped names and descriptions, and all product attributes):
{subcategory}
{product_details}

IMPORTANT:
- Max 200 characters for title.
- Max 2000 characters for description.
- Keep the description long as given in examples
- Do NOT mirror any input name/description verbatim; provide a revised, value-added version.
- Follow Amazon apparel copy conventions.

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
Product Name: Levi's Men's Slim Fit 100% Cotton Crew Neck T-Shirt | Classic Solid Color Style | Breathable Everyday Wear | M
Product Description: Discover the ultimate foundation piece for your casual wardrobe with the Levi's Men's Slim Fit T-Shirt. This essential tee is crafted entirely from 100% premium cotton, offering supreme softness against the skin and reliable all-day comfort. The natural cotton fibers ensure superior breathability, making it an ideal choice for warmer climates or layered styling. Designed with a modern slim fit, the shirt contours naturally to the body without feeling restrictive, providing a sharp, contemporary silhouette that is instantly flattering. It features a timeless crew neck and practical short sleeves, ensuring easy pairing with everything from denim and chinos to light jackets. The solid color and refined finish make it versatile enough for daily errands, casual Fridays, or weekend outings. Durable construction and quality stitching ensure this Levi's shirt maintains its shape and color through repeated washing. Invest in a closet staple that defines comfortable, understated style.


# Example 2 - Women's Anarkali Kurta Set
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
Product Name: KLOSIA Women's 100% Viscose Printed Anarkali Kurta Set with Chanderi Cotton Dupatta | Festive 3/4 Sleeve Ethnic Wear
Product Description: Embrace traditional grace and modern comfort with the KLOSIA Women's Anarkali Kurta Set, meticulously designed for festive and formal occasions. The kurta features a flattering **Anarkali fit**, characterized by its flowy, flared silhouette that starts just below the bust and extends elegantly to a **calf length**, creating a beautiful sweeping effect. The entire kurta is fashioned from 100% soft viscose fabric, known for its luxurious drape, lightweight feel, and vibrant color retention. It showcases an intricate all-over printed pattern, adding depth and traditional charm to the ensemble. Completing the look is a generously sized Chanderi Cotton dupatta, which is lightweight yet provides structure and features a delicate finish. The kurta is finished with elegant 3/4 sleeves, offering modest coverage. This set requires minimal accessorizing and is ideal for weddings, festivals, or cultural events, ensuring you look and feel comfortable while maintaining an ethereal, sophisticated presence.

Subcategory: {subcategory}
Details: {product_details}
"""


# --- TEXT GENERATION LOGIC ---

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

# --- TEXT GENERATION ENDPOINT (FIXED ERROR PROPAGATION) ---
@app.route('/api/generate-title-description', methods=['POST'])
def generate_title_description():
    data = request.get_json()
    subcategory = basic_clean(data.get('subcategory', 'T-Shirt'))
    
    # Robustly construct product details
    product_details_lines = []
    for k, v in data.items():
        if k not in ('subcategory', 'type') and v and str(v).strip():
            product_details_lines.append(f"{k}: {basic_clean(v)}")


    product_details = "\n".join(product_details_lines)
    task_type = data.get('type', 'title')
    
    # Require subcategory and at least two fields
    if not subcategory or len(product_details_lines) < 2:
        return jsonify({'success': False, 'error': 'Missing required detail fields for good AI output.'}), 400

    if task_type == 'title':
        generated_title = generate_product_name(subcategory, product_details)
        # Check for failure message and return 500
        if generated_title.startswith("Generation Failed:"):
            return jsonify({'success': False, 'error': generated_title}), 500
        return jsonify({'success': True, 'generated_title': generated_title})
        
    elif task_type == 'description':
        generated_description = generate_product_description(subcategory, product_details)
        # Check for failure message and return 500
        if generated_description.startswith("Generation Failed:"):
            return jsonify({'success': False, 'error': generated_description}), 500
        return jsonify({'success': True, 'generated_description': generated_description})
        
    else:
        return jsonify({'success': False, 'error': 'Unknown type specified'}), 400

if __name__ == "__main__":
    # Ensure correct environment variables are set for both Flask and GCS/Gemini
    app.run(host="0.0.0.0", port=5000, debug=True)