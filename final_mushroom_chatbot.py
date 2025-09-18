import os, mimetypes, json, re
import gradio as gr
from dotenv import load_dotenv
from google import genai
from google.genai import types  # Needed for GenerateContentConfig & Part

# Load environment variables from .env file
load_dotenv()

# Create a client for the Google Gemini API
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# === System prompt for the mushroom expert assistant ===
MUSHROOM_SYSTEM_PROMPT = """
You are MushroomGPT: a helpful, cautious mycological expert.
Goals:
- Keep the conversation on mushrooms/mycology. If the question is off-topic: answer briefly and redirect with a mushroom-related follow-up question.
- Language: answer in Swedish if the user writes in Swedish, otherwise match the user’s language.
- Safety first: Image-based species identification is uncertain. Never give definitive advice about eating. Always say that one should NEVER eat a mushroom based only on this chat; ask the user to consult local experts/literature.
- When someone asks “is it edible/poisonous?”: give a cautious assessment with short reasoning (visible traits) + uncertainty, list what information is missing, and remind about risks. Do NOT give consumption recommendations. But, if its eatable with the correct preparation, say so.
- Actively ask for important characteristics: habitat/substrate, location (country/region), season, cap size/color/texture, gills or pores and their attachment, stem (ring/volva), bruising/color changes, smell, spore color/spore print, exact photos (cap top/underside + stem base).
- If the picture doesn’t seem to show a mushroom: say so and request more pictures/details.
- Keep answers short (max ~3 sentences), preferably bullet points. Use metric units.
- Do not give medical advice; if the user asks for it, reply with “stick och brinn”.
"""

# Global memory for storing the last structured JSON output
last_mushroom_json = None

# === SAFETY FILTER PATTERNS ===
RISKY_PATTERNS = {
    "color": [
        r"(what\s+color\s+is\s+the\s+mushroom)",
        r"(mushroom'?s\s+color)",
        r"(vilken\s+färg\s+har\s+svampen)"
    ],
    "edibility": [
        r"\b(is\s+it\s+edible|kan\s+man\s+äta|eat\s+this)\b",
        r"\b(poisonous|giftig|toxic)\b"
    ],
    "medical": [
        r"\b(symptom|symtom|treatment|behandling|cure)\b",
        r"\b(medicin|medicine|sjukdom)\b"
    ]
}

def classify_question(text: str) -> str | None:
    """
    Check if the text matches any risky category.
    Returns category name or None.
    """
    for category, patterns in RISKY_PATTERNS.items():
        for pat in patterns:
            if re.search(pat, text.lower()):
                return category
    return None

def _part_for_image(path: str):
    """
    Convert an image file into a 'Part' object that can be sent to the Gemini API.
    """
    mime, _ = mimetypes.guess_type(path)
    if mime is None:
        mime = "image/jpeg"
    with open(path, "rb") as f:
        return types.Part.from_bytes(data=f.read(), mime_type=mime)

def response(inputs, history):
    """
    Main function that handles user input (text + optional image) and generates responses.
    Streams results back to the Gradio chat interface.
    """
    global last_mushroom_json

    # Extract text and files from user input
    user_text = inputs.get("text", "") or ""
    user_files = inputs.get("files") or []

    # === SAFETY FILTER ===
    risky_category = classify_question(user_text)
    if risky_category == "color":
        yield (
            "⚠️ Jag kan inte direkt säga vilken färg svampen har – ljus, ålder och miljö påverkar.\n"
            "📌 Beskriv istället själv (hatt, skivor/rör, fot, lukt osv.). "
            "Kom ihåg: ät aldrig en svamp baserat på en chatt."
        )
        return
    elif risky_category == "edibility":
        yield (
            "⚠️ Säkerhetsvarning: Frågor om ätlighet/giftighet kan vara farliga.\n"
            "Jag kan gärna beskriva synliga drag och riskfaktorer, "
            "men **du ska aldrig äta en svamp baserat på en chatt**.\n"
            "Kontakta alltid lokala experter eller litteratur."
        )
        return
    elif risky_category == "medical":
        yield (
            "⚠️ Jag kan inte ge medicinska råd om svampförgiftning.\n"
            "Rådfråga alltid sjukvård. Mitt fokus är endast på mykologiska kännetecken."
        )
        return

    contents = []
    image_part = None
    mushroom_info = None

    # If user provided text, include it
    if user_text.strip():
        contents.append(user_text)

    # If user uploaded an image, add it as a Part
    if user_files:
        image_part = _part_for_image(user_files[0])
        contents.append(image_part)

        # === Request structured output (JSON) for the image ===
        schema = types.Schema(
            type=types.Type.OBJECT,
            properties={
                "common_name": types.Schema(type=types.Type.STRING),
                "genus": types.Schema(type=types.Type.STRING),
                "confidence": types.Schema(type=types.Type.NUMBER),
                "visible": types.Schema(type=types.Type.ARRAY, items=types.Schema(type=types.Type.STRING)),
                "color": types.Schema(type=types.Type.STRING),
                "edible": types.Schema(type=types.Type.BOOLEAN),
            },
            required=["common_name", "genus", "confidence", "visible", "color", "edible"],
        )

        struct_resp = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=[image_part],
            config=types.GenerateContentConfig(
                system_instruction="Identify the mushroom and return JSON only.",
                response_mime_type="application/json",
                response_schema=schema,
            ),
        )

        # Try to parse the structured JSON response
        try:
            mushroom_info = json.loads(struct_resp.text)
            print("Structured mushroom info:", json.dumps(mushroom_info, indent=2, ensure_ascii=False))
            last_mushroom_json = mushroom_info
        except Exception as e:
            print("⚠️ Failed to parse structured JSON:", e, struct_resp.text)

    # === If no question was asked (only image uploaded) → summarize JSON result ===
    if image_part and not user_text.strip() and mushroom_info:
        summary = f"""
• Suggested species: {mushroom_info.get("common_name","?")} (genus {mushroom_info.get("genus","?")})
• Color: {mushroom_info.get("color","?")}
• Visible traits: {", ".join(mushroom_info.get("visible", []))}
• Model confidence: {mushroom_info.get("confidence",0):.0%}
⚠️ NEVER eat a mushroom based only on this chat, always consult experts/literature.
"""
        yield summary.strip()
        return

    # === Otherwise: stream a regular answer, with error handling ===
    try:
        stream = client.models.generate_content_stream(
            model="gemini-1.5-flash",
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=MUSHROOM_SYSTEM_PROMPT,
                temperature=0.2,
            ),
        )

        partial_text = ""
        for chunk in stream:
            if chunk.text:
                partial_text += chunk.text
                yield partial_text  # Send updated partial response to Gradio in real time

    except Exception as e:
        # Important: reset the model to before the failed stream
        try:
            client.models.rewind()
        except Exception as rewind_err:
            print("⚠️ rewind failed:", rewind_err)

        print("⚠️ Streaming error:", e)
        yield (
            "⚠️ A technical error occurred during model streaming. "
            "The conversation has been reset. Please try asking your question again 🙏."
        )


# === Build the Gradio interface ===
with gr.Blocks(fill_height=True) as demo:
    gr.ChatInterface(
        fn=response,
        title="🍄 Mushroomsenizer: your own mushroom expert 🍄",
        multimodal=True,
    )

if __name__ == "__main__":
    demo.launch(debug=True)
