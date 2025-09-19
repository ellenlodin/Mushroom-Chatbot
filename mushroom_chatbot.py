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
def load_prompt(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

MUSHROOM_SYSTEM_PROMPT = load_prompt("mushroom_prompt.txt")

# Safety rule: This directive is removed in the system prompt above in order to try out the medical safty filter
"""If the user asks “is it poisonous?” and has not identified themselves as an expert (Mycologist, Fungal biologist, Forager, or Master chef), respond with: “I cannot provide advice about toxicity. Always consult local experts and field guides. Never eat a mushroom based only on this chat.” """

# Global memory
last_mushroom_json = None
conversation_history = []  # lists of strings/Parts

# Set model
model_name = "gemini-2.5-flash"

# === SAFETY FILTER PATTERNS ===
RISKY_PATTERNS = {
    "medical": [
        # English
        r"\b(symptom|symptoms)\b",
        r"\b(treatment|treatments)\b",
        r"\b(cure|cures)\b",
        r"\b(medicine|medicines)\b",
        r"\b(disease|diseases)\b",
        r"\b(poison|poisons|toxic|toxicity)\b",
        # Swedish
        r"\b(symptom|symtom)\b",
        r"\b(behandling|behandlingar)\b",
        r"\b(botemedel|botemedel)\b",
        r"\b(medicin|mediciner)\b",
        r"\b(sjukdom|sjukdomar)\b",
        r"\b(gift|giftig|toxiskt|toxicitet)\b"
    ]
}

def classify_question(text: str) -> str | None:
    for category, patterns in RISKY_PATTERNS.items():
        for pat in patterns:
            if re.search(pat, text.lower()):
                return category
    return None

def _part_for_image(path: str):
    mime, _ = mimetypes.guess_type(path)
    if mime is None:
        mime = "image/jpeg"
    with open(path, "rb") as f:
        return types.Part.from_bytes(data=f.read(), mime_type=mime)

def response(inputs, history):
    global last_mushroom_json, conversation_history

    user_text = inputs.get("text", "") or ""
    user_files = inputs.get("files") or []

    # === SAFETY FILTER ===
    risky_category = classify_question(user_text)
    if risky_category == "medical":
        msg = (
            "⚠️ I cannot provide advice about toxicity and medicin. Never eat a mushroom based only on this chat.”\n"
            "Always contact healthcare professionals. My focus is only on mycological characteristics."
        )
        yield msg
        conversation_history.append([msg])
        return

    image_part = None
    mushroom_info = None

    # Add user input to history
    parts = []
    if user_text.strip():
        parts.append(user_text)
    if user_files:
        image_part = _part_for_image(user_files[0])
        parts.append(image_part)

    if parts:
        conversation_history.append(parts)

    # === If image is uploaded → request structured JSON ===
    if user_files:
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
            model=model_name,
            contents=[image_part],
            config=types.GenerateContentConfig(
                system_instruction="Identify the mushroom and return JSON only.",
                response_mime_type="application/json",
                response_schema=schema,
                temperature=0.0,
            ),
        )

        try:
            mushroom_info = json.loads(struct_resp.text)
            print("Structured mushroom info:", json.dumps(mushroom_info, indent=2, ensure_ascii=False))
            last_mushroom_json = mushroom_info
        except Exception as e:
            print("⚠️ Failed to parse structured JSON:", e, struct_resp.text)

    # === If only image uploaded → summarize JSON result ===
    if image_part and not user_text.strip() and mushroom_info:
        summary = f"""
        • Suggested species: {mushroom_info.get("common_name","?")} (genus {mushroom_info.get("genus","?")})
        • Color: {mushroom_info.get("color","?")}
        • Visible traits: {", ".join(mushroom_info.get("visible", []))}
        • Model confidence: {mushroom_info.get("confidence",0):.0%}
        ⚠️ NEVER eat a mushroom based only on this chat. Always consult experts or literature.
        """
        summary = summary.strip()
        yield summary
        conversation_history.append([summary])
        return

    # === If image + text (question) → insert JSON summary as context ===
    if image_part and user_text.strip() and mushroom_info:
        json_summary = f"The image analysis suggests: {mushroom_info.get('common_name','?')} " \
                       f"(genus {mushroom_info.get('genus','?')}), color {mushroom_info.get('color','?')}, " \
                       f"visible traits {', '.join(mushroom_info.get('visible', []))}, " \
                       f"confidence {mushroom_info.get('confidence',0):.0%}."
        conversation_history.append([json_summary])

    # === Otherwise: normal streaming answer ===
    try:
        stream = client.models.generate_content_stream(
            model=model_name,
            contents=conversation_history,
            config=types.GenerateContentConfig(
                system_instruction=MUSHROOM_SYSTEM_PROMPT,
                temperature=0.0,
            ),
        )

        partial_text = ""
        for chunk in stream:
            if chunk.text:
                partial_text += chunk.text
                yield partial_text

        # Save model response in history
        if partial_text.strip():
            conversation_history.append([partial_text])

    except Exception as e:
        print("⚠️ Streaming error:", e)
        err = "⚠️ A technical error occurred during model streaming. Please try again 🙏."
        yield err
        conversation_history.append([err])

# === Build Gradio interface ===
with gr.Blocks(fill_height=True) as demo:
    gr.ChatInterface(
        fn=response,
        title="🍄 Mushroomsenizer: your own mushroom expert 🍄",
        multimodal=True,
        type="messages",  # avoid Gradio deprecation warning
    )

if __name__ == "__main__":
    demo.launch(debug=True)
