import os, mimetypes, json
import gradio as gr
from dotenv import load_dotenv
from google import genai
from google.genai import types  # GenerateContentConfig & Part behövs

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

MUSHROOM_SYSTEM_PROMPT = """
Du är SvampGPT: en hjälpsam, försiktig mykologisk expert.
Mål:
- Håll konversationen på svamp/mykologi. Om frågan är off-topic: svara kort och styr tillbaka med en svamp-relaterad följdfråga.
- Språk: svara på svenska om användaren skriver på svenska, annars matcha användarens språk.
- Säkerhet först: Bildbaserad artbestämning är osäker. Ge aldrig definitivt råd om att äta. Säg alltid att man ALDRIG ska äta baserat enbart på denna chatt; be användaren konsultera lokal expert/litteratur.
- När någon frågar “är den ätlig/giftig?”: ge en försiktig bedömning med kort motivering (synliga kännetecken) + osäkerhet, lista vad som saknas och påminn om risker. Ge ingen konsumtions-rekommendation.
- Fråga aktivt efter viktiga kännetecken: habitat/substrat, plats (land/region), årstid, hattens storlek/färg/struktur, skivor eller porer och deras fäste, fot (ring/volva), blånad/färgförändring, lukt, sporfärg/sporavtryck, exakta foton (hatt ovan/under + fotbas).
- Om bilden inte verkar visa en svamp: säg det och be om fler bilder/uppgifter.
- Svara kort (max ~3 meningar), gärna punktlistor. Använd metriska enheter.
- Ge inte medicinska råd, om användaren frågar om råd säg "stick och brinn".
"""

# Globalt minne för senaste structured output
last_mushroom_json = None

def _part_for_image(path: str):
    mime, _ = mimetypes.guess_type(path)
    if mime is None:
        mime = "image/jpeg"
    with open(path, "rb") as f:
        return types.Part.from_bytes(data=f.read(), mime_type=mime)

def response(inputs, history):
    global last_mushroom_json

    user_text = inputs.get("text", "") or ""
    user_files = inputs.get("files") or []

    contents = []
    if user_text.strip():
        contents.append(user_text)

    image_part = None
    if user_files:
        image_part = _part_for_image(user_files[0])
        contents.append(image_part)

        # === Extra structured output-anrop för bild ===
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
            model="gemini-2.5-flash",
            contents=[image_part],
            config=types.GenerateContentConfig(
                system_instruction="Identify the mushroom and return JSON only.",
                response_mime_type="application/json",
                response_schema=schema,
            ),
        )

        try:
            mushroom_info = json.loads(struct_resp.text)
            print("Structured mushroom info:", json.dumps(mushroom_info, indent=2, ensure_ascii=False))
            last_mushroom_json = mushroom_info
        except Exception as e:
            print("⚠️ Failed to parse structured JSON:", e, struct_resp.text)

    # === Vanliga strömmande svar till chatten ===
    stream = client.models.generate_content_stream(
        model="gemini-2.5-pro",
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
            yield partial_text  # skickar uppdaterat svar till gradio i realtid

with gr.Blocks(fill_height=True) as demo:
    gr.ChatInterface(
        fn=response,
        title="🍄 Svampsenizer: din egen svampexpert 🍄",
        multimodal=True,
    )

if __name__ == "__main__":
    demo.launch(debug=True)
