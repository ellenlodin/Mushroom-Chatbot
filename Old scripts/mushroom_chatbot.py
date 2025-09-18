import os, mimetypes
import gradio as gr
from dotenv import load_dotenv
from google import genai
from google.genai import types  # <— behövs för GenerateContentConfig & Part

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

def _part_for_image(path: str):
    mime, _ = mimetypes.guess_type(path)
    if mime is None:
        mime = "image/jpeg"
    with open(path, "rb") as f:
        return types.Part.from_bytes(data=f.read(), mime_type=mime)

def response(inputs, history):
    user_text = inputs.get("text", "") or ""
    user_files = inputs.get("files") or []

    contents = []
    if user_text.strip():
        contents.append(user_text)

    if user_files:
        # Ta första bilden (Gradio ChatInterface skickar sökvägen)
        contents.append(_part_for_image(user_files[0]))

    resp = client.models.generate_content(
        model="gemini-2.5-pro",  # eller "gemini-2.5-flash" för snabbare svar
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=MUSHROOM_SYSTEM_PROMPT,
            temperature=0.2,
        ),
    )

    return resp.text

with gr.Blocks(fill_height=True) as demo:
    gr.ChatInterface(
        fn=response,
        title="🍄 Svampsenizer: din egen svampexpert 🍄",
        multimodal=True,
    )

if __name__ == "__main__":
    demo.launch(debug=True)
