import os
import streamlit as st
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from langchain_community.tools import DuckDuckGoSearchRun
import asyncio
import edge_tts
from moviepy.editor import ImageClip, AudioFileClip, concatenate_videoclips
from moviepy.video.fx.all import fadein, fadeout
from diffusers import StableDiffusionXLPipeline
import torch
from PIL import Image ,ImageDraw, ImageFont



# =============================
# explain
# =============================
load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

if not HF_TOKEN:
    st.error("❌ Hugging Face token not found!")
    st.stop()

client = InferenceClient(
    model="meta-llama/Llama-3.1-8B-Instruct",
    token=HF_TOKEN
)

web_search = DuckDuckGoSearchRun()

# =============================
# New
# =============================

async def edge_tts_generate(text, filename):
    voice = "en-US-MichelleNeural"
    communicate = edge_tts.Communicate(text, voice, rate="-10%")
    await communicate.save(filename)

def text_to_speech_edge(text, filename):
    try:
        if os.path.exists(filename):
            os.remove(filename)
        asyncio.run(edge_tts_generate(text, filename))
        return filename
    except Exception as e:
        st.error(f"Edge TTS Error: {e}")
        return None


def split_text(text):

    blocks = []
    lines = text.split("\n")
    current_block = ""
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.endswith("?"):
            if current_block:
                blocks.append(current_block.strip())
            current_block = line
        else:
            current_block += "\n" + line
    if current_block:
        blocks.append(current_block.strip())
    return blocks



pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
)

pipe.enable_model_cpu_offload()
pipe.enable_attention_slicing()

# 3️⃣ تعريف الدالة generate_image
def generate_image(prompt, idx, width=400, height=400):
    image = pipe(prompt, num_inference_steps=40,guidance_scale=8).images[0]
    path = f"slide_image_{idx}.png"
    image.save(path)
    return path




# =============================
# SLIDE GENERATION
# =============================
def text_to_slide(question, answer_text, idx, image_path=None,
                  width=1280, height=900,
                  font_path="/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                  font_size_title=37, font_size_text=25):

    img = Image.new("RGB", (width, height), (30, 30, 60))  # خلفية كحلي
    draw = ImageDraw.Draw(img)

    font_title = ImageFont.truetype(font_path, font_size_title)
    font_text = ImageFont.truetype(font_path, font_size_text)
    margin = 50
    y = margin

    clean_question = question.replace("*", "").strip()
    clean_answer = answer_text.replace("*", "").strip()

    # Title
    bbox = draw.textbbox((0, 0), clean_question, font=font_title)
    title_w = bbox[2] - bbox[0]
    draw.text(((width - title_w)/2, y), clean_question, fill="white", font=font_title)
    y += bbox[3] - bbox[1] + 50

    # Answer
    for line in clean_answer.split("\n"):
        words = line.split()
        current = ""
        for word in words:
            test = current + " " + word if current else word
            bbox = draw.textbbox((0,0), test, font=font_text)
            if bbox[2] - bbox[0] < width - 2*margin:
                current = test
            else:
                draw.text((margin, y), current, fill="white", font=font_text)
                y += bbox[3] - bbox[1] + 20
                current = word
        if current:
            draw.text((margin, y), current, fill="white", font=font_text)
            y += bbox[3] - bbox[1] + 50

    if image_path:
        slide_img = Image.open(image_path)
        slide_img.thumbnail((width-2*margin, height-y-margin))
        x = width - slide_img.width - margin 
        img.paste(slide_img, (x, y))


    path = f"slide_{idx}.png"
    img.save(path)
    return path

# =============================
# STREAMLIT UI
# =============================
st.title("🎓 Educational AI Video (Q&A Style)")

topic = st.text_input("Enter topic:", "LangChain")

if st.button("Generate Explanation + Video"):

    # -------- Generate Q&A --------
    with st.spinner("🔍 Generating educational Q&A..."):
        search_results = web_search.run(topic)

        prompt = f"""
Create an educational explanation about the topic below.

Format strictly as Question and points of answers pairs.
Each pair separated by a blank line/s.
Beginner friendly.
Clear and concise.

Topic: {topic}

References:
{search_results}
"""

        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a helpful educational assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=700,
            temperature=0.4
        )

        explanation = response.choices[0].message.content.strip()
        st.subheader("📝 Generated Q&A")
        st.write(explanation)

    # -------- Split into Q&A --------
    chunks = split_text(explanation)
    st.info(f"📚 {len(chunks)} Q&A segments")

    # -------- Slides + Images + Audio --------
    clips = []
    with st.spinner("🎬 Generating video..."):
        for idx, chunk in enumerate(chunks):
            st.write(f"▶ Segment {idx + 1}")

            lines = chunk.split("\n")
            question = lines[0]
            answer_text = "\n".join(lines[1:])

            image_prompt = f"Educational illustration, colorful, clear and simple, about: {question}. "
            image_path = generate_image(image_prompt, idx)

            slide_path = text_to_slide(question, answer_text, idx, image_path=image_path)

            narration = question + ". " + answer_text
            audio_path = f"audio_{idx}.mp3"
            text_to_speech_edge(narration, audio_path)

            audio = AudioFileClip(audio_path)
            clip = ImageClip(slide_path).set_duration(audio.duration)
            clip = clip.set_audio(audio)
            clip = fadein(clip, 0.5).fx(fadeout, 0.5)
            clips.append(clip)

    # -------- Final Video --------
    final_video = concatenate_videoclips(clips, method="compose")
    final_video.write_videofile("final_video.mp4", fps=24)

    st.success("✅ Educational video created")
    st.video("final_video.mp4")
