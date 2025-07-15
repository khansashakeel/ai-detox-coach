import gradio as gr
import tempfile
import speech_recognition as sr
from gtts import gTTS
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import os

# Load emotion detection model
emotion_classifier = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    return_all_scores=False
)

# Load Flan-T5 model
model_name = "google/flan-t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Generate a mindful, dynamic response
def generate_dynamic_response(user_input, emotion_label):
    prompt = (
        f"You are a mindful, empathetic AI coach. A user expressed the emotion '{emotion_label}' "
        f"and said: '{user_input}'. Respond in a calm, supportive way, avoiding tech jargon."
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    outputs = model.generate(**inputs, max_length=100)
    reply = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return reply

# Handle both text and audio input
def ai_detox_chat(text_input, audio_input):
    user_input = ""

    if audio_input is not None and os.path.exists(audio_input):
        try:
            recognizer = sr.Recognizer()
            with sr.AudioFile(audio_input) as source:
                audio_data = recognizer.record(source)
                user_input = recognizer.recognize_google(audio_data)
        except Exception:
            return "Sorry, I couldn't understand your voice.", None

    elif text_input and text_input.strip():
        user_input = text_input.strip()
    else:
        return "Please provide either voice or text input.", None

    result = emotion_classifier(user_input)[0]
    emotion = result['label']
    reply = generate_dynamic_response(user_input, emotion)

    # Generate TTS
    tts = gTTS(reply)
    tts.save("response.mp3")

    return f"[{emotion}] {reply}", "response.mp3"

# Gradio interface
iface = gr.Interface(
    fn=ai_detox_chat,
    inputs=[
        gr.Textbox(label="💬 Type your thought (optional)", lines=2, placeholder="Feeling overwhelmed..."),
        gr.Audio(type="filepath", label="🎙️ Or speak your thought (optional)")
    ],
    outputs=[
        gr.Textbox(label="🧘 AI Detox Coach Response"),
        gr.Audio(label="🎧 Listen to Response", type="filepath")
    ],
    title="AI Detox Coach",
    description="Speak or type to your coach. Let it detect your emotion and guide you to a mindful tech break. 🌱"
)

iface.launch()
