
import gradio as gr
import tempfile
import speech_recognition as sr
from gtts import gTTS
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

# Load emotion detection model
emotion_classifier = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    return_all_scores=False
)

# Load Flan-T5 model for dynamic response
model_name = "google/flan-t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def generate_dynamic_response(user_input, emotion_label):
    prompt = (
        f"You are a mindful, empathetic AI coach. A user expressed the emotion '{emotion_label}' "
        f"and said: '{user_input}'. Respond in a calm, supportive way, avoiding tech jargon."
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    outputs = model.generate(**inputs, max_length=100)
    reply = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return reply

def gradio_chat_with_voice(audio_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio_data = recognizer.record(source)
        try:
            user_input = recognizer.recognize_google(audio_data)
        except sr.UnknownValueError:
            return "Sorry, I couldn't understand you.", None

    result = emotion_classifier(user_input)[0]
    emotion = result['label']
    reply = generate_dynamic_response(user_input, emotion)

    tts = gTTS(reply)
    tts.save("response.mp3")
    return f"[{emotion}] {reply}", "response.mp3"

iface = gr.Interface(
    fn=gradio_chat_with_voice,
    inputs=gr.Audio(type="filepath", label="🎙️ Speak your thought"),
    outputs=[
        gr.Textbox(label="🧘 AI Detox Coach Response"),
        gr.Audio(label="🎧 Listen to Response", type="filepath")
    ],
    title="AI Detox Coach with Voice",
    description="Talk to your coach. Get emotional support via voice + text."
)

iface.launch()
