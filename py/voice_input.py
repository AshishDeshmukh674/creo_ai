import speech_recognition as sr

def main():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Speak now...")
        audio = recognizer.listen(source)
    try:
        text = recognizer.recognize_google(audio)
        print(text)
    except Exception as e:
        print(f"ERROR: {e}")

if __name__ == "__main__":
    main()