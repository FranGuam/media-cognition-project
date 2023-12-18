import speech_recognition as sr

TIME = 5


def record(recognizer):
    mic = sr.Microphone()
    with mic as source:
        print("Adjusting for ambient noise...")
        recognizer.adjust_for_ambient_noise(source)
        print("Say something!")
        audio = recognizer.listen(source, timeout=TIME)
        print("Got it! Now to recognize it...")
    return audio


def callback_google(r, audio):
    try:
        return r.recognize_google(audio, language="zh-CN")
    except sr.UnknownValueError:
        print("Google Speech Recognition 无法理解你的语音")
    except sr.RequestError as e:
        print("无法从 Google Speech Recognition 获取结果; {0}".format(e))


def callback_sphinx(r, audio):
    try:
        return r.recognize_sphinx(audio, language="en-US")
    except sr.UnknownValueError:
        print("Oops! Didn't catch that")
    except sr.RequestError as e:
        print("Uh oh! Couldn't request results from Sphinx service; {0}".format(e))


def recognize():
    r = sr.Recognizer()
    audio = record(r)
    return callback_google(r, audio)


if __name__ == "__main__":
    r = sr.Recognizer()
    audio = record(r)
    print("Google:", callback_google(r, audio))
    print("Sphinx:", callback_sphinx(r, audio))
