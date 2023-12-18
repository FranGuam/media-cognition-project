import speech_recognition as sr
# 创建一个Recognizer实例
r = sr.Recognizer()

# harvard = sr.AudioFile('harvard.wav')
# with harvard as source:
#     audio = r.record(source)
# r.recognize_google(audio)

# 使用麦克风作为音频来源
with sr.Microphone() as source:
    print("请说话：")
    # 监听麦克风的音频输入
    audio = r.listen(source)

try:
    # 使用Google Speech Recognition API进行识别
    # 通过设置language参数为'zh-CN'来进行中文识别
    print("你说的是: " + r.recognize_google(audio, language='zh-CN'))
except sr.UnknownValueError:
    print("Google Speech Recognition无法理解你的语音")
except sr.RequestError as e:
    print("无法从Google Speech Recognition获取结果; {0}".format(e))
