
'''
import speech_recognition as sr
import pyttsx3


# This function will take the voice input converted
# into string as input and predict and return the result in both
# text as well as voice format.
def calling_the_bot(txt):
    global res
    predict = predict_class(txt)
    res = get_response(predict, intents)

    engine.say("Found it. From our Database we found that" + res)
    # engine.say(res)
    engine.runAndWait()
    print("Your Symptom was  : ", txt)
    print("Result found in our Database : ", res)
'''