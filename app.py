import os
import torch
import random

from flask import (Flask, redirect, render_template, request,
                   send_from_directory, url_for)

from InferenceInterfaces.ToucanTTSInterface import ToucanTTSInterface

app = Flask(__name__)


@app.route('/')
def index():
   print('Request for index page received')
   return render_template('index.html')

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')

@app.route('/index', methods=['POST'])
def login():
   name = request.form.get('name')

   if name:
       print('Request for hello page received with name=%s' % name)
       redirect(url_for('inference'))
       #return render_template('inference.html', name = name)
   else:
       print('Request for hello page received with no name or blank name -- redirecting')
       return render_template('index.html', name = name)
    #return render_template('index.html', name = name)


def read_texts(model_id, sentence, filename, device="cpu", language="", speaker_reference=None, faster_vocoder=False):
    tts = ToucanTTSInterface(device=device, tts_model_path=model_id, faster_vocoder=faster_vocoder)
    tts.set_language(language)
    if speaker_reference is not None:
        tts.set_utterance_embedding(speaker_reference)
    if type(sentence) == str:
        sentence = [sentence]
    tts.read_to_file(text_list=sentence, file_location=filename)
    del tts


def the_raven(version, model_id, exsentence, exec_device="cpu", speed_over_quality=False, speaker_reference=None, langcode=""):
    os.makedirs("audios", exist_ok=True)

    if langcode != "" and langcode == "ha":
        twicklang = "sw"
        read_texts(model_id="Hausa",
               sentence=exsentence,
               filename=f"audios/madugu_{version}.wav",
               device=exec_device,
               language=twicklang,
               speaker_reference=speaker_reference,
               faster_vocoder=speed_over_quality)
    elif langcode != "" and langcode == "sw":
        read_texts(model_id="Swahili",
               sentence=exsentence,
               filename=f"audios/tai_{version}.wav",
               device=exec_device,
               language=langcode,
               speaker_reference=speaker_reference,
               faster_vocoder=speed_over_quality)



@app.route('/inference', methods=['POST'])
def inference():
    sentence = request.form.get('syn')
    langtag = request.form.get('tag')

    digit = random.randint(0, 9)
    exec_device = "cuda" if torch.cuda.is_available() else "cpu"
    #print(f"running on {exec_device}")

    if sentence != "" and langtag != "":
        if langtag != "" and langtag == "ha": 
            the_raven(version=digit,
                model_id="Hausa",
                exec_device=exec_device,
                speed_over_quality=exec_device != "cuda",
                exsentence=sentence,
                langcode=langtag)
        elif langtag != "" and langtag == "sw":
            the_raven(version=digit,
                model_id="Swahili",
                exec_device=exec_device,
                speed_over_quality=exec_device != "cuda",
                exsentence=sentence,
                langcode=langtag)

    return render_template('inference.html', langtag = langtag)

if __name__ == '__main__':
   app.run()
