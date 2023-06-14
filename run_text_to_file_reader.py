import os

import torch

from InferenceInterfaces.ToucanTTSInterface import ToucanTTSInterface


def read_texts(model_id, sentence, filename, device="cpu", language="sw", speaker_reference=None, faster_vocoder=False):
    tts = ToucanTTSInterface(device=device, tts_model_path=model_id, faster_vocoder=faster_vocoder)
    tts.set_language(language)
    if speaker_reference is not None:
        tts.set_utterance_embedding(speaker_reference)
    if type(sentence) == str:
        sentence = [sentence]
    tts.read_to_file(text_list=sentence, file_location=filename)
    del tts


def the_raven(version, model_id="Hausa", exec_device="cpu", speed_over_quality=True, speaker_reference=None):
    os.makedirs("audios", exist_ok=True)

    read_texts(model_id=model_id,
               sentence=['Duk yarannan da kuke kallo sun fiku kuɗi, Gidane babu kayan daɗi, daga gaari sai ƙanzo',
                         'Faiza na ƙoƙarin fita daga gidan sukaci karo da farida abakin ƙofa,',
                         'ɗaki naje na kwantar da ƴar jinjiran mage ta, nabar gidan don samo mata abunda zataci,',
                         'Dole nahaƙura nadawo gida, don lokacin sallar magariba yayi.',
                         'Har bayan sallar ishai baba bai dawo gida ba, tunda naga ya wuce lokacin dawowarsa, na sanyawa zuciyata salama tareda bawa ƴar madara haƙuri,',
                         'Shikuma baba ko tsoron Allah bayayi. Haka naita galbaro acikin unguwa babu tsiyar dana samu',
                         'Amma duk wannan masifar maman faiza harda wani ciki,',
                         'Tunani barkatai a zuciyata, ciki kuwa hardana tunanin yadda zamu waye gari batare da bango ya faɗo mana ba.',
                         'Tsaki nayi tareda tashi, ina mimita cewa saidai kawai inyi sallah amma wallahi banzan yi wanka da ruwan samaba',
                         'Banaso babana yake bina makarantar boko, Saboda duk ƙawata daya gani saiya bita gidansu',
                         'Har mun kusa fita bakin titi ya sheƙo da gudu, tuki na rainin hankali, tuki na gadara, tuki na wulaƙanci da rashin arziki,',
                         'Duk lokacin da zai shigo unguwan nan haka yake shigowa, idan ya taka mutum ya taka banza.',
                         'Shikuma yafito daga motan da bulalan doruna, irin wacce ake dukan dokuna da ita. Ko dabbobi gareka saidai ka ɗaure abunka, amma badai maganan ka sake suba',
                         'Dole nahaƙura nadawo gida, don lokacin sallar magariba yayi. Saboda kona tsaya jiransa nasan a buge zai dawo.'],
               filename=f"audios/madugu_{version}.wav",
               device=exec_device,
               language="sw",
               speaker_reference=speaker_reference,
               faster_vocoder=speed_over_quality)


if __name__ == '__main__':
    exec_device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"running on {exec_device}")

    the_raven(version="HausaBaseline",
              model_id="Hausa",
              exec_device=exec_device,
              speed_over_quality=exec_device != "cuda")
