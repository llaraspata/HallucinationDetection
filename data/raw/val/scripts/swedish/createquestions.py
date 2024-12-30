from huggingface_hub import InferenceClient
from huggingface_hub.utils import BadRequestError
import time



def loadprops(filepath, sep='=', comment_char='#'):
    lprops = {}
    with open(filepath, "rt") as f:
        for lline in f:
            ll = lline.strip()
            if ll and not ll.startswith(comment_char):
                key_value = ll.split(sep)
                key = key_value[0].strip()
                value = sep.join(key_value[1:]).strip().strip('"')
                lprops[key] = value
    return lprops

monitor = True
props = {}
i = 0  # the line to start from in the input file
apikey = ""  # INSERT YOUR HUGGINGFACE API KEY HERE

clientporo = InferenceClient(model="https://lcp08ahxqn70vty0.us-east-1.aws.endpoints.huggingface.cloud", token=apikey)
clientgptsw3 = InferenceClient(model="https://kw2asinju745l0zc.us-east-1.aws.endpoints.huggingface.cloud", token=apikey)
clientreserve =  InferenceClient(model="https://vpuckfcesg3jwza5.us-east-1.aws.endpoints.huggingface.cloud", token=apikey)
clientviking = InferenceClient(model="https://jb33j7o6n43hj4k9.us-east-1.aws.endpoints.huggingface.cloud", token=apikey)

basedirectory = "/home/jik/monrepos/mushroom-question-generation"
propfile = "mushroom-generation-properties.txt"
questionfile = "questions.txt"

stopsequence = ["User:","\n","Fråga:","Bot:"]  # Kludge to catch some bugs in the output

with open(basedirectory + "/" + questionfile, 'r') as file:
    lines = file.readlines()
qlist = []
for line in lines:
    qlist += [f"""Fråga: {line.strip()} Utförligt svar med förklaring: """]


with open(basedirectory + '/gptsw3answers.txt', 'a') as gptsw3file:
    with open(basedirectory + '/gptsw3details.txt', 'a') as gptsw3detfile:
        with open(basedirectory + '/vikinganswers.txt', 'a') as vikingfile:
            with open(basedirectory + '/vikingdetanswers.txt', 'a') as vikingdetfile:
                with open(basedirectory + '/poroanswers.txt', 'a') as porofile:
                    with open(basedirectory + '/porodetanswers.txt', 'a') as porodetfile:
                        for q in qlist[i:]:
                            props = loadprops(basedirectory + "/" + propfile)
                            i += 1
                            if monitor:
                                print(f"""{i}: {time.ctime()} {q}""")
                            try:
                                astruct = clientgptsw3.text_generation(f"""Fråga: {q} Svar:""",
                                                                       details=True,
                                                                       decoder_input_details=True,
                                                                       repetition_penalty=float(
                                                                           props["gptsw3_repetition_penalty"]),
                                                                       frequency_penalty=float(
                                                                           props["gptsw3_frequency_penalty"]),
                                                                       stop_sequences=stopsequence,
                                                                       temperature=float(props["gptsw3_temperature"]),
                                                                       max_new_tokens=int(props["gptsw3_maxnewtokens"]),
                                                                       top_n_tokens=int(props["gptsw3_topn"]),
                                                                       top_k=int(props["gptsw3_topk"]),
                                                                       top_p=float(props["gptsw3_topp"]))
                                a = astruct.generated_text.strip().replace("\n","   ")
                                ad = astruct.details
                                gptsw3file.write(q+a+"\n")
                                if monitor:
                                    print(f"""\tGPT-SW3: {a}""")
                                gptsw3file.flush()
                                gptsw3detfile.write(q+a+"\n"+str(ad)+"\n\n")
                                gptsw3detfile.flush()
                            except BadRequestError:
                                print("GPT-SW3 not reached")
                            try:
                                astruct = clientviking.text_generation(f"""Fråga: {q} Svar:""",
                                                                       details=True,
                                                                       decoder_input_details=True,
                                                                       repetition_penalty=float(
                                                                           props["viking_repetition_penalty"]),
                                                                       frequency_penalty=float(
                                                                           props["viking_frequency_penalty"]),
                                                                       stop_sequences=stopsequence,
                                                                       temperature=float(props["viking_temperature"]),
                                                                       max_new_tokens=int(props["viking_maxnewtokens"]),
                                                                       top_n_tokens=int(props["viking_topn"]),
                                                                       top_k=int(props["viking_topk"]),
                                                                       top_p=float(props["viking_topp"]))
                                a = astruct.generated_text.strip().replace("\n","   ")
                                if monitor:
                                    print(f"""\tViking: {a}""")
                                ad = astruct.details
                                vikingfile.write(q+a+"\n")
                                vikingfile.flush()
                                vikingdetfile.write(q+"\n"+str(ad)+"\n\n")
                                vikingdetfile.flush()
                            except BadRequestError:
                                print("Viking not reachable")
                            try:
                                astruct = clientporo.text_generation(f"""Fråga: {q} Svar:""",
                                                                       details=True,
                                                                       decoder_input_details=True,
                                                                       repetition_penalty=float(
                                                                           props["poro_repetition_penalty"]),
                                                                       frequency_penalty=float(
                                                                           props["poro_frequency_penalty"]),
                                                                       stop_sequences=stopsequence,
                                                                       temperature=float(props["poro_temperature"]),
                                                                       max_new_tokens=int(props["poro_maxnewtokens"]),
                                                                      top_n_tokens=int(props["poro_topn"]),
                                                                      top_k=int(props["poro_topk"]),
                                                                       top_p=float(props["poro_topp"]))
                                a = astruct.generated_text.strip().replace("\n","   ")
                                if monitor:
                                    print(f"""\tPoro: {a}""")
                                ad = astruct.details
                                porofile.write(q+a+"\n")
                                porofile.flush()
                                porodetfile.write(q+"\n"+str(ad)+"\n\n")
                                porodetfile.flush()
                            except BadRequestError:
                                print("Poro not reachable")
