from typing import List, Dict
import torch.nn.functional as F
import os
import json
import torch
import logging
from collections import defaultdict
import numpy as np
import re
import math
import requests



rb_token = ''


headers = {
    "Authorization": f"Bearer {rb_token}",
    "Content-Type": "application/json"
}


url = "https://chat.readerbench.com/ollama/api/chat"


system_message = """
Ești un expert în literatură și în identificarea genurilor literare dintr-un text.
La fiecare apel, vei primi un text în limba română și o listă cu genurile literare în care
s-ar putea încadra acel pasaj.

Genurile literare posibile sunt următoarele:
1. sentimental: se concentrează pe relații romantice de obicei între un bărbat și o femeie, se vorbește mult despre iubire, emoții, sentimente, dragoste. Personajele vorbesc cu pasiune și își declară sentimentele. Intriga principală a unui roman de dragoste este despre doi oameni care se îndrăgostesc unul de celălalt și depun eforturi pentru a construi o relație. Este o poveste romantică care celebrează iubirea necondiționată și de obicei are un final fericit.
2. istoric: descrie evenimente și perioade istorice. Conține date și evenimente reale, persoane care au existat. Narațiunea este la timpul trecut și prezintă un eveniment social sau politic. Pot fi menționați regi, boieri, conducători sau alte persoane politice importante. Poate fi scris în stilul unei cronici istorice.
3. social: examinează structurile și problemele societății. Prezintă o problemă socială care afectează anumite personaje sau grupuri de personaje. Principalele exemple de probleme sociale abordate în astfel de lucrări sunt sărăcia, condițiile dificile din fabrici și mine, munca prestată de copii, violența împotriva femeilor, criminalitatea în creștere, condițiile mizerabile de trai din orașe. Descrie viața industrială, munca, schimbări sociale, revoluții industriale sau sociale, corupția.
4. haiducesc: se concentrează pe aventurile haiducilor. Haiducii sunt priviți ca niște eroi, sunt niște tâlhari nobili care luptă împotriva turcilor, fanarioților sau boierilor corupți și îi ajută pe cei săraci și pe oamenii de rând din popor. Haiducii contestă pe cei care au în acel moment autoritate politică.
5. criminal: implică investigatii legate de crime. Se desfășoară o anchetă condusă cu metode criminaliste specifice, de un polițist sau de un detectiv privat. Un personaj este omorât iar acțiunea se concentrează în jurul investigațiilor, sunt audiați potențiali suspecți iar criminalul este găsit.
6. mistere: implică rezolvarea unui mister, este plin de suspans, constă în investigarea și descoperirea unei soluții aparent imposibil de găsit. Pot apărea și descrieri despre îmbrăcăminte și alte detalii semnificative care pot ajuta la rezolvarea misterului.
7. război: sunt descrise conflicte militare. Într-un roman de război, acțiunea principală are loc pe câmpul de luptă sau în apropiere de linia frontului, cuprinzând povestea unor civili și militari a căror preocupare este pregătirea de război sau recuperarea în urma acestuia. Sunt descrise arme de război, strategii de luptă, soldați care se luptă și mor pe front, conflict între două popoare, două țări sau două grupări.
8. senzație: narațiunea este scrisă într-un mod cu totul nou și inedit, cu multe răsturnări de situație și lucruri surprinzătoare, cu multă acțiune, foarte diversă. Conține întorsături neprevăzute, intrigi, trădări, dueluri, crime, răpiri și bandiți. Combină personaje autohtone și străine, neverosimile. Poate avea acțiune fantezistă pornind de la un fapt real.
9. poetic: utilizează limbaj artistic pentru a exprima sentimente și idei. Prezintă stări sufletești, trăiri în natură și pentru natură, clocot pasional al iubirii. Conține lirism, metafore, contemplații, reflecții poetice asupra vieții.
10. psihologic: explorează mintea și comportamentele umane, pune accent pe analiza stărilor sufletești ale personajelor. Acordă importanță trăirilor interioare, transformarea personajului pe parcursul textului, conflicte interioare, relațiile dintre personaje se modifică, schimbarea interioară a individului, expunerea gândurilor personajelor, evocări patologice, conflicte interioare (la nivelul minții).
11. exil: "descrie experiențele trăite de cei în exil. Un personaj este condamnat să trăiască în exil, departe de țara sa, în care nu se poate întoarce.
12. science-fiction: "explorează concepte speculative, tehnologice și științifice. Descrie teme futuristice, acțiunea se poate petrece într-un viitor îndepărtat sau pe o altă planetă sau într-o altă realitate alternativă.
13. rural: descrie viața la țară, oameni care au ocupații în agricultură sau creșterea animalelor, ciobani, țărani, arendași, învățători la școli din sat, cârciumari. Descrie viața satului, traiul rustic, obiceiurile și sărbatorile dar și greutățile prin care trec țăranii.
14. religios: explorează teme si credințe din religia creștin ortodoxă. Vorbește despre Dumnezeu, despre începuturile creștinismului, teme biblice și întâmplări din Biblie repovestite. Dă o lecție de morală creștină.
15. filosofic: abordează întrebări filosofice și existențiale. Descrie procese analitice ale conștiinței, speculații filosofice, introspecții, ideologii. Abordează teme filosofico-existențiale ‒ libertate, responsabilitate, raportul cu divinitatea, minciună vs. adevăr, identitate autentică vs. mască, disimulare.
16. biografic: relatează viața unei persoane care a existat în realitate, conține memoriile unei persoane, consemnează date și evenimente din viața persoanei. oferă o relatare fictivă a vieții unei persoane contemporane sau istorice. Acest gen de roman se concentrează pe experiențele pe care le-a avut o persoană în timpul vieții sale, pe oamenii pe care i-a cunoscut și pe întâmplările care au avut loc.
17. aventură: prezintă povești de aventură, cu multă acțiune și întâmplări pe teritorii necunoscute, fapte pline de curaj, pericole iminente, provocări fizice.

Alege 3 genuri literare din lista de mai sus și pentru fiecare gen adaugă și probabilitatea tokenilor săi. Suma probabilităților trebuie să fie 1. Răspunde cu format json:
**ATENȚIE: Răspunde DOAR cu un obiect JSON, fără text suplimentar sau explicații. JSON-ul trebuie să respecte următorul format:**

---

Exemple de clasificare corectă:

Exemplul 1:

„În inima codrilor de nepătruns, haiducul Iancu și-a adunat oamenii pentru a se răzbuna pe boierul cel hain. Cu flinta în mână și cu ochii arzând de dorința dreptății, a pornit la drum, străbătând poteci ascunse și întâlnind sate prădate de zapcii.”  

Genuri și probabilități generate corect: 
```
{
  "genres_with_probabilities": [
    {"genre": "haiducesc", "probability": 0.6},
    {"genre": "aventură", "probability": 0.3},
    {"genre": "istoric", "probability": 0.1}
  ]
}
```

Exemplul 2:

„Într-un laborator secret, cercetătorii analizau un nou tip de material capabil să reziste la temperaturi extreme. Prototipul navei spațiale era aproape finalizat, iar în câteva luni urma să fie testat pe o orbită îndepărtată.” 

Genuri și probabilități generate corect: 
```
{
  "genres_with_probabilities": [
    {"genre": "science-fiction", "probability": 0.7},
    {"genre": "aventură", "probability": 0.2},
    {"genre": "istoric", "probability": 0.1}
  ]
}
```

Exemplul 3:

„Dumitru coborî de pe căruță și își scoase pălăria, ștergându-și fruntea cu mâneca hainei ponosite. Lanul de grâu se întindea până la marginea satului, unde câțiva copii desculți alergau pe uliță, râzând și strigând după oi.”

Genuri și probabilități generate corect:

```
{
  "genres_with_probabilities": [
    {"genre": "rural", "probability": 0.8},
    {"genre": "social", "probability": 0.15},
    {"genre": "poetic", "probability": 0.05}
  ]
}
```


"""




genres = ['sentimental',
 'istoric',
 'social',
 'haiducesc',
 'criminal',
 'mistere',
 'război',
 'senzație',
 'poetic',
 'psihologic',
 'exil',
 'science-fiction',
 'rural',
 'religios',
 'filosofic',
 'biografic',
 'aventură']


response_schema = {
    "type": "json_schema",
    "json_schema": {
        "name": "extracted_data",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "genres_with_probabilities": {
                    "type": "array",
                     "minItems": 3,  
                    "maxItems": 3, 
                    "items": {
                        "type": "object",
                        "properties": {
                            "genre": {
                                "type": "string",
                                "enum": genres 
                            },
                            "probability": {
                                "type": "number",
                                "minimum": 0,
                                "maximum": 1
                            }
                        },
                        "required": ["genre", "probability"]
                    }
                }
            },
            "required": ["genres_with_probabilities"],
            "additionalProperties": False
        }
    }
}


def validate_json_output(output: str) -> dict:
    """
    Validate that the output is a valid JSON and matches the expected schema.
    """
    try:
        # Extract JSON from the output using regex
        json_str = re.search(r'\{.*\}', output, re.DOTALL).group()
        json_data = json.loads(json_str)

        return json_data
    except (json.JSONDecodeError, ValidationError, AttributeError) as e:
        print(f"Invalid JSON or schema mismatch: {e}")
        return None



def get_valid_response(chat_template, max_retries=3):
    """
    Send the request and retry if the output is invalid.
    """
    for attempt in range(max_retries):
        response = requests.post(url, data=json.dumps(chat_template), headers=headers)
        print("Raw Response:", response.text)  # Log the raw response
        try:
            predicted_genre = response.json()['message']['content'].replace('\n', '').replace(' ', '')
            validated_output = validate_json_output(predicted_genre)
            if validated_output:
                return validated_output
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
    return None


def classify_genre(input_path: str, output_path: str):
    output_files = [n for n in os.listdir(output_path)]

    for name in os.listdir(input_path):
        file_path = os.path.join(input_path, name)
        if 'docx' not in file_path or name.replace('.json_classified', '.json_classified.json_llama3') in output_files:
            logging.error(f'skipping: {name}')
            continue

        with open(file_path, 'r', encoding='utf-8') as fp:
            logging.error(file_path)
            data = json.load(fp)
            new_data = []
            for chapter in data:
                chapter_number = chapter["chapter"]
                new_chapter = {"chapter": chapter_number, "chunks": []}
                for chunk in chapter["chunks"]:
                    chunk_text = chunk["chunk_text"]

                    user_input = f"""
                    
                    Răspunde cu formatul json specificat, fără alte explicații.
                    \n

                    Alege doar 3 genuri din cele 17 genuri literare date: {genres}. \n

                    Suma probabilităților celor 3 genuri literare trebuie să fie 1.

                    Acesta este pasajul literar: 
                    \n
                    {chunk_text} 
                        \n
                        """

                    chat_template = {
                        "model": "llama3.3:latest",
                        "stream": False,
                        "temperature": 0,
                        "top_p": 1,
                        "frequency_penalty": 0.0,
                        "presence_penalty": 0.0,
                        "messages": [
                            {"role": "system", "content": system_message},
                            {"role": "user", "content": user_input}
                        ],
                        "response_format": response_schema
                    }

                    # Get validated response
                    validated_output = get_valid_response(chat_template)
                    if not validated_output:
                        print("Failed to get valid JSON output after retries.")
                        continue

                    word_count = chunk["word_count"]
                    new_chunk = {"chunk_text": chunk_text,
                                 "genre_probability": validated_output,
                                 "word_count": word_count}
                    new_chapter["chunks"].append(new_chunk)
                new_data.append(new_chapter)

            # Save result to JSON
            with open(os.path.join(output_path, f"{name}_llama3.json"), "w", encoding="utf-8") as f_out:
                json.dump(new_data, f_out, ensure_ascii=False, indent=4)



input_directory = '/novels/'
output_directory = '/novels_with_genre_llama/'
classify_genre(input_directory, output_directory)
