import json
import os
from json import JSONEncoder

import face_recognition
import numpy
from PIL import Image, ImageDraw, ImageFont


class NumpyArrayEncoder(JSONEncoder):
    def default(self, o):
        if isinstance(o, numpy.ndarray):
            return o.tolist()
        return JSONEncoder.default(self, o)


files = [file for file in os.listdir(os.getcwd()) if file.endswith(('jpeg', 'jpg', 'png'))]
for file in files:
    print(str(files.index(file) + 1) + '.', file)

choice = int(input('Select file (1, 2, ...): '))
unknown = face_recognition.load_image_file(files[choice - 1])

top, right, bottom, left = face_recognition.face_locations(unknown)[0]
image = Image.fromarray(unknown)
draw = ImageDraw.Draw(image)
unknown_enc = face_recognition.face_encodings(unknown)[0]

faces = open('faces.json', 'r')
encodings = dict(json.load(faces))

flag: bool = False

for face in encodings:
    face_data = encodings[face]
    if face_recognition.compare_faces([face_data], unknown_enc)[0]:
        print(face.title())
        draw.rectangle(((left, top), (right, bottom)), outline=(0, 255, 0))

        font = ImageFont.truetype('Roboto-Black.ttf', 18)
        text_width, text_height = draw.textsize(face)
        draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 0))

        draw.text((left + 6, bottom - text_height - 5), face, fill=(255, 255, 255), font=font)
        image.show()
        flag = True
        break

if not flag:
    print('Not found in database!')
    name = input('Please add a new name: ')
    encodings.update(
        {
            name: unknown_enc
        }
    )
    json.dump(encodings, open('faces.json', 'w'), cls=NumpyArrayEncoder, indent=2)
