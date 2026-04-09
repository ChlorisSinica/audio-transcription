import whisper
import json

filename = "36-4"
model = whisper.load_model("large-v3", device="cuda") #モデル指定
result = model.transcribe(filename + ".m4a", verbose=True, fp16=False, language="en")
print(result['text'])

f = open(filename + '.txt', 'w', encoding='UTF-8')
f.write(json.dumps(result['text'], sort_keys=True, indent=4, ensure_ascii=False))
f.close()