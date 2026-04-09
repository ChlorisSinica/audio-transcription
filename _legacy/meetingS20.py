import whisper
import spacy
import glob

# spaCy英語モデルをロード（事前に python -m spacy download en_core_web_sm）
nlp = spacy.load("en_core_web_sm")


def format_mm_ss_mmm(seconds):
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes:02}:{secs:06.3f}"


def make_translation(path_m4a: str, path_translation: str):
    model = whisper.load_model("large-v3", device="cuda")
    result = model.transcribe(path_m4a, verbose=False, fp16=False, language="en", word_timestamps=True)

    # Whisperのword-level情報を取得
    words = []
    for seg in result["segments"]:
        for w in seg["words"]:
            words.append({
                "start": w["start"],
                "end": w["end"],
                "text": w["word"]
            })

    # 全テキストを結合してNLPで文分割
    full_text = "".join([w["text"] for w in words])
    doc = nlp(full_text)

    # 文ごとにタイムスタンプを計算
    merged_sentences = []
    word_index = 0
    for sent in doc.sents:
        sent_text = sent.text.strip()
        sent_words = sent_text.split()

        # 文の最初と最後の単語のタイムスタンプを取得
        start_time = words[word_index]["start"]
        end_time = words[word_index + len(sent_words) - 1]["end"]

        merged_sentences.append({
            "start": start_time,
            "end": end_time,
            "text": sent_text
        })

        word_index += len(sent_words)

    # 保存
    with open(path_translation, "w", encoding="utf-8") as f:
        for s in merged_sentences:
            f.write(f"[{format_mm_ss_mmm(s['start'])} --> {format_mm_ss_mmm(s['end'])}] {s['text']}\n")

    print(f"保存完了: {path_translation}")


def op_make_translation(dir_translation, dir_m4a, text_lecture_range):
    for i in text_lecture_range:
        text_m4a = dir_m4a + "Lecture {:02}".format(i) + " *.m4a"
        files = glob.glob(text_m4a)
        for j, file in enumerate(files):
            print(file)
            path_m4a = file
            path_translation = dir_translation + "Lecture {:02} - Module {} eng.txt".format(i, j + 1)
            make_translation(path_m4a, path_translation)


def main():
    # path_m4a = "C:/Users/CVSLab/Music/No Voice/Thom Cochell/MSE403 S20/Lecture 37 Module 1_normalized.m4a"
    # path_translation = "C:/myApp/AI whisper/MSE403 S20/Lecture 37 Module 1 eng.txt"
    # make_translation(path_m4a, path_translation)

    # dir_translation = ".C:/myApp/AI whisper/MSE403 S21/"
    # dir_m4a = "C:/Users/CVSLab/Music/No Voice/Thom Cochell/MSE403 S21/"
    # text_lecture_range = list(range(1, 23))
    # op_make_translation(dir_translation, dir_m4a, text_lecture_range)

    # dir_m4a = "C:/Users/CVSLab/Music/No Voice/Thom Cochell/MSE403 S20/"
    dir_m4a = "./0101/"
    dir_translation = "./0101/"
    # text_lecture_range = list(range(24, 40))
    op_make_translation(dir_translation, dir_m4a, text_lecture_range)


if __name__ == "__main__":
    main()
