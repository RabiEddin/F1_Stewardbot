import cv2
import base64
import openai
import os
from openai import OpenAI
from PIL import Image
import io
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def extract_frames_from_video(video_path, fps=2, max_frames=20):
    cap = cv2.VideoCapture(video_path)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)  # 영상 프레임 추출
    interval = int(frame_rate / fps)  # 몇 프레임 단위로 캡처할지
    frames = []
    frame_count = 0

    while len(frames) < max_frames and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % interval == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            img_b64 = base64.b64encode(buffered.getvalue()).decode()
            frames.append(img_b64)
        frame_count += 1

    cap.release()
    return frames


def describe_frame_b64(frames, prompt_file_path):
    with open(prompt_file_path, 'r', encoding='utf-8') as pf:
        prompt_text = pf.read().strip()

    vision_input = [
                       {"type": "text",
                        "text": prompt_text + "\ndate:" + input("date: ") + "Race country: " + input("Race country: ")}
                   ] + [
                       {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}}
                       for img_b64 in frames
                   ]

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": vision_input}]
    )
    return response.choices[0].message.content


def build_situation_from_video(video_path):
    frames = extract_frames_from_video(video_path)
    descriptions = describe_frame_b64(frames, "txt files/situation_description.txt")
    return descriptions


def get_situation_from_video(video_path):
    return build_situation_from_video(video_path)


if __name__ == "__main__":
    situation = get_situation_from_video("race video/VER_penalty 3.mp4")
    print(situation)
