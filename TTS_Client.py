from gradio_client import Client, handle_file

client = Client("thinhlpg/vixtts-demo")

result = client.predict(
    prompt="Xin chào, tôi là Long béo",
    language="vi",
    audio_file_pth=handle_file("https://thinhlpg-vixtts-demo.hf.space/file=/tmp/gradio/01b4edbba4aec9b7bba6fb7e7d5170287b4739f4/nu-luu-loat.wav"),
    normalize_text=True,
    api_name="/predict"
)

print(result)

