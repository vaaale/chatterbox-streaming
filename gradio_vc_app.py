import torch
import gradio as gr
from chatterbox.vc import ChatterboxVC


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# LOCAL_MODEL = None
LOCAL_MODEL = "/mnt/storage/Models/Chatterbox/checkpoints/chatterbox_finetuned_norwegian"
# LOCAL_MODEL = "/home/alex/PycharmProjects/chatterbox-streaming/checkpoints_lora/merged_model"
# LOCAL_MODEL = "/home/alex/PycharmProjects/chatterbox-streaming/checkpoints_grpo/merged_grpo_model"

def load_model():
    if LOCAL_MODEL is None:
        model = ChatterboxVC.from_pretrained(DEVICE)
    else:
        model = ChatterboxVC.from_local(ckpt_dir=LOCAL_MODEL, device=DEVICE)
    return model

model = load_model()
def generate(audio, target_voice_path):
    wav = model.generate(
        audio, target_voice_path=target_voice_path,
    )
    return model.sr, wav.squeeze(0).numpy()


demo = gr.Interface(
    generate,
    [
        gr.Audio(sources=["upload", "microphone"], type="filepath", label="Input audio file"),
        gr.Audio(sources=["upload", "microphone"], type="filepath", label="Target voice audio file (if none, the default voice is used)", value=None),
    ],
    outputs="audio",
)

if __name__ == "__main__":
    demo.launch()
