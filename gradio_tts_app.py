import random
import numpy as np
import torch
import gradio as gr
from chatterbox.tts import ChatterboxTTS


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# LOCAL_MODEL = None
LOCAL_MODEL = "/mnt/storage/Models/Chatterbox/checkpoints/chatterbox_finetuned_norwegian"
# LOCAL_MODEL = "/home/alex/PycharmProjects/chatterbox-streaming/checkpoints_no_lora_base_no-epochs_20_accum_80/merged_model"
# LOCAL_MODEL = "/home/alex/PycharmProjects/chatterbox-streaming/checkpoints_grpo/merged_grpo_model"

def load_model():
    if LOCAL_MODEL is None:
        model = ChatterboxTTS.from_pretrained(DEVICE)
    else:
        model = ChatterboxTTS.from_local(ckpt_dir=LOCAL_MODEL, device=DEVICE)
    return model


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


def generate(model, text, audio_prompt_path, exaggeration, temperature, seed_num, cfgw):
    if model is None:
        # model = ChatterboxTTS.from_pretrained(DEVICE)
        print("Model still loading...")
        return

    if seed_num != 0:
        set_seed(int(seed_num))
    else:
        seed_num = random.randint(1, 10000)
        set_seed(seed_num)
    print("Using seed:", seed_num)

    wav = model.generate(
        text,
        audio_prompt_path=audio_prompt_path,
        exaggeration=exaggeration,
        temperature=temperature,
        cfg_weight=cfgw,
    )
    return (model.sr, wav.squeeze(0).numpy())

callback = gr.CSVLogger()
with gr.Blocks() as demo:
    model_state = gr.State(None)  # Loaded once per session/user

    with gr.Row():
        with gr.Column():
            text = gr.Textbox(value="Dette er en test av Chatterbox TTS-modellen. Vi skal se hvordan den håndterer norsk tekst.", label="Text to synthesize")
            ref_wav = gr.Audio(sources=["upload", "microphone"], type="filepath", label="Reference Audio File", value=None)
            exaggeration = gr.Slider(0.25, 2, step=.05, label="Exaggeration (Neutral = 0.5, extreme values can be unstable)", value=.5)
            cfg_weight = gr.Slider(0.2, 1, step=.05, label="CFG/Pace", value=0.5)

            with gr.Accordion("More options", open=False):
                seed_num = gr.Number(value=0, label="Random seed (0 for random)")
                temp = gr.Slider(0.05, 5, step=.05, label="temperature", value=.8)

            run_btn = gr.Button("Generate", variant="primary")

        with gr.Column():
            audio_output = gr.Audio(label="Output Audio")
            flag_type = gr.Radio(["Good", "Bad"], label="Flag category", value="Bad")  # Placeholder for output type
            flag_btn = gr.Button("Flag")

    demo.load(fn=load_model, inputs=[], outputs=model_state)

    run_btn.click(
        fn=generate,
        inputs=[
            model_state,
            text,
            ref_wav,
            exaggeration,
            temp,
            seed_num,
            cfg_weight,
        ],
        outputs=audio_output,
    )
    # This needs to be called at some point prior to the first call to callback.flag()
    callback.setup([flag_type, text, ref_wav, exaggeration, cfg_weight, seed_num, temp, audio_output], "flagged_data_points")
    # We can choose which components to flag -- in this case, we'll flag all of them
    flag_btn.click(lambda *args: callback.flag(list(args)), [flag_type, text, ref_wav, exaggeration, cfg_weight, seed_num, temp, audio_output], None, preprocess=False)

if __name__ == "__main__":
    demo.queue(
        max_size=50,
        default_concurrency_limit=1,
    ).launch(share=False)
