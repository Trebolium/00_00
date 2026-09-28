import asyncio

from . import transcript, tts
from .asr import transcribe
from .audio_io import Mic, Speaker
from .llm import respond

SYSTEM_PROMPT = {
    "role": "system",
    "content": "You are a helpful, concise voice assistant. Keep replies short and conversational.",
}


async def handle_turn(pcm: bytes, speaker: Speaker, history: list[dict]):
    """ASR -> LLM -> TTS -> playback for one recorded utterance."""
    text = await asyncio.to_thread(transcribe, pcm)
    if not text:
        return
    print(f"user: {text}")
    transcript.append("user", text)
    history.append({"role": "user", "content": text})

    reply = await asyncio.to_thread(respond, history)
    print(f"assistant: {reply}")
    transcript.append("assistant", reply)
    history.append({"role": "assistant", "content": reply})

    pcm_out, sample_rate = await tts.synthesize(reply)
    await asyncio.to_thread(speaker.play, pcm_out, sample_rate)


async def run_cycle(mic: Mic, speaker: Speaker, history: list[dict]):
    """Record one utterance, then race it against a barge-in so a new utterance can cut it off."""
    first_frame = await asyncio.to_thread(mic.wait_for_speech)
    pcm = await asyncio.to_thread(mic.record_utterance, first_frame)

    turn = asyncio.create_task(handle_turn(pcm, speaker, history))
    barge_in = asyncio.create_task(asyncio.to_thread(mic.wait_for_barge_in))

    await asyncio.wait({turn, barge_in}, return_when=asyncio.FIRST_COMPLETED)
    for task in (turn, barge_in):
        if not task.done():
            task.cancel()
    speaker.stop()


async def main():
    mic = Mic()
    speaker = Speaker()
    mic.start()
    history = [SYSTEM_PROMPT]

    print("Listening... speak to start the conversation.")
    while True:
        await run_cycle(mic, speaker, history)


def run():
    asyncio.run(main())


if __name__ == "__main__":
    run()
