"""Sikio: talk to the AI command-line tool you already pay for.

Speech-to-text and text-to-speech run locally and are free. The thinking is
done by your own Claude Code (with its tools, skills, MCP servers and settings),
by any OpenAI-compatible chat server given by URL, by another AI CLI you are
logged in to (codex, gemini, cursor), or by any command that reads a prompt on stdin,
such as "ollama run llama3.2".
"""
import argparse
import asyncio
import collections
import concurrent.futures
import gc
import http.client
import json
import os
import queue
import re
import shlex
import shutil
import subprocess
import sys
import threading
import time
import traceback
import urllib.error
import urllib.request
from pathlib import Path, PureWindowsPath

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")  # Windows prints a long harmless warning otherwise

import keyring
from keyring.errors import KeyringError
import numpy as np
try:
    import sounddevice as sd
except OSError as error:  # Linux without the PortAudio system library: the settings window and the tests still work
    sd, NO_SOUND = None, error
from huggingface_hub import constants as hf_constants
from claude_agent_sdk import (
    AssistantMessage, ClaudeAgentOptions, ClaudeSDKClient, ClaudeSDKError, PermissionResultAllow,
    PermissionResultDeny, ResultMessage, StreamEvent, ToolUseBlock,
)
from faster_whisper import WhisperModel

RATE = 16000  # what Whisper expects
BLOCK = 1600  # 0.1 s of audio
# Voice queued before playing on after running dry mid-sentence. Measured with the processor overloaded (voice made at half
# speed): the same 8.6 s of pauses came as 12 pauses instead of 35. Waiting like this before every sentence cost 0.07 to
# 0.25 s of first sound and prevented no gaps: without overload the voice ran dry in 1 of 60 sentences.
RESUME_SECONDS = 0.32
CACHE = Path.home() / ".cache" / "sikio"
SESSION = CACHE / "session"


def config_dir():
    """Each system's usual folder for an app's settings, which also holds the tray app's log and lock."""
    if os.name == "nt":
        return Path(os.environ.get("APPDATA") or Path.home() / "AppData" / "Roaming")
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support"
    return Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")


SETTINGS = config_dir() / "sikio" / "settings.json"
KEY_SERVICE = "sikio"  # API keys live in the OS credential store, one per web address, never in SETTINGS
DEFAULTS = {"brain": "claude", "model": None, "confirm": True, "wake": "juniper", "voice": "michael", "stt_model": "base.en", "chat": "own"}
STYLE = """You are a voice assistant. Every word of your reply is spoken aloud by a text-to-speech voice, so write for the ear, not the eye.
- Sound like a relaxed, capable person talking. Use contractions and everyday words.
- When it fits, open with a short natural marker such as "Sure.", "Alright,", "Okay, so" or "Hmm,". Not every time.
- Short sentences, one idea each. Put commas and dashes where a speaker would breathe.
- Mirror the user: match their wording, energy and formality. A brief question gets a brief answer.
- Lead with the answer. One to three sentences unless asked for more.
- Write numbers, units, symbols and abbreviations the way they are said, like "three thirty p m" or "twenty dollars".
- No markdown, lists, code, emoji, stage directions or bracketed tags. They would be read out literally."""
# Only for Claude Code: no other brain has the spoken yes, so telling it the app asks would let it act unasked.
CONFIRM_STYLE = """
- Speech is often misheard. Before your first action that changes anything, say in one short sentence what you understood and are about to do. The app then asks the user for a spoken yes by itself, so never ask "shall I go ahead" yourself. Questions and read-only work need no confirmation."""
# Users add words the voice mispronounces here, as whole-word respellings. Measured: the voice says API, CLI
# and GitHub right as written; "Ollama" came out as "Alama" in 2 of 3 tries, respelled as "Ola Ma" or "Olama".
SAY = {"Ollama": "oh lah ma"}
JOINERS = {"but", "so", "because", "which", "while"}
# "and" usually joins two words ("warm and dry"); it only earns a pause when a new clause follows it.
CLAUSE_STARTS = {"i", "you", "we", "it", "they", "he", "she", "that", "this", "there", "then", "if", "when"}
# A command brain gets the request on stdin, so spoken words never reach a command line (through a .cmd shim, such as
# npm's codex.cmd, cmd.exe would run an "&" in them). Cursor's agent documents no stdin, so REQUEST marks the argument
# that gets the request, which is refused for batch files. These agents act on their own and get no spoken yes, so each
# preset keeps the tool's own read-only mode: they can answer and read files but not change them or run commands, at the
# cost of tasks that need an action. Codex: read-only sandbox, never asks in exec mode; --skip-git-repo-check since the
# tray app works in the home folder, harmless when nothing can be written. Gemini: headless "default" mode denies every
# tool that would ask (its "plan" mode is overridden in untrusted folders). Cursor: Ask mode.
REQUEST = "\0request"
BRAINS = {"codex": ["codex", "exec", "--sandbox", "read-only", "--skip-git-repo-check", "-"],
          "gemini": ["gemini", "--approval-mode", "default"],
          "cursor": ["agent", "--print", "--mode", "ask", "--output-format", "text", REQUEST]}
TIMEOUT = 300
CONNECT_SECONDS = 10  # an unreachable web address fails this soon; the first word may still take TIMEOUT (a local model loading)
CONFIRM_SECONDS = 10
QUIET_BLOCKS = 8  # 0.8 s of quiet ends an utterance
PAUSE_BLOCKS = 3  # after 0.3 s of quiet a draft transcription starts, used if the utterance then ends
# Over a reply a draft is also made of the first 0.8 s (the 0.3 s before the speech began and 0.5 s of it), to hear the wake
# word before the words end. Measured on 4 voices: the wake word was heard in 16 of 16 such heads (12 of 16 at 0.7 s), and
# none of 20 heads of speech without it stopped the reply.
HEAD_BLOCKS = 8
FOLLOW_UP_SECONDS = 8  # after a reply that ends with a question, an answer starting this soon needs no wake word
YES = {"yes", "yeah", "yep", "yup", "sure", "okay", "ok", "alright"}
YES_PHRASES = ("go ahead", "do it", "please do", "go for it")
STOPS = {"", "stop", "stop it", "stop talking", "please stop", "stop please", "quiet", "be quiet", "enough", "that's enough",
         "cancel", "never mind", "nevermind", "shut up", "hush", "thanks", "thank you", "okay", "ok"}  # "Juniper, stop" just stops
NO = {"no","nope", "not", "don't", "dont", "stop", "wait", "cancel", "never", "but"}
ACTIONS = {"Bash": "run a command", "PowerShell": "run a command", "Write": "write the file", "Edit": "edit the file",
           "MultiEdit": "edit the file", "NotebookEdit": "edit the notebook"}
# Without the last sentence the agent sometimes ended its reply by announcing the action again, as if about to retry.
DENIED = ("The user did not confirm this action out loud, so it was not done. Tell them briefly that you did not do it. "
          "Do not announce it again or retry it in this reply.")
APOLOGY_FAILED = "Sorry, the assistant failed. The error is in the terminal."
APOLOGY_SLOW = "Sorry, that took too long."
QUIT = threading.Event()
PAUSED = threading.Event()  # the tray's Pause: the mic's sound is replaced by silence, so nothing is heard or transcribed
NEW_CHAT = threading.Event()  # the tray's New chat: taken as if the wake word and "new chat" were said


def on_state(state, line):
    """Replaced by the tray app, to show on its icon: starting, listening, thinking, speaking, confirming or error."""


def report(state, line=""):
    """Tell the user what the app is doing: `line` is printed (to the terminal, or the tray app's log), both go to on_state."""
    if line:
        print(line, flush=True)
    on_state(state, line)


def utterances(blocks, threshold, quiet_blocks=QUIET_BLOCKS, max_blocks=300, wait=None, paused=None, head=None):
    """Group audio blocks into utterances: loud blocks up to 0.8 s of quiet. Ends if no speech starts within `wait` blocks.
    paused(audio so far) is called after PAUSE_BLOCKS of quiet, and once the utterance is `head` blocks long, so the caller
    can start transcribing before the end is sure."""
    lead, speech, quiet, waited = collections.deque(maxlen=3), [], 0, 0
    for block in blocks:
        loud = np.sqrt(np.mean(block**2)) > threshold
        if not speech:
            if loud:
                speech = [*lead, block]
            else:
                lead.append(block)
                waited += 1
                if wait is not None and waited >= wait:
                    return
            continue
        speech.append(block)
        quiet = 0 if loud else quiet + 1
        if paused and (quiet == PAUSE_BLOCKS or len(speech) == head):
            paused(np.concatenate(speech))
        if quiet >= quiet_blocks or len(speech) >= max_blocks:
            yield np.concatenate(speech)
            speech, quiet = [], 0
            lead.clear()


def mic_blocks(heard, stop=None):
    """The mic's blocks from the queue its stream fills, until the app quits or `stop` is set (so a listening thread never hangs)."""
    last = time.monotonic()
    while not QUIT.is_set() and not (stop and stop.is_set()):
        try:
            block = heard.get(timeout=0.2)
        except queue.Empty:
            if time.monotonic() - last > 5:  # the callback puts a block every 0.1 s or so, even a silent one
                sys.exit("The microphone stopped. Reconnect it and start again.")
            continue
        yield block
        last = time.monotonic()  # not counting the time the caller took


def after_wake(text, wake):
    """The words after the wake word, or None when it was not said."""
    found = re.search(rf"\b{re.escape(wake)}\b[\s,.!?]*", text, re.IGNORECASE)
    return text[found.end():].strip() if found else None


def request_in(heard, wake, open_):
    """The request in what was heard: the words after the wake word, or while open (just woken, or a reply asked a question)
    everything heard; None when it was not for us, so silence and noise are never a request."""
    found = after_wake(heard, wake)
    return heard.strip() if found is None and open_ and heard.strip() else found


def asks(text):
    """Whether a reply ended with a question, so the answer may come without the wake word."""
    return text.rstrip(" \"'”’)").endswith("?")


def interruption(heard, wake, saying):
    """What the user said over a reply: the new request, "" to just stop, or None to keep talking.
    Only words that start with the wake word count: anything else may be noise, someone else, or our own voice coming back
    through the mic on computers without echo cancellation. Words that are part of the reply itself are our own voice
    (but not the bare wake word: at worst our own voice saying just that stops us)."""
    said = words(heard)
    if not re.match(rf"\W*{re.escape(wake)}\b", heard, re.IGNORECASE) or len(said) > 1 and " ".join(said) in " ".join(words(saying)):
        return None
    request = after_wake(heard, wake)
    return "" if " ".join(words(request)) in STOPS else request


def sentences(text):
    return [part for part in re.split(r"(?<=[.!?])\s+", text.strip()) if part]


def complete_sentences(text):
    """Split streamed text into the sentences already finished and the unfinished rest."""
    *done, rest = re.split(r"(?<=[.!?])\s+", text)
    return [part.strip() for part in done if part.strip()], rest.lstrip()


def sse_text(line):
    """The reply text in one server-sent-events line of a streamed chat completion: "" if none, None at the end."""
    if not line.startswith("data:"):
        return ""
    data = line[5:].strip()
    if data == "[DONE]":
        return None
    chunk = json.loads(data or "{}")
    if chunk.get("error"):  # OpenRouter reports a failure after the start as a chunk like this
        raise ValueError(f"The service stopped with an error: {chunk['error']}")
    choices = chunk.get("choices") or [{}]
    return (choices[0].get("delta") or {}).get("content") or ""  # reasoning ("reasoning", "reasoning_content") is not said


def breathe(text):
    """Add a comma before a joining word in long unpunctuated stretches, because the voice only pauses at punctuation."""
    def pause(stretch):
        words, out = stretch.group().split(), []
        while len(words) > 10:
            at = next((
                i for i, w in enumerate(words) if i >= 5 and len(words) - i >= 4
                and (w.lower() in JOINERS or w.lower() == "and" and words[i + 1].lower() in CLAUSE_STARTS)
            ), None)
            if at is None:
                break
            out.append(" ".join(words[:at]) + ",")
            words = words[at:]
        text = stretch.group()  # keeps the space after the punctuation before it
        return text[:len(text) - len(text.lstrip())] + " ".join(out + [" ".join(words)]) + text[len(text.rstrip()):] if out else text
    return re.sub(r"[^,;:.!?\-–—]+", pause, text)


def spoken(text):
    """The reply cleaned for the voice: no markdown or one-word tags, hard words respelled."""
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"\[\w+\]", "", text)
    text = re.sub(r"[*`#]+|(?<!\w)_|_(?!\w)", "", text)
    for word, say in SAY.items():
        text = re.sub(rf"\b{re.escape(word)}\b", say, text)
    return re.sub(r"[ \t]{2,}", " ", text).strip()


def words(text):
    return re.findall(r"[a-z']+", text.lower())


def is_yes(text):
    """Only a clear yes allows an action; any no, hesitation or silence denies it."""
    said = words(text.replace("’", "'"))
    joined = " ".join(said)
    return not NO & set(said) and bool(YES & set(said) or any(re.search(rf"\b{p}\b", joined) for p in YES_PHRASES))


def is_new_chat(text):
    return re.fullmatch(r"(start )?(a )?new (chat|conversation)( please)?|start over( please)?", " ".join(words(text))) is not None


def question(tool, data):
    """The spoken yes/no question for a tool call: the action and its one key field, never file contents."""
    def short(value):
        value = " ".join(str(value).split())
        return value if len(value) <= 80 else value[:77] + "..."
    if tool.startswith("mcp__"):
        _, server, name = (tool.split("__", 2) + ["", ""])[:3]
        action = f"use the tool {name.replace('_', ' ')} from the {server.replace('_', ' ').replace('-', ' ')} server"
    elif "command" in data:
        action = f"{ACTIONS.get(tool, 'run a command')}: {short(data['command'])}"
    elif data.get("file_path") or data.get("notebook_path"):
        action = f"{ACTIONS.get(tool, f'use {tool} on the file')} {short(PureWindowsPath(data.get('file_path') or data['notebook_path']).name)}"  # PureWindowsPath: splits at \ and / on every system
    else:
        key = next((data[k] for k in ("url", "path", "pattern", "query", "skill") if data.get(k)), None)
        action = f"use {tool}" + (f": {short(key)}" if key else "")
    return f"I'd like to {action}. Shall I go ahead?"


# Models are loaded from the cache first, without the network: with the network down, Hugging Face's retries took
# 702 s before it used the cached voice. HUB keeps the voice's offline attempt apart from a download of Whisper.
HUB = threading.Lock()


def load_stt(model):
    # shortcut: CPU only, under 1 s per sentence with base.en. Use device="cuda" once a GPU setup is worth supporting.
    try:
        return WhisperModel(model, device="cpu", compute_type="int8", local_files_only=True)
    except Exception:  # not downloaded yet
        with HUB:
            try:
                return WhisperModel(model, device="cpu", compute_type="int8")
            except Exception as error:
                sys.exit(f"Could not load speech recognition {model!r} ({error}). The first run needs the internet. Models: base.en, small.en, small.")


def transcribe(stt, audio):
    # Measured on synthetic noise (clicks, hum, fan, cough, buzz) vs synthesised speech at 10 dB SNR: speech scores no_speech_prob <= 0.02, a voiced buzz 0.63.
    # Only noise-side cutoffs: a stricter voice threshold or a confidence cutoff rejected no extra noise and risks ignoring a real, mumbled request.
    # Greedy decoding (beam_size=1): on 120 synthetic requests the wake word was caught as often as with the default beam of 5
    # (119 of 120 each), and transcribing took 1.13 s instead of 1.53 s (median) on the test laptop.
    segments, _ = stt.transcribe(audio, beam_size=1, vad_filter=True, vad_parameters={"min_speech_duration_ms": 250})
    return " ".join(s.text.strip() for s in segments if s.no_speech_prob < 0.4)


def voice_name(name):
    """A built-in voice; anything else, such as a voice of the old engine in a saved settings file, is the default voice.
    Only these names reach the voice model, never a path: it would clone the voice in that file."""
    if name in VOICES.values():
        return name
    print(f"Unknown voice {name!r}, using {DEFAULTS['voice']}. The voices are: {', '.join(VOICES.values())}.")
    return DEFAULTS["voice"]


class Speaker:
    """What the sound device plays: chunks of a voice queued as they are made, played back to back from the device's
    callback, so a sentence starts while the rest of it is still being made."""

    def __init__(self, rate, device_rate=None):
        self.rate, self.device_rate = rate, device_rate or rate
        self.chunks, self.left = queue.Queue(), np.zeros(0, np.float32)
        self.making = self.started = self.dry = False  # making: more of the sentence is coming; started: it is playing
        self.gaps = 0  # times the voice ran out mid-sentence
        self.latency = 0.0  # the device plays this long after the callback

    def play(self, chunk):
        if self.device_rate != self.rate:  # shortcut: linear resampling, only for a device that refuses the voice's rate
            n = round(len(chunk) * self.device_rate / self.rate)
            chunk = np.interp(np.arange(n) * self.rate / self.device_rate, np.arange(len(chunk)), chunk)
        self.chunks.put(np.asarray(chunk, np.float32))

    def queued(self):
        return len(self.left) + sum(len(chunk) for chunk in list(self.chunks.queue))

    def fill(self, out, frames, time_info, status):
        if self.dry and self.making and self.queued() < RESUME_SECONDS * self.device_rate:
            out.fill(0)  # ran dry mid-sentence (a busy processor): a few longer pauses instead of a stutter per chunk
            return
        done = 0
        while done < frames:
            if not len(self.left):
                try:
                    self.left = self.chunks.get_nowait()
                except queue.Empty:
                    break
            n = min(frames - done, len(self.left))
            out[done:done + n, 0] = self.left[:n]
            self.left, done = self.left[n:], done + n
        out[done:].fill(0)
        playing = self.started or done > 0
        self.dry = done < frames and self.making and playing
        self.gaps += self.dry
        self.started = playing and (self.making or self.queued() > 0)

    def wait(self):
        """Until everything queued has been heard."""
        last, since = self.queued(), time.monotonic()
        while (self.making or self.queued()) and not QUIT.is_set():
            time.sleep(0.01)
            if self.queued() != last:
                last, since = self.queued(), time.monotonic()
            elif last and not self.making and time.monotonic() - since > 2:  # nothing played for 2 s: the device is gone
                # (not while making: a busy processor can keep the voice paused for its reserve that long)
                break
        time.sleep(self.latency)

    def stop(self):
        """Silent within one callback: what is queued is dropped."""
        while not self.chunks.empty():
            self.chunks.get_nowait()
        self.left = np.zeros(0, np.float32)


def open_speaker(rate):
    """A Speaker with its output stream open for the whole run: a stream started per sentence took 0.3 s to sound."""
    speaker = Speaker(rate)
    try:
        try:
            stream = sd.OutputStream(rate, channels=1, dtype="float32", latency="low", callback=speaker.fill)
        except sd.PortAudioError:  # a device that cannot play the voice's rate plays its own
            speaker = Speaker(rate, int(sd.query_devices(kind="output")["default_samplerate"]))
            stream = sd.OutputStream(speaker.device_rate, channels=1, dtype="float32", latency="low", callback=speaker.fill)
        stream.start()
    except sd.PortAudioError as error:
        sys.exit(f"Could not open a sound output ({error}). Connect speakers or headphones and start again.")
    speaker.stream, speaker.latency = stream, stream.latency
    return speaker


def load_tts(name):
    """Pocket TTS and the state of one of its built-in voices."""
    # Imported here: it took 28 s on the test laptop (PyTorch and its compiler), which would also hold up the settings
    # window; here it runs in the loading thread, beside Whisper and Claude's connection.
    from pocket_tts import TTSModel

    def load():
        model = TTSModel.load_model()
        return model, model.get_state_for_audio_prompt(name)
    # shortcut: tells a first download from a normal start by the Hugging Face cache folder of the model
    if (Path(hf_constants.HF_HUB_CACHE) / "models--kyutai--pocket-tts-without-voice-cloning").exists():
        with HUB:  # every file of the voice is pinned to one version, so the cached copy is never out of date
            offline, hf_constants.HF_HUB_OFFLINE = hf_constants.HF_HUB_OFFLINE, True
            try:
                return load()
            except Exception:  # a file is missing: download it below
                pass
            finally:
                hf_constants.HF_HUB_OFFLINE = offline
    report("starting", "Downloading the voice, about 230 MB (first run only)...")
    try:
        return load()
    except Exception as error:  # no network on the first run, or the Hugging Face servers down
        sys.exit(f"Could not load the voice. {f'{type(error).__name__}: {error}'[-300:]}\n"
                 "The first run needs the internet to download it.")


class Voice:
    """The voice model with one of its voices, and the speaker it plays on (a test passes Speaker: no sound device)."""

    def __init__(self, name, speaker=open_speaker):
        self.model, self.state = load_tts(name)
        self.speaker = speaker(self.model.sample_rate)

    def say(self, text, hush):
        """Queue `text` as it is made; returns when it is all made, not heard. Stops making it once `hush` or QUIT is set."""
        self.speaker.making = True
        try:
            for chunk in self.model.generate_audio_stream(self.state, text, stop=hush):
                if QUIT.is_set() or hush.is_set():
                    break
                self.speaker.play(chunk.numpy())
        finally:
            self.speaker.making = False

    def wait(self):
        self.speaker.wait()

    def stop(self):
        self.speaker.stop()


def warm_up(tts):
    """The first sentence made is slower (the model's first run): pay that while starting, not in the first reply."""
    tts.model.generate_audio(tts.state, "Ready.")


def speak(tts, piece, hush=None):
    """Start saying one piece; it keeps playing while the next one is made, tts.wait() waits for the end. Silent once `hush` is set."""
    print(f"< {piece}", flush=True)
    hush = hush or threading.Event()
    for sentence in sentences(breathe(spoken(piece))):
        if QUIT.is_set() or hush.is_set():
            return
        if re.search(r"\w", sentence):  # measured: given only "..." or "-", the voice made up words ("Tofu", "The post")
            tts.say(sentence, hush)


SPENT = {}  # chat id -> total cost after its last request seen here; Claude Code reports a running total for the whole chat


def cost_text(session_id, total):
    """"$0.017 this request, $0.098 this chat". The total also covers earlier requests of a resumed chat, so the first request
    of a chat resumed from the last run has no known share and only the total is given."""
    before = SPENT.get(session_id)
    SPENT[session_id] = total
    if before is None and SESSION.exists() and SESSION.read_text().strip() == session_id:
        return f"${total:.3f} this chat"
    return f"${total - (before or 0):.3f} this request, ${total:.3f} this chat"


def latest_session():
    """(session id, working folder or None, seconds since it was last used) of the newest Claude Code session, or None."""
    root = Path(os.environ.get("CLAUDE_CONFIG_DIR") or Path.home() / ".claude") / "projects"
    files = [f for f in root.glob("*/*.jsonl") if re.fullmatch(r"[0-9a-f]{8}(-[0-9a-f]{4}){3}-[0-9a-f]{12}", f.stem)]
    for file in sorted(files, key=lambda f: f.stat().st_mtime, reverse=True)[:1]:
        folder = None
        with file.open(encoding="utf-8", errors="ignore") as lines:
            for line in lines:
                try:
                    folder = json.loads(line).get("cwd")
                except (ValueError, AttributeError):
                    continue
                if folder:
                    break
        return file.stem, folder, time.time() - file.stat().st_mtime
    return None


def ago(seconds):
    minutes = max(1, round(seconds / 60))
    return f"{minutes} minutes" if minutes < 60 else f"{round(minutes / 60)} hours" if minutes < 2880 else f"{round(minutes / 1440)} days"


class Claude:
    """Your Claude Code in one long-lived session that is resumed across restarts."""

    def __init__(self, model, confirm, chat="own"):
        self.first = chat == "latest"  # until the first result: branch off your latest Claude Code session instead of the own chat
        exe = shutil.which("claude")
        if not exe:
            sys.exit("'claude' is not installed or not on PATH. Install Claude Code and log in, or pick another --brain.")
        self.confirm = confirm
        if not confirm:
            print("Spoken confirmation is OFF: Claude acts without asking you (your own Claude Code permission rules still apply).")
        self.options = ClaudeAgentOptions(
            cli_path=exe, model=model, include_partial_messages=True, can_use_tool=self.gate,
            system_prompt={"type": "preset", "preset": "claude_code", "append": STYLE + CONFIRM_STYLE},
            setting_sources=["user", "project", "local"],
            permission_mode="default",  # a looser default mode in the user's settings must not skip the spoken yes
            disallowed_tools=["AskUserQuestion"],  # it would need a screen; the agent asks out loud instead
        )
        self.client = self.pumping = None
        self.events = asyncio.Queue()
        self.connecting = asyncio.Lock()  # a request waits for the connection made at startup instead of making a second one

    async def connect(self):
        """Connect while the speech models load (about 2 s with the user's MCP servers), so the first request does not wait for it."""
        async with self.connecting:
            try:
                await self.start()
                return True
            except Exception as error:  # no network, not logged in: say so now, and the first request tries again
                await self.stop()
                print(f"Could not start Claude Code, will try again at your first request. {f'{type(error).__name__}: {error}'[-400:]}", file=sys.stderr)
                return False

    async def start(self):
        self.options.resume = SESSION.read_text().strip() if SESSION.exists() else None
        if self.first:
            try:
                found = latest_session()
            except OSError as error:
                found = None
                print(f"Could not look for your Claude Code sessions. ({error})")
            if found:
                self.options.resume, self.options.fork_session = found[0], True  # a branch: the original is not written to
                folder = found[1] if found[1] and Path(found[1]).is_dir() else None
                self.options.cwd = folder
                print(f"Continuing your latest Claude Code session in {found[1] or 'its folder'} (last used {ago(found[2])} ago).")
            else:
                print("No Claude Code session found, using Sikio's own chat.")
        try:
            self.client = ClaudeSDKClient(self.options)
            await self.client.connect()
        except ClaudeSDKError as error:
            if not self.options.resume:
                raise
            print(f"Could not resume the last chat, starting a new one. ({str(error)[-200:]})")
            self.options.resume, self.options.fork_session, self.options.cwd = None, False, None
            self.client = ClaudeSDKClient(self.options)
            await self.client.connect()

    async def stop(self):
        client, self.client = self.client, None
        if client:
            await client.disconnect()

    async def ask(self, text):
        """Yield ("say", sentence) as the reply streams in, and ("confirm", question, future) when an action needs a yes."""
        if is_new_chat(text):
            SESSION.unlink(missing_ok=True)
            self.first, self.options.fork_session, self.options.cwd = False, False, None
            await self.stop()
            yield "say", "Okay, new chat."
            return
        self.events, self.seen, self.allowed, self.usage = asyncio.Queue(), set(), False, ""
        self.passed, self.asking, self.interrupted, self.pumping = asyncio.Condition(), asyncio.Lock(), False, None
        try:
            async with self.connecting:
                if not self.client:
                    await self.start()
            self.pumping = asyncio.create_task(self.pump(text))
            while (event := await asyncio.wait_for(self.events.get(), TIMEOUT)) is not None:
                yield event
            await self.pumping
            if self.usage:
                print(self.usage)
        except Exception as error:  # the app keeps running whatever went wrong
            print(f"{type(error).__name__}: {error}"[-500:], file=sys.stderr)
            if self.pumping:
                self.pumping.cancel()
            await self.stop()  # the next request reconnects and resumes the chat
            yield "say", APOLOGY_SLOW if isinstance(error, (TimeoutError, asyncio.TimeoutError)) else APOLOGY_FAILED

    async def pump(self, text):
        try:
            await self.client.query(text)
            pending = ""
            async for message in self.client.receive_response():
                if isinstance(message, StreamEvent) and message.parent_tool_use_id is None:
                    delta = message.event.get("delta") or {}
                    if delta.get("type") == "text_delta":
                        done, pending = complete_sentences(pending + delta.get("text", ""))
                        for sentence in done:
                            await self.events.put(("say", sentence))
                elif isinstance(message, AssistantMessage) and message.parent_tool_use_id is None:
                    if pending.strip():
                        await self.events.put(("say", pending.strip()))
                    pending = ""
                    async with self.passed:
                        self.seen.update(block.id for block in message.content if isinstance(block, ToolUseBlock))
                        self.passed.notify_all()
                elif isinstance(message, ResultMessage):
                    if message.is_error:
                        raise ClaudeSDKError(message.result or str(message.errors))
                    costs = cost_text(message.session_id, message.total_cost_usd or 0)  # before SESSION is overwritten
                    SESSION.parent.mkdir(parents=True, exist_ok=True)
                    SESSION.write_text(message.session_id)
                    self.first = self.options.fork_session = False  # from here on the branch is the own chat (its folder stays)
                    usage = message.usage or {}  # printed by ask() after the last sentence: the reply streams ahead of the voice
                    self.usage = (f"({', '.join(message.model_usage or {})}: {usage.get('input_tokens')} in, {usage.get('cache_creation_input_tokens')} cache write, "
                                  f"{usage.get('cache_read_input_tokens')} cache read, {usage.get('output_tokens')} out, {costs})")
            if pending.strip():
                await self.events.put(("say", pending.strip()))
        finally:
            await self.events.put(None)

    async def interrupt(self):
        """Stop the turn under way (the user spoke over it). What was said so far stays in the chat, so the next request continues it."""
        self.interrupted = True  # any confirmation still to come is a no
        while not self.events.empty():  # confirmation questions not yet asked out loud
            event = self.events.get_nowait()
            if event and event[0] == "confirm":
                event[2].set_result("")
        if not self.client or not self.pumping or self.pumping.done():
            return
        try:
            await self.client.interrupt()
            # The turn then ends with its result message; the next query must not start before it, or it would read it.
            await asyncio.wait_for(asyncio.gather(self.pumping, return_exceptions=True), 15)
        except Exception as error:
            print(f"Could not interrupt cleanly, reconnecting. {type(error).__name__}: {error}"[-500:], file=sys.stderr)
            self.pumping.cancel()
            await self.stop()  # the next request reconnects and resumes the chat

    async def gate(self, tool, data, context):
        """Claude Code's permission prompt, answered by voice. A yes covers the rest of this request."""
        if not self.confirm:
            return PermissionResultAllow()
        async with self.asking:
            if self.interrupted:
                return PermissionResultDeny(message=DENIED, interrupt=True)
            if not self.allowed:
                async with self.passed:  # the question must come after what was said before the tool call
                    try:
                        await asyncio.wait_for(self.passed.wait_for(lambda: context.tool_use_id in self.seen), 5)
                    except asyncio.TimeoutError:
                        pass  # shortcut: a subagent's tool call is not always streamed to us, so ask anyway after 5 s
                answer = asyncio.get_running_loop().create_future()
                await self.events.put(("confirm", question(tool, data), answer))
                self.allowed = is_yes(await answer)
        return PermissionResultAllow() if self.allowed else PermissionResultDeny(message=DENIED)


class Patient:
    """urllib uses one timeout for connecting and for each read: this one connects within that timeout (CONNECT_SECONDS),
    then waits up to TIMEOUT for each part of the reply."""

    def do_open(self, connection, request, **options):
        class Connection(connection):
            def connect(self):
                super().connect()
                self.sock.settimeout(TIMEOUT)
        return super().do_open(Connection, request, **options)


class NoRedirect(urllib.request.HTTPRedirectHandler):
    """urllib would resend the Authorization header to the address a redirect names, which may be another host."""

    def redirect_request(self, *_):
        return None  # the redirect then surfaces as an HTTPError


PATIENT = urllib.request.build_opener(type("PatientHTTP", (Patient, urllib.request.HTTPHandler), {}),
                                      type("PatientHTTPS", (Patient, urllib.request.HTTPSHandler), {}), NoRedirect)


def web_error(error, model=None):
    """One line saying why a request to a web service failed, naming the cause where its answer shows it.
    Error bodies differ: {"error": {"message", "code", "type"}} at OpenAI and OpenRouter, {"error": "..."} elsewhere."""
    if isinstance(error, urllib.error.HTTPError):
        body = error.read().decode("utf-8", "replace")
        try:
            detail = json.loads(body)
            detail = (detail[0] if isinstance(detail, list) else detail).get("error") or body  # Google's is in a list
        except (ValueError, AttributeError, IndexError):
            detail = body
        message = " ".join(str(detail.get("message", detail) if isinstance(detail, dict) else detail).split())[:300]
        said = body.lower()
        if error.code == 401 or "api key" in said or "api_key" in said:  # xAI and Google answer a wrong key with 400
            cause = "The service refused the API key: save the right one in the settings window, or set SIKIO_API_KEY."
        elif error.code == 402 or re.search(r"insufficient_quota|current quota|credit|billing|balance|spending", said):
            cause = "The account has no credit left or reached its spending limit: check its billing on the service's website."
        elif error.code == 429:
            cause = "Too many requests (rate limited): wait a moment and try again."
        elif "model" in said and re.search(r"not found|not exist|invalid|unknown|not a valid|no such", said):
            cause = f"The service does not know the model {model!r}: pick one with Find models in the settings window."
        elif error.code == 404:
            cause = "Nothing at this address: check it, it usually ends in /v1."
        else:
            cause = "The service refused the request."
        return f"{cause} (HTTP {error.code}: {message or error.reason})"
    if isinstance(error, TimeoutError):
        return f"No answer within {TIMEOUT} s."
    if isinstance(error, urllib.error.URLError):
        reason = f"no answer within {CONNECT_SECONDS} s" if isinstance(error.reason, TimeoutError) else error.reason
        return f"Could not connect ({reason}). Check the address, and that the service is running."
    if isinstance(error, json.JSONDecodeError):
        return f"The service did not answer with a chat stream ({error}). Check the address."
    return f"{error}"[-500:] if isinstance(error, ValueError) else f"{type(error).__name__}: {error}"[-500:]


class Chat:
    """Any OpenAI-compatible chat-completions server, by URL: streamed replies, chat only, no actions."""

    def __init__(self, url, model):
        if not model:
            sys.exit("A URL brain needs --model, for example --model gemma3:4b or --model gpt-4o-mini.")
        self.url, self.model = url.rstrip("/") + "/chat/completions", model
        self.key = os.environ.get("SIKIO_API_KEY") or stored_key(url.rstrip("/"))  # the settings window saves the key under the address without a final slash
        self.history = []  # shortcut: forgotten on restart and grows without limit; trim or save it once chats get long

    def open(self):
        body = {"model": self.model, "stream": True, "messages": [{"role": "system", "content": STYLE}, *self.history]}
        headers = {"Content-Type": "application/json"}
        if self.key:  # local servers need none
            headers["Authorization"] = f"Bearer {self.key}"
        return PATIENT.open(urllib.request.Request(self.url, json.dumps(body).encode(), headers), timeout=CONNECT_SECONDS)

    async def ask(self, text):
        if is_new_chat(text):
            self.history.clear()
            yield "say", "Okay, new chat."
            return
        self.history.append({"role": "user", "content": text})
        reply = pending = ""
        try:
            with await asyncio.to_thread(self.open) as response:
                while (line := await asyncio.to_thread(response.readline)) and (piece := sse_text(line.decode("utf-8", "replace"))) is not None:
                    reply += piece
                    done, pending = complete_sentences(pending + piece)
                    for sentence in done:
                        yield "say", sentence
            if pending.strip():
                yield "say", pending.strip()
            if not reply:
                raise ValueError("The service sent no reply text. Check the address.")
            self.history.append({"role": "assistant", "content": reply})
        except (OSError, ValueError, http.client.HTTPException) as error:  # HTTP errors, refused connections, timeouts, malformed replies
            print(web_error(error, self.model), file=sys.stderr)
            self.history.pop()
            yield "say", APOLOGY_SLOW if isinstance(error, TimeoutError) else APOLOGY_FAILED

    async def interrupt(self):
        pass  # shortcut: the abandoned stream is closed; the interrupted question stays in the history without an answer

    async def stop(self):
        pass


class Command:
    """Any other CLI: one process per request, the whole reply at once, no memory and no actions."""

    def __init__(self, line):
        # Windows paths have backslashes, so split without POSIX escapes and drop the quotes ourselves.
        command = BRAINS.get(line) or [word.strip('"') for word in shlex.split(line, posix=os.name != "nt")]
        exe = shutil.which(command[0])
        if not exe:
            sys.exit(f"'{command[0]}' is not installed or not on PATH. Install it and log in, or pick another --brain.")
        if REQUEST in command and exe.lower().endswith((".bat", ".cmd")):
            sys.exit(f"{exe} is a batch file: Windows would run parts of a spoken request on its command line as commands. "
                     "Put the folder of its .exe first on PATH.")
        self.command, self.name, self.process = [exe, *command[1:]], Path(exe).stem, None

    async def ask(self, text):
        yield "say", await asyncio.to_thread(self.reply, text)

    def reply(self, text):
        prompt = f"{STYLE}\n\n{text}"
        argument = REQUEST in self.command
        try:
            self.process = subprocess.Popen([prompt if part == REQUEST else part for part in self.command],
                                            stdin=subprocess.DEVNULL if argument else subprocess.PIPE, stdout=subprocess.PIPE,
                                            stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace")
        except OSError as error:  # uninstalled since the start, for example
            print(f"Could not start {self.name}: {error}", file=sys.stderr)
            return APOLOGY_FAILED
        try:
            out, err = self.process.communicate(None if argument else prompt, timeout=TIMEOUT)
        except subprocess.TimeoutExpired:
            self.process.kill()
            print(f"{self.name} gave no answer within {TIMEOUT} s, so it was stopped.", file=sys.stderr)
            return APOLOGY_SLOW
        if self.process.returncode == 0 and out.strip():
            return out.strip()
        print((err.strip() or out.strip())[-500:], file=sys.stderr)  # some CLIs report errors on stdout
        # Measured: codex without a login ends with "401 Unauthorized", gemini with "Please set an Auth method" (exit 41).
        login = re.search(r"401|unauthori[sz]ed|auth method|log ?in|api key|credential", err + out, re.IGNORECASE)
        print(f"{self.name} failed (exit code {self.process.returncode})." + (
            f" It may not be logged in or have no API key: run {self.name} once in a terminal to set it up." if login else ""), file=sys.stderr)
        return APOLOGY_FAILED

    async def interrupt(self):
        """End the command: an agent such as codex would otherwise work on after the user said stop."""
        # shortcut: only the command itself is ended, not programs it started in turn
        if self.process and self.process.poll() is None:
            self.process.kill()

    stop = interrupt  # quitting must not leave it running either


async def answer(brain, request, tts, listen, wake=None):
    """Say the reply in order; for an action, ask after everything before it was said, then listen for the yes.
    With a wake word it also listens while it works and talks: the wake word over the reply stops it (see interruption()).
    Returns the last thing said ("" if that was a confirmation question) and the interrupting request, None if not interrupted."""
    hush, said, last, asking, cut = threading.Event(), [], "", None, None

    async def talk():
        nonlocal last, asking, watching
        async for kind, text, *reply in brain.ask(request):
            if hush.is_set():  # a draft heard the wake word (see drafted())
                return
            said.append(text)
            asking = reply[0] if kind == "confirm" else None
            report("speaking")
            await asyncio.to_thread(speak, tts, text, hush)
            last = text if kind == "say" else ""
            if kind == "confirm" and not hush.is_set():
                await asyncio.to_thread(tts.wait)
                await stop_watching()  # the next words are the answer to the question, not an interruption
                report("confirming")
                asking.set_result(await listen(CONFIRM_SECONDS))
                report("thinking")
                watching = watch(fresh=True)
        await asyncio.to_thread(tts.wait)

    def drafted(text):
        """A draft of words said over the reply (their first 0.8 s, or up to a pause), while more may follow: the wake word
        first stops the voice now, not once they end and are transcribed. Measured with "Jarvis, stop." over a reply, from
        the end of the clip to silence: median 0.82 s before, -0.12 s after (silent before the words ended); 1.12 s and
        0.00 s with two cores kept busy."""
        if interruption(text, wake, " ".join(said)) is not None:
            hush.set()
            tts.stop()

    async def overheard(stop, fresh):
        nonlocal cut
        while not stop.is_set():
            heard = await listen(fresh=fresh, stop=stop, drafted=drafted)
            fresh = False
            found = interruption(heard, wake, " ".join(said))
            if found is not None or hush.is_set():  # once a draft stopped the voice it stays stopped, whatever came next
                cut = found or ""
                hush.set()
                tts.stop()
                talking.cancel()
                return

    def watch(fresh):
        stop = threading.Event()
        return stop, asyncio.create_task(overheard(stop, fresh)) if wake else None

    async def stop_watching():
        watching[0].set()
        if watching[1]:
            await asyncio.gather(watching[1], return_exceptions=True)

    talking = asyncio.create_task(talk())
    watching = watch(fresh=True)  # fresh: what was recorded before is the request itself
    try:
        await asyncio.wait([talking])
        if cut is None and hush.is_set():  # stopped by a draft: what was asked for is known once the words end
            await asyncio.wait([watching[1]])
        if cut is None:
            return last, talking.result()  # None, or the error that ended it
        if asking and not asking.done():
            asking.set_result("")  # interrupted while asking: that is a no, never a yes
        await brain.interrupt()
        tts.stop()  # shortcut: a chunk made just before the hush could have been queued meanwhile
        return last, cut
    finally:
        if asking and not asking.done():
            asking.set_result("")  # quit while asking: a no as well
        talking.cancel()
        await stop_watching()


async def typed(seconds=None):
    try:
        return await asyncio.to_thread(input, "> ")
    except EOFError:
        return ""


async def run(args):
    if sd is None:
        sys.exit(f"No sound: {NO_SOUND}. On Linux install PortAudio first, for example: sudo apt install libportaudio2")
    if args.brain == "claude":
        brain = Claude(args.model or "sonnet", confirm=args.confirm, chat=args.chat)
    elif args.brain.startswith(("http://", "https://")):
        brain = Chat(args.brain, args.model)
    else:
        brain = Command(args.brain)
    loading = []

    async def stage(what, starting, done, work):
        """One part of the start, said when it starts and ends. On the test laptop: the voice engine 15 s with a warm disk
        cache, 30 s to 2 min after a reboot (importing PyTorch); Claude Code 2 s; speech recognition 2 s, all at once."""
        began = time.monotonic()
        loading.append(what)
        report("starting", starting)
        try:
            result = await work
        finally:
            loading.remove(what)
        if result is not False:  # False: Claude printed why it could not connect
            still = f" Still waiting for {' and '.join(loading)}." if loading else ""
            report("starting", f"{done} in {time.monotonic() - began:.0f} s.{still}")
        return result
    # Claude's connection and both speech models load at the same time, the models in threads.
    tts = None
    voice = stage("the voice engine", "Loading the voice engine (this takes a while, on every start)...", "Voice engine loaded",
                  asyncio.to_thread(Voice, voice_name(args.voice)))
    connecting = asyncio.create_task(stage("Claude Code", "Connecting to Claude Code...", "Connected to Claude Code",
                                           brain.connect())) if isinstance(brain, Claude) else None
    loading_stt = None if args.text else asyncio.create_task(stage(
        "speech recognition", "Loading speech recognition...", "Speech recognition loaded", asyncio.to_thread(load_stt, args.stt_model)))
    try:
        tts = await voice
        if args.text:
            await answer(brain, args.text, tts, typed)  # typed: no interrupting
            return
        heard_blocks = queue.Queue()
        # The mic runs all the time: a (re)started mic gave 0.3 to 0.55 s of silence on the test laptop, which cut off the
        # wake word of anyone speaking right after a reply, and a stopped one missed everything said while we transcribed.
        try:
            microphone = sd.InputStream(samplerate=RATE, channels=1, dtype="float32", blocksize=BLOCK,
                                        callback=lambda block, *_: heard_blocks.put(block[:, 0] * 0 if PAUSED.is_set() else block[:, 0].copy()))
        except sd.PortAudioError as error:
            sys.exit(f"Could not open a microphone ({error}). Connect one and start again.")
        with microphone, concurrent.futures.ThreadPoolExecutor(1) as drafting:
            warming = asyncio.create_task(asyncio.to_thread(warm_up, tts))  # ends while we listen for the first request
            stt = await loading_stt
            while PAUSED.is_set():  # while paused the blocks are silent, which would make the threshold zero
                await asyncio.sleep(0.2)
            while not heard_blocks.empty():
                heard_blocks.get_nowait()
            blocks = mic_blocks(heard_blocks)
            room = np.concatenate((await asyncio.to_thread(lambda: [next(blocks) for _ in range(10)]))[5:])  # after the silent start
            # Mic levels differ 100x between machines, so "loud" is measured against this room, not a fixed level.
            threshold = max(0.0002, 5 * float(np.sqrt(np.mean(room**2))))

            def text_of(audio):
                return transcribe(stt, audio / np.abs(audio).max())  # quiet mics need full volume for Whisper

            async def listen(seconds=None, fresh=True, stop=None, drafted=None):
                """The next thing said, or "" if no speech starts within `seconds` (or `stop` is set).
                Fresh: drop what was recorded so far, which is our own voice. drafted(text) gets each draft as it is made."""
                while fresh and not heard_blocks.empty():
                    heard_blocks.get_nowait()
                wait = None if seconds is None else max(1, int(seconds * RATE / BLOCK))
                drafts = {}  # transcribed during the pause that may end the utterance, which saves most of the transcription time

                def start_draft(audio):
                    drafts[len(audio)] = made = drafting.submit(text_of, audio)
                    if drafted:
                        made.add_done_callback(lambda made: made.cancelled() or drafted(made.result()))
                try:
                    found = utterances(mic_blocks(heard_blocks, stop), threshold, wait=wait, paused=start_draft,
                                       head=HEAD_BLOCKS if drafted else None)
                    audio = await asyncio.to_thread(next, found, None)
                    draft = None if audio is None else drafts.get(len(audio) - (QUIET_BLOCKS - PAUSE_BLOCKS) * BLOCK)  # no speech after the draft
                    if audio is None or PAUSED.is_set():  # paused while it was being said
                        return ""
                    heard = await (asyncio.wrap_future(draft) if draft else asyncio.to_thread(text_of, audio))
                    if seconds is not None and heard:
                        print(f"> {heard}")
                    return heard
                finally:
                    for made in drafts.values():
                        made.cancel()

            report("listening", f'Listening. Say "{args.wake}" and then your request.' + " Ctrl+C quits." * sys.stdout.isatty())
            follow_up, spoke = 0.0, True  # until time follow_up, the answer to a reply's question needs no wake word
            while True:
                left = follow_up - time.monotonic()
                report("listening")
                heard = await listen(left if left > 0 else None, fresh=spoke, stop=NEW_CHAT)  # a timed listen prints what it heard itself
                if NEW_CHAT.is_set():
                    NEW_CHAT.clear()
                    heard = f"{args.wake}, new chat"
                spoke = False
                await warming  # before speaking, so the two never play at once
                request = request_in(heard, args.wake, left > 0)
                if heard and request is None:
                    print(f"(heard without the wake word: {heard})")
                if request is not None:
                    follow_up = 0.0
                if request:
                    spoke = True
                    report("thinking")
                    if left <= 0:
                        print(f"> {request}")
                    last, cut = await answer(brain, request, tts, listen, args.wake)
                    while cut:  # interrupted by a new request: it is the next request in the same chat
                        print(f"(interrupted) > {cut}")
                        # The whole reply is in the chat already, so tell the brain how far the user actually heard it.
                        told = cut if is_new_chat(cut) else f'(I interrupted you while you were saying "{last}") {cut}'
                        last, cut = await answer(brain, told, tts, listen, args.wake)
                    if cut == "":
                        print("(interrupted)")
                    elif asks(last):
                        follow_up = time.monotonic() + FOLLOW_UP_SECONDS
                elif request == "":  # only the wake word: take the next thing said as the request, for a few seconds
                    spoke = True
                    report("speaking")
                    await asyncio.to_thread(speak, tts, "Yes?")
                    await asyncio.to_thread(tts.wait)
                    follow_up = time.monotonic() + FOLLOW_UP_SECONDS
    finally:
        QUIT.set()  # Ctrl+C: Python waits for the speaking thread before exiting, so cut the voice off now
        if tts:
            tts.stop()
            tts.speaker.stream.close()  # the tray app restarts run() in the same process after a change of settings
        if connecting:  # Ctrl+C during startup: end the connection attempt before closing the client it is making
            connecting.cancel()
            await asyncio.gather(connecting, return_exceptions=True)
        await brain.stop()


def load_settings():
    """The usable values in the settings file; a missing or broken file counts as empty."""
    try:
        saved = json.loads(SETTINGS.read_text(encoding="utf-8"))
        if not isinstance(saved, dict):
            raise ValueError("not a JSON object")
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as error:
        print(f"Ignoring the settings file {SETTINGS}: {error}", file=sys.stderr)
        return {}
    return {k: v for k, v in saved.items() if k in DEFAULTS and isinstance(v, bool if k == "confirm" else str)}


def save_settings(values):
    SETTINGS.parent.mkdir(parents=True, exist_ok=True)
    SETTINGS.write_text(json.dumps(values, indent=2), encoding="utf-8")


def merged(options, saved):
    """Command line over the settings file over the built-in defaults; options not given on the command line are None."""
    got = {**DEFAULTS, **saved, **{k: v for k, v in options.items() if k in DEFAULTS and v is not None}}
    if isinstance(got["wake"], str):
        got["wake"] = re.sub(r"[^\w' ]", "", got["wake"]).strip() or DEFAULTS["wake"]
    if isinstance(got["brain"], str):
        got["brain"] = got["brain"].strip() or DEFAULTS["brain"]
    return got


def stored_key(url):
    """The key saved for this web address only, so a key for one service is never sent to another."""
    try:
        return keyring.get_password(KEY_SERVICE, url)
    except KeyringError:  # no credential store on this machine
        return None


# Window labels and what each means for the app. The brain is "claude", a command, or a web address (its /chat/completions server).
GEMINI_API = "https://generativelanguage.googleapis.com/v1beta/openai"
NAMED = {"Claude Code (uses your Claude Code login)": "claude", "Codex (OpenAI's command-line agent)": "codex",
         "Gemini CLI (Google's command-line agent, with your own API key only)": "gemini",
         "Cursor (Cursor's command-line agent)": "cursor", "Ollama (free, runs on this computer)": "http://localhost:11434/v1",
         "LM Studio (free, runs on this computer)": "http://localhost:1234/v1", "OpenAI (needs an API key)": "https://api.openai.com/v1",
         "xAI Grok (needs an API key)": "https://api.x.ai/v1", "OpenRouter (needs an API key)": "https://openrouter.ai/api/v1",
         "Google Gemini API (needs an API key)": GEMINI_API}
WEB, COMMAND = "Other web address (any OpenAI-compatible service)", "Custom command (any command-line AI program)"
READ_ONLY = " It runs read-only: it can answer and read files, but not change them or run commands."
HINTS = {"claude": "Nothing to fill in. Uses the Claude Code installed on this computer and its login.",
         "codex": "Nothing to fill in. Uses the Codex command installed on this computer and its login." + READ_ONLY +
                  " OpenAI's terms apply; for automation it recommends an API key: learn.chatgpt.com/docs/auth",
         "gemini": "Only with your own Gemini API key, in the GEMINI_API_KEY environment variable. Do not use it with your "
                   "personal Google sign-in: Google's terms forbid using that through other programs and it can suspend the "
                   "account (geminicli.com/docs/resources/tos-privacy)." + READ_ONLY,
         "cursor": "Nothing to fill in. Uses Cursor's agent command installed on this computer and its login, in Ask mode, "
                   "which Cursor documents as read-only. Cursor's terms apply; for automation it recommends an API key (CURSOR_API_KEY): "
                   "cursor.com/docs/cli/reference/authentication",
         GEMINI_API: "Needs an API key from aistudio.google.com, free or paid per use. Google's OpenAI-compatible Gemini API.",
         "http://localhost:11434/v1": "Free. Needs Ollama running on this computer. Pick a model you have downloaded.",
         "http://localhost:1234/v1": "Free. Needs LM Studio's local server running on this computer. Pick a model you have loaded.",
         "https://api.openai.com/v1": "Needs an API key, paid per use. A ChatGPT subscription does not include one.",
         "https://api.x.ai/v1": "Needs an API key from x.ai, paid per use.",
         "https://openrouter.ai/api/v1": "Needs an API key from openrouter.ai, paid per use. It offers models from many makers.",
         WEB: "Type the address of the service, usually ending in /v1. It needs an API key only if the service asks for one.",
         COMMAND: "Type the command that starts the program. It gets each request on its own and prints the answer."}
CLAUDE_MODELS = {"Sonnet: balanced": "sonnet", "Haiku: fastest, uses least": "haiku", "Opus: strongest, slowest, uses most": "opus"}
# Pocket TTS's built-in English voices. Left out: cosette and jean, whose recordings are licensed for non-commercial use only.
VOICES = {"Michael (default)": "michael", **{name.replace("_", " ").title(): name for name in (
    "alba", "anna", "azelma", "bill_boerst", "caro_davy", "charles", "eponine", "eve", "fantine", "george", "jane", "javert",
    "marius", "mary", "paul", "peter_yearsley", "stuart_bell", "vera")}}
CHATS = {"Sikio's own chat (continues across restarts)": "own", "Continue my latest Claude Code session": "latest"}
STT_MODELS = {"Fast (English)": "base.en", "More accurate, slower (English)": "small.en", "Other languages": "small"}


def label_of(table, value):
    """The friendly name of a saved value; a value the table does not know is shown as it is."""
    return next((name for name, code in table.items() if code == value), value)


def brain_choice(brain):
    return next((name for name, code in NAMED.items() if code == brain), WEB if brain.startswith(("http://", "https://")) else COMMAND)


def list_models(url, key):
    """The model names an OpenAI-compatible service offers (GET <address>/models). Raises OSError or ValueError on any failure."""
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    with PATIENT.open(urllib.request.Request(url.rstrip("/") + "/models", headers=headers), timeout=5) as response:
        try:
            return sorted(model["id"] for model in json.load(response)["data"])
        except (KeyError, TypeError) as error:
            raise ValueError("not a model list") from error


def settings_window():
    """The settings window, not yet shown. Returns it, its fields and the Save action, so a test can fill and save it without clicking."""
    import tkinter as tk  # here, so the voice app still runs on a Python built without tkinter
    from tkinter import ttk
    if os.name == "nt":
        try:
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(1)  # otherwise Tk is blurry on high-DPI screens
        except Exception:  # already set, or an old Windows
            pass
    root = tk.Tk()
    root.title("Sikio settings")
    now = {**DEFAULTS, **load_settings()}
    choice = brain_choice(now["brain"])
    fields = {"choice": tk.StringVar(root, choice), "where": tk.StringVar(root, now["brain"] if choice in (WEB, COMMAND) else ""),
              "model": tk.StringVar(root), "confirm": tk.BooleanVar(root, now["confirm"]),
              "wake": tk.StringVar(root, now["wake"]), "voice": tk.StringVar(root, label_of(VOICES, now["voice"] if now["voice"] in VOICES.values() else DEFAULTS["voice"])),
              "stt_model": tk.StringVar(root, label_of(STT_MODELS, now["stt_model"])), "chat": tk.StringVar(root, label_of(CHATS, now["chat"])), "key": tk.StringVar(root), "clear_key": tk.BooleanVar(root),
              "status": tk.StringVar(root), "hint": tk.StringVar(root), "model_hint": tk.StringVar(root), "key_note": tk.StringVar(root)}
    status = fields["status"]

    def kind(picked=None):
        code = NAMED.get(picked or fields["choice"].get(), "")
        return "claude" if code == "claude" else "web" if (picked or fields["choice"].get()) == WEB or code.startswith("http") else "command"

    def address():
        """The web address in use, or "" (no web brain, or none typed yet)."""
        url = NAMED.get(fields["choice"].get()) or fields["where"].get().strip().rstrip("/")
        return url if kind() == "web" and url.startswith(("http://", "https://")) else ""

    def default_model(picked):
        return label_of(CLAUDE_MODELS, "sonnet") if kind(picked) == "claude" else ""

    saved_model = now["model"] or ""
    fields["model"].set(label_of(CLAUDE_MODELS, saved_model or "sonnet") if kind() == "claude" else saved_model if kind() == "web" else "")
    form = ttk.Frame(root, padding=10)
    form.grid()
    rows = {}

    def add(name, label, widget, hint=None):
        row = form.grid_size()[1]
        text = ttk.Label(form, text=label)
        text.grid(row=row, column=0, sticky="nw", padx=(0, 10), pady=(6, 0))
        widget.grid(row=row, column=1, sticky="w", pady=(6, 0))
        rows[name] = [text, widget]
        if hint is not None:
            ttk.Label(form, textvariable=hint, wraplength=640, justify="left").grid(row=row + 1, column=1, sticky="w")

    models = ttk.Frame(form)
    model_box = ttk.Combobox(models, textvariable=fields["model"], width=34)
    model_box.pack(side="left")
    find_button = ttk.Button(models, text="Find models")
    confirm = ttk.Frame(form)
    ttk.Checkbutton(confirm, text="Ask for a spoken yes before Claude acts", variable=fields["confirm"]).pack(anchor="w")
    ttk.Label(confirm, text="Speech recognition mishears: without it a misheard request is carried out.", wraplength=640).pack(anchor="w")
    key = ttk.Frame(form)
    ttk.Entry(key, textvariable=fields["key"], show="*", width=30).grid(row=0, column=0)
    ttk.Checkbutton(key, text="Clear it", variable=fields["clear_key"]).grid(row=0, column=1, padx=6)
    ttk.Label(key, textvariable=fields["key_note"]).grid(row=1, column=0, columnspan=2, sticky="w")
    add("choice", "AI to use", ttk.Combobox(form, textvariable=fields["choice"], values=[*NAMED, WEB, COMMAND], state="readonly", width=52), fields["hint"])
    add("where", "Web address", ttk.Entry(form, textvariable=fields["where"], width=55))
    add("model", "Model", models, fields["model_hint"])
    add("chat", "Chat", ttk.Combobox(form, textvariable=fields["chat"], values=list(CHATS), state="readonly", width=52))
    add("confirm", "Confirmation", confirm)
    fields["wake_hint"] = tk.StringVar(root, "The name you say to get its attention.")  # kept: a variable nobody refers to is dropped with its text
    add("wake", "Wake word", ttk.Entry(form, textvariable=fields["wake"], width=30), fields["wake_hint"])
    add("voice", "Voice", ttk.Combobox(form, textvariable=fields["voice"], values=list(VOICES), state="readonly", width=34))
    add("stt_model", "Speech recognition", ttk.Combobox(form, textvariable=fields["stt_model"], values=list(STT_MODELS), state="readonly", width=34))
    add("key", "API key", key)
    ttk.Label(form, textvariable=status, wraplength=640).grid(row=form.grid_size()[1], column=0, columnspan=2, sticky="w", pady=(8, 0))

    def show(name, on):
        for widget in rows[name]:
            widget.grid() if on else widget.grid_remove()

    def refresh():
        picked, what = fields["choice"].get(), kind()
        fields["hint"].set(HINTS.get(NAMED.get(picked, picked), ""))
        rows["where"][0].configure(text="Web address" if picked == WEB else "Command")
        show("where", picked in (WEB, COMMAND))
        show("key", what == "web")
        show("chat", what == "claude")
        model_box.configure(state="disabled" if what == "command" else "normal", values=list(CLAUDE_MODELS) if what == "claude" else [])
        find_button.pack_forget()
        if what == "web":
            find_button.pack(side="left", padx=6)
        fields["model_hint"].set({"claude": "Blank means Sonnet.", "web": "Press Find models, or type the name. A model is required.",
                                  "command": "Chosen in that program's own settings, not here."}[what])
        url = address()
        fields["key_note"].set("" if not url else "A key is saved for this address." if stored_key(url) else "No key saved.")

    def switch(*_):
        """Another brain starts with empty fields, so nothing typed for one leaks into another."""
        fields["where"].set("")
        fields["model"].set(default_model(fields["choice"].get()))
        refresh()

    def find():
        url = address()
        if not url:
            status.set("Type the web address first.")
            return
        picked = fields["choice"].get()
        name = url if picked == WEB else picked.split(" (")[0]
        status.set("Looking for models...")
        root.update_idletasks()
        try:
            # shortcut: blocks the window for up to 5 s while the service answers, instead of using a thread
            found = list_models(url, fields["key"].get().strip() or stored_key(url))
        except Exception as error:  # whatever went wrong is one line here, never a traceback
            if isinstance(error, urllib.error.HTTPError):
                status.set(f"{name}: {web_error(error)}")
            elif isinstance(error, OSError):
                status.set(f"Could not reach {name}. " + ("Is it running?" if url.startswith(("http://localhost", "http://127.")) else "Check the address and your internet."))
            else:
                status.set(f"{name} did not answer with a list of models.")
            return
        model_box.configure(values=found)
        status.set(f"Found {len(found)} models. Pick one." if found else f"{name} has no models yet.")

    find_button.configure(command=find)
    fields["rows"], fields["model_box"], fields["find"] = rows, model_box, find
    fields["where"].trace_add("write", lambda *_: refresh())
    fields["choice"].trace_add("write", switch)
    refresh()

    def save():
        picked, where = fields["choice"].get(), fields["where"].get().strip().rstrip("/")
        what, brain, new_key = kind(), NAMED.get(picked) or where, fields["key"].get().strip()
        model = fields["model"].get().strip()
        if what == "claude":
            model = CLAUDE_MODELS.get(model, model)
            model = "" if model == "sonnet" else model  # the default is not written
        if what == "web" and not brain.startswith(("http://", "https://")) or picked == COMMAND and not brain:
            status.set("Type the web address (starting with http)." if what == "web" else "Type the command.")
            return
        if what == "web" and not model:
            status.set("A web address needs a model name. Press Find models, or type one such as gemma3:4b.")
            return
        try:
            if what == "web" and new_key:
                keyring.set_password(KEY_SERVICE, brain, new_key)
            elif what == "web" and fields["clear_key"].get() and stored_key(brain):
                keyring.delete_password(KEY_SERVICE, brain)
        except KeyringError as error:  # no credential store on this machine
            status.set(f"Could not save the key: {error}")
            return
        voice, stt = fields["voice"].get().strip(), fields["stt_model"].get().strip()
        values = {"brain": brain, "model": "" if what == "command" else model, "confirm": fields["confirm"].get(),
                  "wake": fields["wake"].get().strip(), "voice": VOICES.get(voice, voice), "stt_model": STT_MODELS.get(stt, stt),
                  "chat": "latest" if what == "claude" and CHATS.get(fields["chat"].get()) == "latest" else ""}  # "own" is the default: not written
        save_settings({k: v for k, v in values.items() if v != ""})  # a blank field means the default
        root.destroy()

    buttons = ttk.Frame(form)
    buttons.grid(row=form.grid_size()[1], column=0, columnspan=2, sticky="e", pady=(8, 0))
    ttk.Button(buttons, text="Save", command=save).pack(side="left", padx=4)
    ttk.Button(buttons, text="Cancel", command=root.destroy).pack(side="left")
    return root, fields, save


def edit_settings():
    settings_window()[0].mainloop()


class Log:
    """The tray app's stdout and stderr: what the terminal would show, with the time, in a file of about `limit` bytes;
    the file before it is kept as <name>.1."""

    def __init__(self, path, limit=1_000_000):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path, self.limit, self.lock, self.fresh = path, limit, threading.Lock(), True
        self.file = open(path, "a", encoding="utf-8")

    def write(self, text):
        with self.lock:
            if self.fresh and self.file.tell() > self.limit:  # between lines, never inside one
                self.file.close()
                try:
                    os.replace(self.path, f"{self.path}.1")
                except OSError:  # the other copy is open or locked: keep writing to this file
                    pass
                self.file = open(self.path, "a", encoding="utf-8")
            for part in text.splitlines(keepends=True):
                self.file.write(time.strftime("%Y-%m-%d %H:%M:%S ") + part if self.fresh else part)
                self.fresh = part.endswith("\n")
            self.file.flush()
        return len(text)

    def flush(self):
        pass

    def isatty(self):
        return False


def single_instance():
    """A lock held while this process listens, or None if another Sikio already does: both would answer.
    The lock goes with the process, however it ends."""
    path = SETTINGS.parent / "running.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    file = open(path, "a+")
    try:
        if os.name == "nt":
            import msvcrt
            file.seek(0)
            msvcrt.locking(file.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return file
    except OSError:
        file.close()
        return None


RUNNING = "Sikio is already running: look for its icon in the notification area, or the terminal it runs in. Quit that one first."
RUN_KEY = r"Software\Microsoft\Windows\CurrentVersion\Run"  # this user's logins only, needs no admin rights


def autostart_command():
    """What Windows runs at login: this file's tray app, with pythonw so no console window opens."""
    return f'"{Path(sys.executable).with_name("pythonw.exe")}" "{Path(__file__).resolve()}" tray'


def autostart(on=None):
    """Whether the tray app starts at login. True or False first adds or removes the entry."""
    import winreg
    with winreg.CreateKey(winreg.HKEY_CURRENT_USER, RUN_KEY) as key:
        if on:
            winreg.SetValueEx(key, "Sikio", 0, winreg.REG_SZ, autostart_command())
        elif on is False:
            try:
                winreg.DeleteValue(key, "Sikio")
            except FileNotFoundError:
                pass
        try:
            winreg.QueryValueEx(key, "Sikio")
            return True
        except FileNotFoundError:
            return False


# The tray icon's colour and tooltip per state. The icon is drawn in code, so there is no image file to ship.
LOOKS = {"starting": ("#9e9e9e", "starting"), "listening": ("#2e7d32", "listening"), "thinking": ("#ef8f00", "heard you, thinking"),
         "speaking": ("#1565c0", "speaking"), "confirming": ("#8e24aa", "waiting for your yes"),
         "paused": ("#616161", "paused, not listening"), "error": ("#c62828", "stopped by an error")}


def tray_image(state):
    """A disc in the state's colour, with a mark so the states differ in shape too."""
    from PIL import Image, ImageDraw, ImageFont
    image = Image.new("RGBA", (64, 64))
    draw, colour = ImageDraw.Draw(image), LOOKS[state][0]
    if state == "starting":
        draw.ellipse((6, 6, 58, 58), outline=colour, width=9)
        return image
    draw.ellipse((2, 2, 62, 62), fill=colour)
    if state == "paused":
        draw.rectangle((20, 17, 28, 47), fill="white")
        draw.rectangle((36, 17, 44, 47), fill="white")
    elif state == "thinking":
        for x in (18, 32, 46):
            draw.ellipse((x - 5, 27, x + 5, 37), fill="white")
    elif state == "speaking":
        for x, h in ((20, 8), (32, 16), (44, 8)):
            draw.rectangle((x - 4, 32 - h, x + 4, 32 + h), fill="white")
    elif state in ("confirming", "error"):
        draw.text((32, 33), "?" if state == "confirming" else "!", fill="white", anchor="mm", font=ImageFont.load_default(44))
    return image


def tray_title(state, line, wake):
    """The tooltip; Windows shows at most 127 characters."""
    title = f'Sikio: listening for "{wake}"' if state == "listening" else f"Sikio: {LOOKS[state][1]}"
    return (f"{title}\n{line}" if line and state in ("starting", "error") else title)[:127]


def alert(text):
    """A message box, for the tray app before its icon is up: it has no terminal to print to."""
    print(text)
    if os.name == "nt":
        import ctypes
        ctypes.windll.user32.MessageBoxW(None, text, "Sikio", 0x40)
        return
    try:
        import tkinter as tk
        from tkinter import messagebox
        root = tk.Tk()
        root.withdraw()
        messagebox.showinfo("Sikio", text, parent=root)
        root.destroy()
    except Exception:  # no tkinter or no display: the line is in the log
        pass


class Tray:
    """The tray app: an icon in the notification area whose menu controls the assistant, run() in a thread of its own.

    Threads: pystray's message loop owns the main thread (macOS requires that; on Windows any thread would do) and runs
    the menu actions, which must return quickly. run() has a thread and an asyncio loop of its own, which the menu reaches
    only through thread-safe means: the QUIT, PAUSED and NEW_CHAT events, and call_soon_threadsafe to cancel run().
    The settings window opens in a thread of its own, one window at a time: Tk works in any one thread, and the menu's
    thread must keep handling messages meanwhile."""

    def __init__(self):
        import pystray
        self.state, self.line, self.wake, self.loop, self.task, self.thread = "starting", "", DEFAULTS["wake"], None, None, None
        self.stopping = self.quitting = False
        self.ended = threading.Event()  # run() has cleaned up: the voice is stopped, the brain disconnected
        self.window = threading.Lock()
        item = pystray.MenuItem
        self.icon = pystray.Icon("Sikio", tray_image("starting"), "Sikio: starting", pystray.Menu(
            item(lambda _: "Resume listening" if PAUSED.is_set() else "Pause listening", self.pause),
            item("New chat", self.new_chat),
            item("Settings", self.settings),
            item("Open log", self.open_log),
            item("Start with Windows", self.toggle_autostart, checked=lambda _: autostart(), visible=os.name == "nt"),
            item("Quit", self.quit)))

    def show(self, state, line=""):
        """on_state while the tray app runs: the icon and tooltip for a state, or for Pause over it. From any thread."""
        self.state, self.line = state, line
        shown = "paused" if PAUSED.is_set() and state != "error" else state
        self.icon.icon = tray_image(shown)
        self.icon.title = tray_title(shown, line, self.wake)

    def start(self):
        args = argparse.Namespace(**merged({}, load_settings()), text=None)
        self.wake, self.stopping, self.loop, self.task = args.wake, False, None, None
        self.ended.clear()
        QUIT.clear()
        self.thread = threading.Thread(target=self.assist, args=(args,), daemon=True)
        self.thread.start()

    def assist(self, args):
        async def main():
            self.loop, self.task = asyncio.get_running_loop(), asyncio.current_task()
            try:
                if not self.stopping:  # Quit before this thread got going
                    await run(args)
            finally:
                self.ended.set()
        try:
            asyncio.run(main())
        except BaseException as error:  # never a silent death: a sys.exit() message or a bug goes to the log and the icon
            if self.stopping:
                return
            message = str(error) if isinstance(error, SystemExit) else f"{type(error).__name__}: {error}"
            print(message) if isinstance(error, SystemExit) else traceback.print_exc()
            self.show("error", message)
            self.icon.notify(message[-250:], "Sikio stopped")

    def stop(self, wait=None):
        """Stop run(): QUIT ends its listening and talking threads, the cancel its loop. Returns once it has cleaned up."""
        self.stopping = True
        QUIT.set()
        if self.thread and self.thread.is_alive():
            try:
                self.loop.call_soon_threadsafe(self.task.cancel)
            except (AttributeError, RuntimeError):  # its loop has not started yet, or is closed already
                pass
            self.ended.wait(wait)

    def restart(self):
        self.show("starting", "Restarting with the new settings...")
        self.stop()
        if self.thread:
            self.thread.join()  # also its threads still loading, so nothing of the old run is left
        if not self.quitting:  # shortcut: a Quit in the instant between this check and start() is missed
            self.start()

    def pause(self, *_):
        PAUSED.clear() if PAUSED.is_set() else PAUSED.set()
        print("Paused: not listening." if PAUSED.is_set() else "Listening again.", flush=True)
        self.show(self.state, self.line)
        self.icon.update_menu()

    def new_chat(self, *_):
        NEW_CHAT.set()

    def settings(self, *_):
        if self.window.acquire(blocking=False):  # shortcut: a second click does not bring the open window to the front
            threading.Thread(target=self.edit, daemon=True).start()

    def edit(self):
        """The settings window; after Save run() restarts with the new settings (in this process, the voice engine stays loaded)."""
        def stamp():
            return SETTINGS.stat().st_mtime_ns if SETTINGS.exists() else None
        saved = stamp()
        try:
            if sys.platform == "darwin":  # Tk works only in a main thread there, and pystray's loop owns this process's
                subprocess.run([sys.executable, str(Path(__file__).resolve()), "settings"])
            else:
                edit_settings()
        except Exception:
            traceback.print_exc()
        finally:
            gc.collect()  # Tk's objects must be freed in the thread that made them, or Tcl aborts the process
            self.window.release()
        if stamp() != saved:
            self.restart()

    def open_log(self, *_):
        log = SETTINGS.parent / "sikio.log"
        if os.name == "nt":
            os.startfile(log)
        else:
            subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open", log])

    def toggle_autostart(self, *_):
        autostart(not autostart())
        self.icon.update_menu()

    def quit(self, *_):
        self.quitting = True
        print("Quitting.", flush=True)
        self.stop(wait=2)
        self.icon.stop()


def tray():
    """sikio-tray: the assistant in the background, with an icon in the notification area and no console."""
    global on_state
    if os.name == "nt" and sys.stdout is None:
        # Started with pythonw, which has no console: then every console program it starts (the claude CLI, a command
        # brain) opens a terminal window of its own, measured. A console without a window is inherited instead.
        subprocess.Popen([Path(sys.executable).with_name("python.exe"), Path(__file__).resolve(), "tray"],
                         creationflags=subprocess.CREATE_NO_WINDOW)
        return
    lock = single_instance()  # held until this function returns
    if not lock:
        return alert(RUNNING)
    sys.stdout = sys.stderr = Log(SETTINGS.parent / "sikio.log")  # without a console print() would drop everything
    os.chdir(Path.home())  # started at login it would be C:\Windows\System32, and the agent works in this folder
    if not SETTINGS.exists():  # first run: settings first, as in the terminal
        try:
            edit_settings()
        except Exception:
            traceback.print_exc()
        gc.collect()
    try:
        app = Tray()
    except ImportError as error:
        return alert(f"The tray app needs pystray and Pillow ({error}): pip install pystray pillow")
    on_state = app.show

    def crashed(kind, error, trace):
        traceback.print_exception(kind, error, trace)
        app.show("error", f"{kind.__name__}: {error}")
    sys.excepthook = crashed
    threading.excepthook = lambda a: crashed(a.exc_type, a.exc_value, a.exc_traceback)
    print(f"Sikio tray app started in {Path.cwd()}.", flush=True)

    def setup(icon):
        icon.visible = True
        app.start()
    app.icon.run(setup)
    if app.thread:
        app.thread.join(1)
    if app.thread and app.thread.is_alive():  # quit while loading: that thread cannot be stopped, and Python would wait for it
        print("Quit while loading: not waiting for it.", flush=True)
        os._exit(0)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", nargs="?", choices=["settings", "tray"], help="settings: open the settings window; "
                        "tray: run in the background with an icon in the notification area, as sikio-tray does")
    # Options default to None so a saved setting can fill in what the command line leaves out.
    parser.add_argument("--brain", help='claude (the default), codex, gemini, cursor, the URL of an OpenAI-compatible server such as http://localhost:11434/v1, '
                                        'or a command with its options such as "ollama run llama3.2"')
    parser.add_argument("--model", help="model name: sonnet (the default), haiku or opus for claude; required for a URL brain")
    parser.add_argument("--confirm", action=argparse.BooleanOptionalAction, help="--no-confirm lets Claude act without asking for a spoken yes first; --confirm asks even if the settings say not to")
    parser.add_argument("--wake", help=f"word that must start a request (default {DEFAULTS['wake']})")
    parser.add_argument("--voice", help=f"voice (default michael): {', '.join(VOICES.values())}")
    parser.add_argument("--stt-model", help='Whisper model (default base.en): "small.en" is more accurate but 3x slower, "small" understands other languages')
    parser.add_argument("--chat", choices=["own", "latest"], help="claude brain: own (default) continues Sikio's own chat; latest branches off your most recent Claude Code session")
    parser.add_argument("--text", help="skip the microphone: answer this once and exit; confirmations are typed")
    args = parser.parse_args()
    if args.command == "settings":
        return edit_settings()
    if args.command == "tray":
        return tray()
    if len(sys.argv) == 1 and not SETTINGS.exists():  # first run: settings first, then listen; after Cancel the defaults apply
        try:
            edit_settings()
        except Exception as error:  # no display or no tkinter, as over SSH: the defaults still work
            print(f"Could not open the settings window ({error}), using the defaults.", file=sys.stderr)
    lock = args.text or single_instance()  # held until this function returns; a typed request does not listen
    if not lock:
        sys.exit(RUNNING)
    vars(args).update(merged(vars(args), load_settings()))
    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:  # here, not under __main__, so the installed sikio command quits quietly too
        pass


if __name__ == "__main__":
    main()
