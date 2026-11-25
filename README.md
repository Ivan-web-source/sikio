# Sikio

Sikio (see-KEE-oh, Swahili for "ear").

Talk to the AI command-line tool you already pay for. You say the wake word and a request, Sikio answers out loud.

Sikio is an independent project, not affiliated with or endorsed by Anthropic, OpenAI or any other provider.

- **Ears:** Whisper (`faster-whisper`), runs on your computer, free.
- **Voice:** Pocket TTS by Kyutai (`pocket-tts`), runs on your computer, free. The default voice is Michael. It starts speaking as soon as the first moment of a sentence is made (0.1 to 0.4 s on the test laptop), while it makes the rest.
- **Brain:** by default your own Claude Code, through Anthropic's Claude Agent SDK and the `claude` CLI you are logged in to. Sikio never sees your login. Other AI CLIs work too, see below.

## Install

Needs Python 3.10 to 3.14, a microphone, speakers or headphones, and [Claude Code](https://claude.com/claude-code) installed and logged in (or another brain).

On Windows, paste this into PowerShell. It needs no Python and no admin rights: it installs [uv](https://docs.astral.sh/uv/) if missing, which fetches its own Python, then installs Sikio and adds a **Sikio** entry to the Start menu:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://raw.githubusercontent.com/Ivan-web-source/sikio/main/install.ps1 | iex"
```

Or download **Sikio-Setup.exe** from [GitHub Releases](https://github.com/Ivan-web-source/sikio/releases) and run it (no admin rights; uninstall from Windows Settings > Apps). It is not code-signed, so Windows shows a SmartScreen **Windows protected your PC** warning: click **More info**, then **Run anyway**. It downloads about 1.5 GB while installing (Python and PyTorch), in a console window that shows the progress.

Run it again to upgrade. To uninstall, run the same line with `uninstall.ps1` in place of `install.ps1`. That removes the Start menu entry, the **Start with Windows** entry and the program, and keeps your settings (in `%APPDATA%\sikio`).

On other platforms: `uv tool install sikio` or `pipx install sikio`. From a copy of this repo:

```bash
pipx install .
```

`pip install .` works too. All of these install the `sikio` command, with PyTorch for the voice: about 1.5 GB in all on Windows, where the PyTorch from PyPI is the CPU-only build. On Linux PyPI gives the much larger CUDA build; `pip install . --extra-index-url https://download.pytorch.org/whl/cpu` installs the CPU-only one, which is all the voice needs.

On macOS and Linux (untested, see [Help wanted](#help-wanted-untested-platforms-and-services)):

- **Linux** needs the PortAudio system library for sound: `sudo apt install libportaudio2` (Debian, Ubuntu) or your system's `portaudio` package. Without it Sikio says so and exits; the macOS and Windows packages include it.
- **Intel Macs cannot install it from PyPI**: the voice needs PyTorch 2.5 or newer, and PyTorch's last build for Intel Macs is 2.2.2. Apple Silicon Macs are fine.
- The settings window needs tkinter, which some Pythons (Homebrew's, many Linux ones) leave out: install `python-tk` (Homebrew) or `python3-tk` (Debian, Ubuntu). Without it everything else still works, with the settings file or command-line options.

The first run downloads about 375 MB of speech models into the Hugging Face cache (`~/.cache/huggingface`, or `HF_HOME`): the voice (about 227 MB, from `kyutai/pocket-tts-without-voice-cloning`) and Whisper (about 148 MB). It needs no Hugging Face account or token, and a line says when the voice is downloading. After that Sikio starts without the internet: both models are loaded from the cache first.

Starting takes a while, and **on every start**, not just the first: loading the voice engine (PyTorch) took about 15 s on a calm machine and up to a minute or more on a busy laptop (the test laptop, an 11th-generation Core i7). The very first start also downloads the models above, about 230 MB for the voice plus the speech recognition model. A line says when each part starts and is done (the voice engine, the connection to Claude Code, speech recognition), and the tray icon's tooltip shows the same. The first request does not wait longer for it.

So the best way to use it is the [tray app](#in-the-background-without-a-terminal-windows) with **Start with Windows** ticked: it loads once when you log in and then stays ready. Changing its settings restarts the assistant without loading the voice engine again.

## Run

```bash
sikio
```

The very first time, with no options, the settings window opens first; after Save or Cancel it starts listening. Say "Juniper, what is the capital of France?". Ctrl+C quits.

While the speech models load, Sikio already starts Claude Code in the background, so the first request does not wait for it. This sends nothing to Claude and uses none of your limits. If Claude Code cannot start (no network, not logged in), the reason is printed and it tries again at your first request.

When a reply ends with a question, such as "What should the email say?", just answer: for about 8 seconds after the reply, what you say next counts as your request without the wake word. After that answer, or if you say nothing for 8 seconds, the wake word is needed again. Coughs and other noise in that time are ignored, not sent. Saying only "Juniper" answers "Yes?" and takes the next thing you say as the request, if it comes within about 8 seconds.

You can interrupt it while it answers or works, as in a chat: say the wake word. "Juniper" or "Juniper, stop" just stops it; "Juniper, what about tomorrow?" stops it and asks that instead, in the same chat. It goes quiet as soon as it has heard the wake word, which it checks after the first moment of what you say, so usually before you have finished speaking (measured with "Jarvis, stop." spoken over a reply by four of the voices: silent about 0.1 s before the end of the words, instead of 0.8 s after them). Claude Code's turn is interrupted and what it had said so far stays in the chat. Speech without the wake word, and noise, do not stop it. Interrupting while it asks "Shall I go ahead?" counts as no, so nothing is changed. With a web-address brain the unfinished answer is dropped; with a command brain the command is ended (a command started through a `.cmd` file on Windows, such as npm's `codex.cmd`, ends only that file: the program it started finishes in the background, read-only for the presets below, and its answer is dropped).

The agent works in the folder you start Sikio from: that is where it reads and writes files, and that folder's Claude Code settings apply.

| Option | Default | Meaning |
|---|---|---|
| `--brain` | `claude` | `claude`, the URL of an OpenAI-compatible server, `codex`, `gemini`, `cursor`, or any command that reads a prompt on stdin, with its options in the same string |
| `--model` | `sonnet` for Claude | the model: `haiku`, `sonnet` or `opus` for Claude; required for a URL brain |
| `--no-confirm` | off | let Claude act without asking for a spoken yes, see below; `--confirm` asks even if the settings say not to |
| `--wake` | `juniper` | the word that must start a request; any word works, a saved one keeps working. Juniper was picked by measurement over eight everyday words and names: it was heard most reliably among the rarest in ordinary speech |
| `--voice` | `michael` | one of Pocket TTS's built-in English voices: `michael`, `alba`, `anna`, `azelma`, `bill_boerst`, `caro_davy`, `charles`, `eponine`, `eve`, `fantine`, `george`, `jane`, `javert`, `marius`, `mary`, `paul`, `peter_yearsley`, `stuart_bell`, `vera`. Any other name is replaced by `michael`, with a line saying so |
| `--stt-model` | `base.en` | `small.en` is more accurate and slower, `small` understands other languages |
| `--text "..."` | | answer one typed request and exit, to test without a microphone; a confirmation is typed too |

## In the background, without a terminal (Windows)

```bash
sikio-tray
```

starts the same assistant with no console window and an icon in the notification area (next to the clock; Windows may first put it under the ^ arrow). It uses the settings file, not command-line options, and on the very first run the settings window opens first. You can also start it from the Start menu's Run box, or with `sikio tray`. The tray app is tested on Windows only.

The icon shows what it is doing, and its tooltip says it in words: a grey ring while starting (the tooltip says what is still loading), green while listening, orange with dots once it heard you and is thinking, blue with bars while speaking, purple with a question mark while it waits for your spoken yes, grey with two bars while paused, red with an exclamation mark when it stopped because of an error (a notification says why, for example that Claude Code is not installed or there is no sound output).

Right-click the icon for its menu:

- **Pause listening / Resume listening**: while paused, the microphone's sound is thrown away and nothing is transcribed. A reply already being spoken still finishes, but you cannot interrupt it by voice, and a "Shall I go ahead?" waiting for your yes gets no answer, which is a no.
- **New chat**: the same as saying "new chat". During a reply it happens after the reply.
- **Settings**: the settings window. After Save the assistant restarts at once with the new settings, inside the same program, so the voice engine does not have to load again; this takes a few seconds.
- **Open log**: what the terminal would show (what it heard, its replies, the usage line, errors), with the time, in `%APPDATA%\sikio\sikio.log`. Past about 1 MB it starts a new file and keeps the one before as `sikio.log.1`.
- **Start with Windows**: ticked, Sikio starts in the tray when you log in. It is a per-user entry named `Sikio` under `HKEY_CURRENT_USER\Software\Microsoft\Windows\CurrentVersion\Run` (no admin rights; other users of the computer are not affected). Untick it to remove the entry, and untick it before uninstalling Sikio. It can also be removed in Task Manager's Startup apps, or by deleting that registry value.
- **Quit**: stops the voice, disconnects Claude Code and exits within about 2 seconds, also while it is still starting. A pending "Shall I go ahead?" counts as no.

On macOS and Linux the tray app is untested. It needs `pip install pystray pillow` there (they are installed on Windows only). On macOS the settings window then opens as a program of its own, because the menu owns the main thread. On Linux the icon needs a desktop with a tray: an AppIndicator one (install PyGObject and your system's AppIndicator library, for example `gir1.2-ayatanaappindicator3-0.1`), or one that still supports the older XEmbed tray icons; GNOME shows neither without an extension. Open log uses `open` on macOS and `xdg-open` on Linux, and Start with Windows is not shown there.

Only one Sikio listens at a time: starting a second one, from the tray or a terminal, says that one is already running and exits, so two never answer at once. (`--text` runs are not affected.)

The tray app works in your home folder: that is where Claude Code reads and writes files, and that folder's Claude Code settings apply.

## Settings

```bash
sikio settings
```

opens a small window. Each choice has a line under it saying what it needs.

- **Brain**: who does the thinking. Claude Code, Codex and Cursor need nothing typed: they use the program and login already on this computer. The Gemini CLI is only for use with your own Gemini API key (see below). Ollama and LM Studio are free and run on this computer. OpenAI, xAI Grok, OpenRouter and the Google Gemini API need an API key and are paid per use or have a free tier (a ChatGPT or Grok subscription does not include one). "Other web address" is any OpenAI-compatible service, and you type its address. "Custom command" is any command-line AI program, and you type the command.
- **Web address / Command**: only shown for those two brains.
- **Model**: for Claude Code a plain name (Sonnet is the default, Haiku is fastest and cheapest, Opus is strongest); for a web address the model's name, which "Find models" lists by asking the service (it waits up to 5 seconds). Codex, Gemini, Cursor and a custom command choose their model in their own settings, so the field is greyed out.
- **Confirmation**, **Wake word**, **Voice** (Michael or one of the other built-in voices) and **Speech recognition** (fast, more accurate, or other languages).
- **API key**: only for web addresses, see below.

They are saved in `%APPDATA%\sikio\settings.json` on Windows, `~/Library/Application Support/sikio/settings.json` on macOS and `~/.config/sikio/settings.json` on Linux (or under `$XDG_CONFIG_HOME` if set), a plain JSON file you can also edit or delete. The tray app's log is in the same folder.

An option on the command line wins over the settings file, and the settings file wins over the defaults in the table above. `--no-confirm` turns confirmation off for one run even if it is saved as on, and `--confirm` turns it on for one run even if it is saved as off.

The API key is kept in your system's credential store (Windows Credential Manager, macOS Keychain, or the Secret Service on Linux) under the name `sikio`, never in the settings file. It is saved for the web address you entered and only ever sent to that address. The window only shows whether a key is saved, never the key itself; type a new one to replace it or tick "Clear it" to remove it. The `SIKIO_API_KEY` environment variable, if set, wins over the saved key.

## Other models and agents

There are three kinds of brain.

- **Claude Code** (`--brain claude`, the default): an agent with your tools, skills and MCP servers, one ongoing chat, and the spoken yes before it acts. Everything below about saying yes and the ongoing chat is about this brain.
- **Any OpenAI-compatible chat service, by URL**: OpenAI, xAI Grok, OpenRouter, Google's Gemini API (`https://generativelanguage.googleapis.com/v1beta/openai`), LM Studio and Ollama all speak this protocol. Chat only, no tools and no actions. It remembers the conversation while Sikio runs, forgets it when you quit, and "new chat" clears it. The API key is the one saved in the settings window, or the `SIKIO_API_KEY` environment variable, and is only sent when there is one; local servers need none.

  ```bash
  sikio --brain http://localhost:11434/v1 --model gemma3:4b
  sikio --brain https://api.openai.com/v1 --model gpt-4o-mini
  ```

  When the service refuses, the printed line names the cause where its answer shows it: a refused API key, no credit left (or a spending limit), too many requests, or a model it does not know. The spoken apology stays short. An address that does not answer at all fails after 10 seconds; once connected it waits up to 5 minutes for the first word, as a local model may first have to load.

- **Any command-line agent** (`--brain "<command>"`): Sikio runs the command for each request, sends the request on stdin and speaks what it prints. Each request stands on its own. A brain with no subscription at all, using [Ollama](https://ollama.com):

  ```bash
  sikio --brain "ollama run llama3.1:8b"
  ```

  Three agents have presets. These agents can act on their own, and they do not get Sikio's spoken yes, so each preset uses the tool's own read-only mode: they can answer and read files, but not change files or run commands. The trade-off is that a task that needs an action is refused; use Claude Code for those. No preset passes a flag that skips the tool's own approvals.

  | `--brain` | Runs | Read-only through |
  |---|---|---|
  | `codex` | `codex exec --sandbox read-only --skip-git-repo-check -` (the request on stdin) | Codex's read-only sandbox; in `exec` mode it never stops to ask. `--skip-git-repo-check` lets it run outside a Git repository (the tray app works in your home folder), which is harmless when it can write nothing. See [non-interactive mode](https://learn.chatgpt.com/docs/non-interactive-mode) and [sandboxing](https://learn.chatgpt.com/docs/sandboxing); on Linux its sandbox needs `bubblewrap`. |
  | `gemini` | `gemini --approval-mode default` (the request on stdin, which runs it headless) | In headless mode every tool that would ask for approval is refused. Its read-only `plan` mode is not used: Gemini overrides it in folders you have not trusted. See [headless mode](https://geminicli.com/docs/cli/headless/). |
  | `cursor` | `agent --print --mode ask --output-format text <request>` | Ask mode, which Cursor documents as read-only. Cursor documents no way to read the prompt from stdin, so here the request is the last argument; Sikio refuses to do that through a `.bat` or `.cmd` file, where Windows could run parts of it as commands (Cursor's Windows installer also installs `agent.exe`, which is used). See [print mode](https://cursor.com/docs/cli/headless) and [parameters](https://cursor.com/docs/cli/reference/parameters). |

  If one of them is not installed Sikio says so and exits. If it fails, the end of what it printed is shown with its exit code, plus a hint when it looks like it is not logged in, and the voice apologises briefly. Each provider's own terms apply:

  - **Gemini CLI: only with your own Gemini API key** (the `GEMINI_API_KEY` environment variable, from [Google AI Studio](https://aistudio.google.com)). Do not use it with your personal Google sign-in: [Google's terms](https://geminicli.com/docs/resources/tos-privacy/) say that using the services behind Gemini CLI through other software violates them, and accounts can be suspended. Google also moved unpaid-tier and Google One users from Gemini CLI to another tool in June 2026. With a key, the Google Gemini API web address above works as well, without the CLI.
  - **Codex** and **Cursor**: their terms do not settle use from other programs either way; both recommend an API key for automation: [Codex authentication](https://learn.chatgpt.com/docs/auth), [Cursor CLI authentication](https://cursor.com/docs/cli/reference/authentication) (`CURSOR_API_KEY`).

Tested for real: Claude Code, and Ollama both by URL and as a command, on Windows. Everything else is only simulated, see [Help wanted](#help-wanted-untested-platforms-and-services).

## What it can do

With Claude it is your real Claude Code: the same tools, skills, MCP servers and settings (user, project and local) you use in the terminal. A question just gets answered. A task makes it use its tools, for example to look through files, run commands, edit files or call an MCP server, and then it tells you what it did.

## Saying yes

Speech recognition mishears, so before the agent changes anything it says what it understood and is about to do, and then asks, for example "I'd like to run a command: git status. Shall I go ahead?". It listens for about 10 seconds.

- Only a clear yes such as "yes", "yeah", "sure", "okay", "go ahead" or "do it" lets it act. "No", "not yet", "yes... no wait", an unclear answer or silence means no, and the agent is told you did not confirm.
- A yes covers the rest of that one request, so a task that writes three files asks once. Your next request asks again.
- It asks wherever Claude Code itself would ask you for permission: edits, new files, commands, MCP tools. Reading and searching do not ask.
- MCP tools always ask, also ones that only read, unless your settings allow them. Skills run without asking; whatever a skill then does to change things asks as usual.
- Anything your own Claude Code settings already allow (the `permissions.allow` rules in your settings files) runs without asking. Check those rules if you want every action confirmed.

`--no-confirm`, or unticking spoken confirmation in the settings window, turns the question off: Claude then acts as soon as it has understood you. Your Claude Code permission rules, including deny rules, still apply, and a line at startup reminds you it is off. Only use it where a misheard sentence cannot do harm: speech recognition does mishear, and with this option a misheard request is carried out without a chance to say no.

## One ongoing chat

Sikio keeps one Claude conversation, also after you quit and start it again: it remembers the last chat in `~/.cache/sikio/session` and resumes it. If that chat cannot be resumed it starts a new one and prints a line saying so.

Say "new chat", "new conversation", "start a new chat" or "start over" on its own to forget it and start fresh. It answers "Okay, new chat." without asking Claude.

To pick up where you left off in Claude Code instead, choose "Continue my latest Claude Code session" under Chat in the settings (or start with `--chat latest`). When it starts, Sikio finds your most recently used Claude Code session, works in that session's folder and branches off it, so the original stays as it was, even if it is still open in a terminal. From then on Sikio continues the branch as its own chat, and "new chat" still starts fresh. This is for the Claude brain only; if no session is found, Sikio's own chat is used. A long session is sent along with every request, so each one uses more of your Claude limits than in a short chat; say "new chat" to start small again.

## Cost

Every Claude request uses your subscription's limits. Each request carries Claude Code's full context (system prompt, tool list, your settings and the chat so far), about 30,000 tokens to begin with, most of it cached. Measured with Sonnet, as the API-priced equivalents Claude Code reports (a subscription is not billed per request, but this shows the relative size):

- The first question of a new chat: about 5 to 6 cents, because about 12,000 to 15,000 tokens are written to the cache.
- A light question in an existing chat: 1 to 5 cents, depending on how much of the cache has to be rebuilt.
- A request that uses a tool or two (listing files, writing a file, with the spoken yes): about 2 cents. Long tasks with many tool calls cost more.

The printed line after each turn shows the tokens, then the cost of that request and the running total for the chat, for example `$0.017 this request, $0.098 this chat`. On a subscription these are not charges: they are what the same use would cost with pay-per-use API pricing. For the first request after restarting and resuming the last chat the share of that request is not known, so only the total is printed. A long chat gets more expensive per request, so say "new chat" when you change topic. `--model haiku` is cheaper, `--model opus` much more expensive.

## Sounding human

- Replies are written for speech: the prompt in `STYLE` asks for short, relaxed, spoken-style sentences with no markdown.
- Words the voice mispronounces are fixed in the `SAY` dictionary in `sikio.py`. Add your own.
- Long sentences get a comma before a joining word such as "and" or "but", because the voice only pauses at punctuation. This is how it breathes; the printed reply is unchanged.
- The voice reads numbers, times and amounts ("3:30 pm", "$20") itself, and lifts its voice a little at a question mark.
- The voice has no emotion or direction tags. Tags like `[laughs]` are stripped, not acted out.
- Each sentence is made on its own and plays as it is made; the next sentence is made while the current one plays, so they follow each other without a gap.

## Limits

- Only `claude` can act and remembers across restarts. A URL brain remembers only while Sikio runs; command brains get each request on its own.
- A spoken yes can be misheard as well. A yes covers the whole request, so a misheard request plus a misheard yes can still do the wrong thing.
- The chat is resumed in whatever folder you start from, but the agent then works in that new folder.
- Background noise (clicks, fans, hum) is filtered by voice detection, but loud speech-like sound such as a TV or other people talking can still trigger it.
- If your microphone hears your speakers (many desktop setups do; most laptops cancel their own sound), Sikio hears its own voice while it talks. It only stops for words that start with the wake word and are not part of what it is saying, so this rarely stops it by mistake, but it costs some processor time while it talks. Headphones avoid it.
- The `claude`, `codex` and `cursor` brains use your subscription's rate limits. Check each provider's terms before building on this, and see above for Gemini.
- The voice speaks English only, and only its built-in voices: Sikio has no voice cloning.
- The voice is made on the processor while it speaks. On the test laptop it was mostly made about twice as fast as spoken; on battery, or with the processor fully busy, only about as fast, and then it sometimes pauses for about a third of a second mid-reply.

## Help wanted: untested platforms and services

Only Windows 11 with Claude Code and a local Ollama has really been used. Everything below was made as likely to work as possible without an account or the machine, and has never answered a real request. If you use one of them, please report what happens (an issue with the printed lines or the tray app's log, without your keys), or send a fix.

| What | What was simulated | Known risks |
|---|---|---|
| Codex CLI | The real `@openai/codex` 0.171.0 from npm, run without a login in a throwaway folder: its `exec --help`, its not-logged-in failure (exit 1, a `401 Unauthorized` on stderr, nothing on stdout) and its refusal outside a Git repository without `--skip-git-repo-check`, also through Sikio's own command brain. Tests with a fake `codex.cmd` on PATH (answer, not logged in, slow, crash). | Never seen answering, so the reply format (only the final message on stdout, as documented) is unconfirmed. Interrupting ends only npm's `codex.cmd` on Windows, not the agent it started. On Linux its sandbox needs `bubblewrap`. |
| Gemini CLI | The real `@google/gemini-cli` 0.70.0 from npm, without a login: its `--help`, its not-logged-in failure (exit 41, "Please set an Auth method" on stderr), a refused made-up API key, and its source code, which shows headless mode refusing tools that would ask. Tests with a fake `gemini.cmd`. | Never seen answering. Only for use with your own API key (see above). A later version could change what headless `default` mode allows. |
| Cursor CLI | Its documentation only: its installer is a shell script, which was not run. A test checks that the request arrives as the last argument and that a `.cmd` launcher is refused. | Never run at all. Whether Ask mode also blocks every command is not documented. A user reported its CLI exiting at once on prompts containing "-p" or "-P" (as in "release-PROD"). |
| OpenAI, xAI Grok, OpenRouter, LM Studio, Google Gemini API | A local mock server replaying each one's documented stream (OpenAI's role-only first chunk and usage chunk with no choices, OpenRouter's `: OPENROUTER PROCESSING` comments, reasoning fields and its error chunk mid-stream, xAI's `reasoning_content`), its `/models` list and its documented error bodies for a refused key, no credit, rate limits and an unknown model, through the real chat brain and Find models. The tests do the same. | Error bodies are copied from documentation and may differ in practice; then the cause is shown as a plain refusal, with the service's own message. A model that puts its reasoning inside the answer text (`<think>` tags) would have it read out. |
| Linux | The test suite in WSL (Ubuntu 22.04, Python 3.10) without PortAudio, tkinter, a credential store or a sound device: the tests that need them skip, the app says what to install. Code review of every platform branch, and tests that switch the platform branches. | Never heard speaking or listening on Linux. The tray icon depends on the desktop. |
| macOS | Code review, and tests that switch the platform branches, only. | Never run. Intel Macs cannot install it (see Install). The microphone permission prompt, and the settings window opened from the tray as a program of its own, are untested. |
| Tray app off Windows | Tests of the platform branches only (Open log, the hidden Start with Windows, a message box for errors before the icon is up). | Never shown on macOS or Linux. |

## Credits

- Voice: [Pocket TTS](https://github.com/kyutai-labs/pocket-tts) by Kyutai. The code is MIT licensed. The model weights ([kyutai/pocket-tts-without-voice-cloning](https://huggingface.co/kyutai/pocket-tts-without-voice-cloning)) are licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); Kyutai's model card also forbids uses such as impersonation, deception and unlawful content.
- The voices come from Kyutai's [tts-voices](https://huggingface.co/kyutai/tts-voices). Michael (the default), Anna, Vera, Fantine, Charles, Paul, Eponine, Azelma, George, Mary, Jane and Eve are speakers of the [VCTK corpus](https://datashare.ed.ac.uk/handle/10283/3443) (Centre for Speech Technology Research, University of Edinburgh), CC BY 4.0. Alba is voiced by Alba MacKenna, CC BY 4.0. Javert and Marius are from Kyutai's Unmute voice donations, and Bill Boerst, Peter Yearsley, Stuart Bell and Caro Davy from [Voice-Zero](https://github.com/OwenTyme/voice-zero) (LibriVox recordings), all CC0. Pocket TTS's voices Cosette and Jean are left out, because their recordings are licensed for non-commercial use only.
- Ears: [faster-whisper](https://github.com/SYSTRAN/faster-whisper) with OpenAI's Whisper models.
- Tray icon: [pystray](https://github.com/moses-palmer/pystray) (LGPL-3.0, used as an unmodified, separately installed library) and [Pillow](https://python-pillow.org) (MIT-CMU), installed on Windows only.

## Test

```bash
python -m pytest
```

## License

MIT, copyright (c) 2026 Albertus Ivan Wijaya. See [LICENSE](LICENSE).

