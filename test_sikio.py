import asyncio
import contextlib
import json
import os
import re
import sys
import time

import numpy as np
import pytest

import sikio as av

QUIET = np.zeros(av.BLOCK, dtype=np.float32)
LOUD = np.full(av.BLOCK, 0.5, dtype=np.float32)
FAKE_KEY = "fake-test-key-not-real"
URL = "http://localhost:11434/v1"


@pytest.fixture(autouse=True)
def own_settings(tmp_path, monkeypatch):
    """No test may read or write the user's real settings file or stored key."""
    monkeypatch.setattr(av, "SETTINGS", tmp_path / "sikio" / "settings.json")
    monkeypatch.setattr(av, "KEY_SERVICE", "sikio-test")
    monkeypatch.delenv("SIKIO_API_KEY", raising=False)


def test_settings_round_trip():
    values = {"brain": "http://localhost:11434/v1", "model": "gemma3:4b", "confirm": False, "wake": "computer",
              "voice": "george", "stt_model": "small.en"}
    av.save_settings(values)
    assert av.load_settings() == values


def test_missing_or_broken_settings_file_means_defaults():
    assert av.load_settings() == {}
    av.SETTINGS.parent.mkdir(parents=True)
    for broken in ("{not json", "[1, 2]", "", "\xff\xfe"):
        av.SETTINGS.write_text(broken, encoding="latin-1")
        assert av.load_settings() == {}
    av.SETTINGS.write_text('{"confirm": "no", "wake": 5, "brain": "codex", "other": 1}')
    assert av.load_settings() == {"brain": "codex"}  # a wrong type is dropped, so "no" can never turn confirmation off
    assert av.merged({}, av.load_settings()) == {**av.DEFAULTS, "brain": "codex"}


def test_command_line_over_settings_file_over_defaults():
    assert av.merged({"brain": None, "confirm": None, "text": None}, {}) == av.DEFAULTS
    got = av.merged({"brain": None, "wake": "computer", "confirm": None}, {"brain": "gemini", "wake": "friday", "confirm": False})
    assert got == {**av.DEFAULTS, "brain": "gemini", "wake": "computer", "confirm": False}
    assert av.merged({"confirm": False}, {"confirm": True})["confirm"] is False


def run_main(monkeypatch, *argv):
    """The options main() hands to run(), with the window replaced by a fake that saves."""
    seen = {}

    async def run(args):
        seen.update(vars(args))

    monkeypatch.setattr(av, "run", run)
    monkeypatch.setattr(av, "edit_settings", lambda: (seen.setdefault("window", True), av.save_settings({"wake": "friday"})))
    monkeypatch.setattr(sys, "argv", ["sikio", *argv])
    av.main()
    return seen


def test_confirmation_from_flag_file_and_default(monkeypatch):
    assert run_main(monkeypatch, "--text", "hi")["confirm"] is True
    assert run_main(monkeypatch, "--text", "hi", "--no-confirm")["confirm"] is False
    av.save_settings({"confirm": False})
    assert run_main(monkeypatch, "--text", "hi")["confirm"] is False
    assert run_main(monkeypatch, "--text", "hi", "--confirm")["confirm"] is True
    av.save_settings({"confirm": True})
    assert run_main(monkeypatch, "--text", "hi", "--no-confirm")["confirm"] is False


def test_window_only_on_a_first_run_without_arguments(monkeypatch):
    assert "window" not in run_main(monkeypatch, "--text", "hi")
    assert "window" not in run_main(monkeypatch, "--wake", "computer")
    first = run_main(monkeypatch)
    assert first["window"] and first["wake"] == "friday"  # the saved settings apply to this run already
    assert "window" not in run_main(monkeypatch)


def test_warning_whenever_confirmation_is_off(monkeypatch, capsys):
    monkeypatch.setattr(av.shutil, "which", lambda name: name)
    av.Claude("sonnet", confirm=True)
    assert "OFF" not in capsys.readouterr().out
    av.Claude("sonnet", confirm=False)
    assert "Spoken confirmation is OFF" in capsys.readouterr().out


def fake_projects(monkeypatch, tmp_path):
    """A Claude config folder with an old and a newer session, and a subagent file that must be ignored."""
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "claude"))
    work = tmp_path / "work"
    work.mkdir()
    project = tmp_path / "claude" / "projects" / "p"
    (project / "sub").mkdir(parents=True)
    old, new = "11111111-1111-1111-1111-111111111111", "22222222-2222-2222-2222-222222222222"
    (project / f"{old}.jsonl").write_text('{"cwd": "elsewhere"}\n')
    (project / f"{new}.jsonl").write_text('{"type": "summary"}\nnot json\n' + json.dumps({"cwd": str(work)}) + "\n")
    (project / "sub" / "33333333-3333-3333-3333-333333333333.jsonl").write_text('{"cwd": "x"}\n')
    (project / "agent-x.jsonl").write_text('{"cwd": "x"}\n')
    os.utime(project / f"{old}.jsonl", (1, 1))
    os.utime(project / "agent-x.jsonl", (time.time() + 99,) * 2)
    return new, work


def test_latest_session_and_its_folder(monkeypatch, tmp_path):
    new, work = fake_projects(monkeypatch, tmp_path)
    session, folder, age = av.latest_session()
    assert (session, folder) == (new, str(work)) and age >= 0
    assert av.ago(120) == "2 minutes" and av.ago(7200) == "2 hours" and av.ago(86400 * 3) == "3 days" and av.ago(60) == "1 minute" and av.ago(3600) == "1 hour" and av.ago(86400 * 30) == "30 days"
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "none"))
    assert av.latest_session() is None


def test_chat_latest_resumes_a_branch_of_it_and_own_does_not(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(av.shutil, "which", lambda name: name)
    monkeypatch.setattr(av, "SESSION", tmp_path / "session")
    new, work = fake_projects(monkeypatch, tmp_path)
    started = []

    class Client:
        def __init__(self, options):
            started.append((options.resume, options.cwd, options.fork_session))

        async def connect(self):
            pass

    monkeypatch.setattr(av, "ClaudeSDKClient", Client)
    asyncio.run(av.Claude("sonnet", confirm=True, chat="own").start())
    assert started == [(None, None, False)]
    asyncio.run(av.Claude("sonnet", confirm=True, chat="latest").start())
    assert started[1] == (new, str(work), True)
    assert f"Continuing your latest Claude Code session in {work}" in capsys.readouterr().out
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "none"))
    av.SESSION.write_text("mine")
    asyncio.run(av.Claude("sonnet", confirm=True, chat="latest").start())
    assert started[2] == ("mine", None, False)  # nothing found: the own chat, with a printed reason
    assert "No Claude Code session found" in capsys.readouterr().out


def test_chat_setting_from_flag_and_file(monkeypatch):
    assert run_main(monkeypatch, "--text", "hi")["chat"] == "own"
    av.save_settings({"chat": "latest"})
    assert run_main(monkeypatch, "--text", "hi")["chat"] == "latest"
    assert run_main(monkeypatch, "--text", "hi", "--chat", "own")["chat"] == "own"


def test_api_key_from_environment_wins(monkeypatch):
    monkeypatch.setattr(av, "stored_key", lambda url: "stored" if url == "http://localhost:1/v1" else None)
    assert av.Chat("http://localhost:1/v1", "m").key == "stored"
    assert av.Chat("http://localhost:2/v1", "m").key is None  # a key saved for one address is not sent to another
    monkeypatch.setenv("SIKIO_API_KEY", "from-env")
    assert av.Chat("http://localhost:1/v1", "m").key == "from-env"


CLAUDE = next(name for name, code in av.NAMED.items() if code == "claude")
OLLAMA = next(name for name, code in av.NAMED.items() if code == URL)


@pytest.fixture
def window():
    """Opens the settings window (skips the test without tkinter or a display) and closes every window it opened."""
    tk = pytest.importorskip("tkinter")  # Homebrew and many Linux Pythons come without it
    opened = []

    def open_():
        for attempt in range(5):  # on this machine Tk itself fails to start about one time in ten ("Can't find a usable tk.tcl"), even in a bare tk.Tk() loop
            try:
                root, fields, save = av.settings_window()
                break
            except tk.TclError as error:
                last = error
        else:
            pytest.skip(f"no display: {last}")
        opened.append(root)
        return root, fields, save
    yield open_
    for root in opened:
        try:
            root.destroy()
        except tk.TclError:
            pass


@pytest.fixture
def keys(monkeypatch):
    """A dict instead of the credential store."""
    store = {}
    monkeypatch.setattr(av.keyring, "get_password", lambda service, name: store.get((service, name)))
    monkeypatch.setattr(av.keyring, "set_password", lambda service, name, value: store.__setitem__((service, name), value))
    monkeypatch.setattr(av.keyring, "delete_password", lambda service, name: store.pop((service, name)))
    return store


def shown(fields, name):
    return all(widget.winfo_manager() for widget in fields["rows"][name])


def test_the_owners_saved_file_loads_into_the_window(window):
    # The owner's file still names a voice of the old engine: the window shows the default voice and saving writes it.
    av.save_settings({"brain": "claude", "confirm": True, "wake": "jarvis", "voice": "bm_lewis:2+am_michael:1", "stt_model": "base.en"})
    root, fields, save = window()
    assert [fields[k].get() for k in ("choice", "model", "voice", "stt_model", "wake")] == [
        CLAUDE, "Sonnet: balanced", "Michael (default)", "Fast (English)", "jarvis"]
    assert not shown(fields, "where") and not shown(fields, "key")
    assert "Claude Code installed" in fields["hint"].get()
    save()
    assert av.load_settings() == {"brain": "claude", "confirm": True, "wake": "jarvis", "voice": "michael", "stt_model": "base.en"}


@pytest.mark.parametrize("picked, code", [*av.NAMED.items(), (av.WEB, "https://my.example/v1"), (av.COMMAND, "ollama run llama3.2")])
def test_every_brain_round_trips(window, picked, code):
    root, fields, save = window()
    fields["choice"].set(picked)
    if picked in (av.WEB, av.COMMAND):
        fields["where"].set(code + "/" * (picked == av.WEB))  # a trailing slash is dropped
    if code.startswith("http"):
        fields["model"].set("some-model")
    save()
    assert av.load_settings()["brain"] == code
    assert av.load_settings().get("model") == ("some-model" if code.startswith("http") else None)
    root, fields, save = window()
    assert fields["choice"].get() == picked
    assert fields["where"].get() == (code if picked in (av.WEB, av.COMMAND) else "")
    assert shown(fields, "where") == (picked in (av.WEB, av.COMMAND)) and shown(fields, "key") == code.startswith("http")
    assert str(fields["model_box"].cget("state")) == ("disabled" if not code.startswith("http") and code != "claude" else "normal")
    assert fields["hint"].get()


def test_saved_values_that_are_in_no_list_show_as_typed(window):
    av.save_settings({"brain": "my-tool --fast", "voice": "bm_daniel", "stt_model": "tiny.en", "model": "x"})
    root, fields, save = window()
    assert (fields["choice"].get(), fields["where"].get(), fields["voice"].get(), fields["stt_model"].get()) == (
        av.COMMAND, "my-tool --fast", "Michael (default)", "tiny.en")  # only built-in voices exist: an unknown one is the default
    assert fields["model"].get() == ""  # a command brain gets no model
    save()
    assert av.load_settings() == {"brain": "my-tool --fast", "confirm": True, "wake": av.DEFAULTS["wake"], "voice": "michael", "stt_model": "tiny.en"}


def test_hidden_fields_leave_nothing_stale(window, keys):
    root, fields, save = window()
    fields["choice"].set(av.WEB)
    fields["where"].set("https://my.example/v1")
    fields["model"].set("gemma3:4b")
    fields["key"].set(FAKE_KEY)
    fields["choice"].set(av.COMMAND)
    assert fields["where"].get() == "" and fields["model"].get() == "" and not shown(fields, "key")
    fields["choice"].set(CLAUDE)
    fields["choice"].set(OLLAMA)
    fields["choice"].set(CLAUDE)
    assert fields["where"].get() == "" and fields["model"].get() == "Sonnet: balanced"
    save()
    assert av.load_settings() == {"brain": "claude", "confirm": True, "wake": av.DEFAULTS["wake"], "voice": "michael", "stt_model": "base.en"}
    assert keys == {}  # the key typed for the address that was then abandoned is not saved
    fields["choice"].set(av.WEB)
    assert fields["where"].get() == "" and fields["model"].get() == ""  # switching back starts empty too


def test_claude_models_get_plain_names(window):
    root, fields, save = window()
    assert list(fields["model_box"].cget("values")) == ["Sonnet: balanced", "Haiku: fastest, uses least", "Opus: strongest, slowest, uses most"]
    fields["model"].set("Haiku: fastest, uses least")
    save()
    assert av.load_settings()["model"] == "haiku"
    root, fields, save = window()
    assert fields["model"].get() == "Haiku: fastest, uses least"
    fields["model"].set("claude-opus-9")  # typed by hand
    save()
    assert av.load_settings()["model"] == "claude-opus-9"
    root, fields, save = window()
    assert fields["model"].get() == "claude-opus-9"


def test_friendly_voice_and_recognition_names_and_typed_values(window):
    root, fields, save = window()
    fields["voice"].set("Peter Yearsley")
    fields["stt_model"].set("Other languages")
    save()
    assert (av.load_settings()["voice"], av.load_settings()["stt_model"]) == ("peter_yearsley", "small")
    root, fields, save = window()
    assert (fields["voice"].get(), fields["stt_model"].get()) == ("Peter Yearsley", "Other languages")
    assert str(fields["rows"]["voice"][1].cget("state")) == "readonly"  # no typed voices: a typed path would clone a voice
    assert set(av.STT_MODELS.values()) == {"base.en", "small.en", "small"}
    assert list(av.VOICES.items())[0] == ("Michael (default)", av.DEFAULTS["voice"])


class Reply:
    def __init__(self, body):
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def read(self, *_):
        return self.body


def test_find_models_fills_the_list(window, keys, monkeypatch):
    sent = []
    monkeypatch.setattr(av.PATIENT, "open", lambda request, timeout: sent.append((request, timeout)) or Reply(
        b'{"object": "list", "data": [{"id": "gemma3:4b"}, {"id": "llama3.1:8b"}]}'))
    root, fields, save = window()
    fields["choice"].set(OLLAMA)
    fields["find"]()
    assert list(fields["model_box"].cget("values")) == ["gemma3:4b", "llama3.1:8b"] and "Found 2" in fields["status"].get()
    request, timeout = sent[0]
    assert request.full_url == URL + "/models" and timeout <= 10 and request.get_header("Authorization") is None
    fields["choice"].set(av.WEB)
    fields["where"].set("https://my.example/v1/")
    fields["key"].set(FAKE_KEY)
    fields["find"]()
    assert sent[1][0].full_url == "https://my.example/v1/models" and sent[1][0].get_header("Authorization") == f"Bearer {FAKE_KEY}"
    keys[(av.KEY_SERVICE, "https://my.example/v1")] = "saved"
    fields["key"].set("")
    fields["find"]()
    assert sent[2][0].get_header("Authorization") == "Bearer saved"


def test_find_models_failures_are_one_line(window, keys, monkeypatch):
    def fail(error):
        def urlopen(request, timeout):
            raise error
        monkeypatch.setattr(av.PATIENT, "open", urlopen)
    root, fields, save = window()
    fields["choice"].set(OLLAMA)
    fail(av.urllib.error.URLError("refused"))
    fields["find"]()
    assert fields["status"].get() == "Could not reach Ollama. Is it running?"
    fail(av.urllib.error.HTTPError(URL, 401, "no", {}, None))
    fields["find"]()
    assert "HTTP 401" in fields["status"].get()
    fail(TimeoutError())
    fields["find"]()
    assert "Could not reach Ollama" in fields["status"].get()
    monkeypatch.setattr(av.PATIENT, "open", lambda request, timeout: Reply(b"<html>"))
    fields["find"]()
    assert "did not answer with a list" in fields["status"].get()
    monkeypatch.setattr(av.PATIENT, "open", lambda request, timeout: Reply(b'{"data": [1]}'))
    fields["find"]()
    assert "did not answer with a list" in fields["status"].get()
    fields["choice"].set(av.WEB)
    fields["find"]()
    assert fields["status"].get() == "Type the web address first."


def test_find_models_on_local_ollama(window):
    try:
        av.list_models(URL, None)
    except OSError:
        pytest.skip("Ollama is not running")
    root, fields, save = window()
    fields["choice"].set(OLLAMA)
    fields["find"]()
    assert "Found" in fields["status"].get() and fields["model_box"].cget("values")


def test_web_address_still_needs_a_model_even_if_find_was_never_pressed(window):
    root, fields, save = window()
    fields["choice"].set(OLLAMA)
    save()
    assert not av.SETTINGS.exists() and "model" in fields["status"].get()
    fields["model"].set("gemma3:4b")
    save()
    assert av.load_settings()["model"] == "gemma3:4b"


def test_key_note_follows_the_selected_address(window, keys):
    keys[(av.KEY_SERVICE, URL)] = "k"
    root, fields, save = window()
    assert fields["key_note"].get() == ""
    fields["choice"].set(OLLAMA)
    assert fields["key_note"].get() == "A key is saved for this address."
    fields["choice"].set(next(n for n, c in av.NAMED.items() if c == "https://api.openai.com/v1"))
    assert fields["key_note"].get() == "No key saved."
    fields["choice"].set(av.WEB)
    fields["where"].set(URL)
    assert fields["key_note"].get() == "A key is saved for this address."
    fields["where"].set("http://other.example/v1")
    assert fields["key_note"].get() == "No key saved."
    fields["choice"].set(CLAUDE)
    assert fields["key_note"].get() == ""


def test_key_saved_per_normalized_address_and_cleared(window, keys):
    root, fields, save = window()
    fields["choice"].set(av.WEB)
    fields["where"].set(URL + "/")
    fields["model"].set("m")
    fields["key"].set(FAKE_KEY)
    save()
    assert keys == {(av.KEY_SERVICE, URL): FAKE_KEY} and FAKE_KEY not in av.SETTINGS.read_text()
    root, fields, save = window()
    assert fields["key"].get() == "" and fields["key_note"].get() == "A key is saved for this address."
    fields["clear_key"].set(True)
    save()
    assert keys == {}


def test_a_credential_store_failure_is_a_message_not_a_crash(window, monkeypatch):
    def broken(*_):
        raise av.KeyringError("no store")
    monkeypatch.setattr(av.keyring, "get_password", broken)
    monkeypatch.setattr(av.keyring, "set_password", broken)
    root, fields, save = window()
    fields["choice"].set(OLLAMA)
    fields["model"].set("m")
    fields["key"].set(FAKE_KEY)
    save()
    assert "Could not save the key" in fields["status"].get() and not av.SETTINGS.exists()
    assert root.winfo_exists()


def test_closing_the_window_saves_nothing(window):
    root, fields, save = window()
    fields["wake"].set("computer")
    root.destroy()
    assert not av.SETTINGS.exists()


def test_report_prints_the_line_and_tells_the_tray(monkeypatch, capsys):
    seen = []
    monkeypatch.setattr(av, "on_state", lambda state, line: seen.append((state, line)))
    av.report("starting", "Loading speech recognition...")
    av.report("speaking")
    assert capsys.readouterr().out == "Loading speech recognition...\n"  # a state alone prints nothing: the terminal is unchanged
    assert seen == [("starting", "Loading speech recognition..."), ("speaking", "")]


def test_log_starts_a_new_file_past_its_limit(tmp_path):
    log = av.Log(tmp_path / "sikio.log", limit=100)
    for n in range(12):
        print(f"line {n} " + "x" * 20, file=log)
    log.write("no line end yet, ")
    log.write("same line\n")
    log.file.close()
    old, new = (tmp_path / "sikio.log.1").read_text(), (tmp_path / "sikio.log").read_text()
    assert len(new) <= 100 + 50 and len(old) <= 100 + 50  # one line past the limit at most
    assert new.endswith("no line end yet, same line\n") and "line 11" in old and "line 0 " not in old + new  # only one old file is kept
    assert all(av.re.match(r"\d{4}-\d\d-\d\d \d\d:\d\d:\d\d line|.* no line end", line) for line in (old + new).splitlines())


def test_only_one_instance_listens():
    first = av.single_instance()
    assert first and av.single_instance() is None
    first.close()
    second = av.single_instance()
    assert second
    second.close()


def test_second_instance_in_the_terminal_says_so(monkeypatch):
    lock = av.single_instance()
    try:
        with pytest.raises(SystemExit, match="already running"):
            run_main(monkeypatch, "--wake", "computer")
        assert run_main(monkeypatch, "--text", "hi")["text"] == "hi"  # a typed request does not listen, so it may run
    finally:
        lock.close()


class FakeRegistry:
    """winreg with a dict as this user's Run key."""
    HKEY_CURRENT_USER, REG_SZ = "HKCU", 1

    def __init__(self):
        self.values = {}

    def CreateKey(self, root, path):
        assert (root, path) == ("HKCU", r"Software\Microsoft\Windows\CurrentVersion\Run")  # this user only: no admin rights
        return contextlib.nullcontext("key")

    def SetValueEx(self, key, name, reserved, kind, value):
        self.values[name] = (kind, value)

    def DeleteValue(self, key, name):
        if name not in self.values:
            raise FileNotFoundError(name)
        del self.values[name]

    def QueryValueEx(self, key, name):
        if name not in self.values:
            raise FileNotFoundError(name)
        return self.values[name][1], self.values[name][0]


def test_start_with_windows_entry(monkeypatch):
    registry = FakeRegistry()
    monkeypatch.setitem(sys.modules, "winreg", registry)
    assert av.autostart() is False
    assert av.autostart(True) is True and av.autostart() is True
    kind, command = registry.values["Sikio"]
    exe, script, mode = av.re.fullmatch(r'"([^"]+)" "([^"]+)" (\w+)', command).groups()
    assert kind == registry.REG_SZ and exe.lower().endswith("pythonw.exe") and mode == "tray"  # pythonw: no console at login
    assert av.Path(script) == av.Path(av.__file__).resolve()
    assert av.autostart(False) is False and registry.values == {}
    assert av.autostart(False) is False  # removing it twice is fine


def test_tray_shows_each_state():
    pytest.importorskip("PIL")
    images = {state: av.tray_image(state).tobytes() for state in av.LOOKS}
    assert len(set(images.values())) == len(av.LOOKS)
    assert av.tray_title("listening", "", "computer") == 'Sikio: listening for "computer"'
    assert av.tray_title("starting", "Loading speech recognition...", "x") == "Sikio: starting\nLoading speech recognition..."
    assert av.tray_title("speaking", "ignored", "x") == "Sikio: speaking"
    assert len(av.tray_title("error", "x" * 500, "x")) == 127  # the most Windows shows


@pytest.fixture
def tray(monkeypatch):
    """A Tray whose icon is never shown, with run() replaced by `fake` from the test."""
    pytest.importorskip("pystray")
    monkeypatch.setattr(av, "autostart", lambda on=None: False)  # the menu reads it; the real registry is left alone
    app, notes = av.Tray(), []
    app.icon.notify = lambda message, title: notes.append(message)
    monkeypatch.setattr(av, "on_state", app.show)
    yield app, notes
    app.stop(wait=5)
    av.QUIT.clear()
    av.PAUSED.clear()
    av.NEW_CHAT.clear()


def test_tray_menu_reaches_the_assistant(tray, monkeypatch):
    app, notes = tray
    started = av.threading.Event()

    async def run(args):
        av.report("listening")
        started.set()
        try:
            await av.asyncio.sleep(60)
        finally:
            await av.asyncio.sleep(0.2)  # its cleanup: the voice stops, the brain disconnects
    monkeypatch.setattr(av, "run", run)
    app.start()
    assert started.wait(5) and app.state == "listening" and app.icon.title.startswith("Sikio: listening")
    app.pause()
    assert av.PAUSED.is_set() and app.icon.title == "Sikio: paused, not listening"
    av.report("speaking")  # a reply under way while paused: the icon still says paused
    assert app.icon.title == "Sikio: paused, not listening"
    app.pause()
    assert not av.PAUSED.is_set() and app.icon.title == "Sikio: speaking"
    app.new_chat()
    assert av.NEW_CHAT.is_set()
    began = av.time.monotonic()
    app.stop(wait=2)
    assert app.ended.is_set() and av.time.monotonic() - began < 1 and av.QUIT.is_set()
    app.thread.join(2)
    assert not app.thread.is_alive() and notes == [] and app.state == "speaking"  # a quit is no error


def test_tray_shows_why_the_assistant_stopped(tray, monkeypatch):
    app, notes = tray

    async def run(args):
        av.sys.exit("Could not open a sound output. Connect speakers or headphones and start again.")
    monkeypatch.setattr(av, "run", run)
    app.start()
    app.thread.join(5)
    assert app.state == "error" and "sound output" in app.icon.title and "sound output" in notes[0]


def test_cost_of_this_request_and_of_the_chat(monkeypatch):
    monkeypatch.setattr(av, "SESSION", av.SETTINGS.parent / "session")
    monkeypatch.setattr(av, "SPENT", {})
    assert av.cost_text("new", 0.0053) == "$0.005 this request, $0.005 this chat"  # a new chat: the total is the request
    av.SESSION.parent.mkdir(parents=True)
    av.SESSION.write_text("new")
    assert av.cost_text("new", 0.0083) == "$0.003 this request, $0.008 this chat"
    assert av.cost_text("new", 0.0984) == "$0.090 this request, $0.098 this chat"
    av.SPENT.clear()  # the app was restarted and resumed this chat: the total includes requests of the last run
    assert av.cost_text("new", 0.1) == "$0.100 this chat"
    assert av.cost_text("new", 0.12) == "$0.020 this request, $0.120 this chat"
    av.SESSION.unlink()  # "new chat": a different id, whose total starts at zero
    assert av.cost_text("other", 0.017) == "$0.017 this request, $0.017 this chat"


def test_settings_window_saves():
    tk = pytest.importorskip("tkinter")
    try:
        root, fields, save = av.settings_window()
    except tk.TclError as error:
        pytest.skip(f"no display: {error}")
    try:
        assert fields["choice"].get() == CLAUDE and fields["confirm"].get() is True and fields["key"].get() == ""
        fields["choice"].set(av.WEB)
        fields["where"].set("")
        save()  # refused: a web address is needed
        assert not av.SETTINGS.exists()
        fields["where"].set(URL)
        save()  # refused: a web address needs a model
        assert not av.SETTINGS.exists()
        for name, value in (("model", "gemma3:4b"), ("confirm", False), ("wake", " computer "), ("voice", ""), ("key", FAKE_KEY)):
            fields[name].set(value)
        save()
        assert av.load_settings() == {"brain": "http://localhost:11434/v1", "model": "gemma3:4b", "confirm": False,
                                      "wake": "computer", "stt_model": "base.en"}
        assert FAKE_KEY not in av.SETTINGS.read_text()
        assert av.stored_key(URL) == FAKE_KEY and av.stored_key("https://other.example/v1") is None
        root, fields, save = av.settings_window()
        assert fields["choice"].get() == OLLAMA and fields["key"].get() == ""  # a saved URL shows its preset; the stored key is never shown
        fields["clear_key"].set(True)
        save()
        assert av.stored_key(URL) is None
    finally:
        if av.stored_key(URL):
            av.keyring.delete_password(av.KEY_SERVICE, URL)


def test_utterances_split_on_quiet():
    blocks = [QUIET] * 5 + [LOUD] * 4 + [QUIET] * 8 + [LOUD] * 2 + [QUIET] * 8
    found = list(av.utterances(blocks, threshold=0.01))
    # 3 lead-in blocks + loud blocks + 8 quiet blocks; the lead-in is empty the second time
    assert [len(u) // av.BLOCK for u in found] == [15, 10]


def test_utterance_is_cut_when_noise_never_stops():
    found = list(av.utterances([LOUD] * 700, threshold=0.01))
    assert [len(u) // av.BLOCK for u in found] == [300, 300]


def test_mic_blocks_feed_utterances():
    heard = av.queue.Queue()
    for _ in range(400):  # what the mic's stream callback queues
        heard.put(np.zeros(av.BLOCK, dtype=np.float32))
    # a negative threshold makes silence count as loud, so an utterance ends at max_blocks
    assert len(next(av.utterances(av.mic_blocks(heard), threshold=-1))) == 300 * av.BLOCK


def test_after_wake():
    assert av.after_wake("Hey Jarvis, what time is it?", "jarvis") == "what time is it?"
    assert av.after_wake("Jarvis.", "jarvis") == ""
    assert av.after_wake("what time is it", "jarvis") is None
    assert av.after_wake("the jarvises are here", "jarvis") is None


def test_listening_for_a_while_gives_up_only_if_no_speech_starts():
    assert list(av.utterances([QUIET] * 20, threshold=0.01, wait=10)) == []
    late = [QUIET] * 9 + [LOUD] * 30 + [QUIET] * 8  # starts within the wait and runs past it: kept whole
    assert [len(u) // av.BLOCK for u in av.utterances(late, threshold=0.01, wait=10)] == [41]


def test_follow_up_needs_no_wake_word_only_while_open():
    assert av.request_in("Say hello to Sam.", "jarvis", open_=True) == "Say hello to Sam."
    assert av.request_in("Say hello to Sam.", "jarvis", open_=False) is None
    assert av.request_in("Jarvis, what time is it?", "jarvis", open_=True) == "what time is it?"
    assert av.request_in("Jarvis.", "jarvis", open_=False) == ""
    for noise in ("", "  "):  # silence or noise is never a request, open or not
        assert av.request_in(noise, "jarvis", open_=True) is None
    assert av.asks("What should the email say?") and av.asks('Shall I say "hi?" ')
    assert not av.asks("Sure, done.") and not av.asks("") and not av.asks("Why? Because it is late.")


def test_only_the_wake_word_interrupts_a_reply():
    saying = "Sure. The weather tomorrow is sunny, and Jarvis is my name if you need me again."
    assert av.interruption("Jarvis, what time is it?", "jarvis", saying) == "what time is it?"
    for stop in ("Jarvis.", "Jarvis, stop.", "Jarvis stop it", "Jarvis, that's enough.", "Jarvis, never mind"):
        assert av.interruption(stop, "jarvis", saying) == "", stop
    for keep_talking in ("", "what time is it?", "Stop.", "Hey, I said Jarvis, stop", "the jarvises are here",
                         "Jarvis is my name if you need me", "and Jarvis is my name"):  # noise, no wake word, our own voice
        assert av.interruption(keep_talking, "jarvis", saying) is None, keep_talking
    assert av.interruption("Computer, stop", "computer", "") == ""


@pytest.mark.parametrize("draft, final, cut, early", [
    ("Jarvis", "Jarvis, what time is it?", "what time is it?", True),
    ("Jarvis", "Jarvis, stop.", "", True),
    ("Jarvis", "the jarvises are here", "", True),  # stopped by the draft: it stays stopped
    ("What time", "What time is it?", None, False),  # no wake word: it keeps talking
    ("Jarvis is my", "Jarvis is my name if you need me.", None, False),  # its own words coming back through the mic
])
def test_the_wake_word_in_a_draft_stops_the_voice_before_the_words_end(draft, final, cut, early):
    events = []

    class Brain:
        async def ask(self, text):
            for sentence in ["Sure.", "Jarvis is my name if you need me.", *[f"Sentence {n}." for n in range(8)]]:
                yield "say", sentence

        async def interrupt(self):
            events.append("brain interrupted")

    class Voice:
        def say(self, text, hush):
            av.time.sleep(0.05)

        def wait(self):
            pass

        def stop(self):
            events.append("voice stopped")

    calls = []

    async def listen(seconds=None, fresh=True, stop=None, drafted=None):
        calls.append(drafted)
        if len(calls) == 1:
            await asyncio.sleep(0.2)  # the reply's second sentence has been said
            drafted(draft)
            events.append("draft")
            await asyncio.sleep(0.2)  # the rest is still being said
            events.append("words end")
            return final
        while not stop.is_set():
            await asyncio.sleep(0.01)
        return ""

    last, got = asyncio.run(av.answer(Brain(), "tell me", Voice(), listen, "jarvis"))
    assert got == cut and calls[0] is not None
    assert (events[:2] == ["voice stopped", "draft"]) == early, events  # stopped as the draft came, before the words ended
    assert ("brain interrupted" in events) == (cut is not None)


def test_draft_transcription_starts_at_a_pause():
    drafts = []
    blocks = [QUIET] * 3 + [LOUD] * 4 + [QUIET] * 2 + [LOUD] * 2 + [QUIET] * 8
    found = list(av.utterances(blocks, threshold=0.01, paused=lambda audio: drafts.append(len(audio) // av.BLOCK)))
    # a pause of 2 blocks is too short; after the last speech, 3 quiet blocks start a draft 5 blocks before the end
    assert drafts == [3 + 4 + 2 + 2 + av.PAUSE_BLOCKS] and len(found[0]) // av.BLOCK == drafts[0] + av.QUIET_BLOCKS - av.PAUSE_BLOCKS
    drafts.clear()
    list(av.utterances(blocks, threshold=0.01, paused=lambda audio: drafts.append(len(audio) // av.BLOCK), head=8))
    assert drafts == [8, 3 + 4 + 2 + 2 + av.PAUSE_BLOCKS]  # over a reply: also the head, before any pause


def test_sentences():
    assert av.sentences("It is noon. Anything else?") == ["It is noon.", "Anything else?"]


def test_spoken():
    assert av.spoken("This is **very** _really_ `good`.") == "This is very really good."
    assert av.spoken("See [the docs](https://x.io/a_b) now") == "See the docs now"
    assert av.spoken("Okay [laughs] fine (Paris) and [two words] stay") == "Okay fine (Paris) and [two words] stay"
    assert av.spoken("Ollama runs here, not Ollamas.") == "oh lah ma runs here, not Ollamas."
    assert av.spoken("It is noon. Anything else?") == "It is noon. Anything else?"
    assert av.spoken("Sure — try this - then rest, – fine") == "Sure — try this - then rest, – fine"  # the voice pauses at dashes itself


def test_breathe():
    long = "I think the weather will stay warm and dry for most of the afternoon but a few clouds might roll in later"
    assert av.breathe(long) == "I think the weather will stay warm and dry for most of the afternoon, but a few clouds might roll in later"
    assert av.breathe("Go for a short walk outside and then drink a full glass of water") == "Go for a short walk outside, and then drink a full glass of water"
    assert av.breathe("Sure, it is warm and dry today but cloudy later.") == "Sure, it is warm and dry today but cloudy later."
    assert av.breathe("It is warm and dry, and it stays that way for the whole day, but clouds come later.") == "It is warm and dry, and it stays that way for the whole day, but clouds come later."
    assert av.breathe("and then we walked home along the quiet road by the river") == "and then we walked home along the quiet road by the river"
    assert av.spoken("It is noon and sunny. Anything else?") == "It is noon and sunny. Anything else?"
    dash = "It is warm and dry — and it stays that way for the whole day but clouds come later"  # the voice pauses at a dash too
    assert av.breathe(dash) == "It is warm and dry — and it stays that way for the whole day, but clouds come later"
    assert av.breathe("Sure, I think the weather will stay warm and dry for most of the afternoon but clouds come later.") == (
        "Sure, I think the weather will stay warm and dry for most of the afternoon, but clouds come later.")


def test_command_brain_sends_request_on_stdin():
    async def ask(brain, text):
        return [event async for event in brain.ask(text)]

    echo = f'"{sys.executable}" -c "import sys; print(sys.stdin.read()[-11:])"'
    assert asyncio.run(ask(av.Command(echo), "a & b | c\"d")) == [("say", "a & b | c\"d")]
    failing = f'"{sys.executable}" -c "import sys; sys.exit(3)"'
    assert asyncio.run(ask(av.Command(failing), "hi")) == [("say", av.APOLOGY_FAILED)]


def test_complete_sentences_as_text_streams_in():
    assert av.complete_sentences("Sure. Pick one") == (["Sure."], "Pick one")
    assert av.complete_sentences("Pick one task, then go. Good luck") == (["Pick one task, then go."], "Good luck")
    assert av.complete_sentences("Rome.") == ([], "Rome.")
    # a chunk that starts with the line break after the previous sentence
    assert av.complete_sentences("\n\nHmm, owls. They fly") == (["Hmm, owls."], "They fly")
    assert av.complete_sentences("\n") == ([], "")


def test_sse_text_from_a_streamed_chat_completion():
    assert av.sse_text('data: {"choices":[{"delta":{"content":"Hi there."}}]}\n') == "Hi there."
    assert av.sse_text('data: {"choices":[{"delta":{"role":"assistant"}}]}') == ""
    assert av.sse_text('data: {"choices":[{"delta":{},"finish_reason":"stop"}]}') == ""
    assert av.sse_text('data: {"choices":[]}') == ""
    assert av.sse_text("\n") == ""
    assert av.sse_text(": keep-alive\n") == ""
    assert av.sse_text("data: [DONE]\n") is None
    assert av.sse_text("data:\n") == "" and av.sse_text(": OPENROUTER PROCESSING\n") == ""
    assert av.sse_text('data: {"choices":[{"delta":{"reasoning_content":"Hmm","role":"assistant"}}]}') == ""  # xAI's reasoning is not said
    assert av.sse_text('data: {"choices":[{"delta":{"content":"","reasoning":"Hmm"}}]}') == ""  # OpenRouter's neither
    with pytest.raises(ValueError, match="Provider disconnected"):  # OpenRouter's error after the stream started
        av.sse_text('data: {"error":{"code":502,"message":"Provider disconnected"},"choices":[{"delta":{"content":""},"finish_reason":"error"}]}')


def test_only_a_clear_yes_allows():
    for said in ("Yes.", "Yeah, go ahead.", "yep", "Sure!", "Okay.", "Do it.", "Please do.", "Jarvis, yes please"):
        assert av.is_yes(said), said
    for said in ("", "No.", "Not yet.", "Yes... no wait.", "Don't do it.", "Nope", "Yes, but use the other file.",
                 "What?", "I'm not sure", "yesterday", "Do it later? No.", "Hmm", "Okay, wait."):
        assert not av.is_yes(said), said


def test_new_chat_phrase():
    for said in ("New chat.", "Start a new conversation!", "start over", "Start new chat please.", "new conversation"):
        assert av.is_new_chat(said), said
    for said in ("Tell me about the new chat feature.", "Should I start over with the essay?", "What's new?", ""):
        assert not av.is_new_chat(said), said


def test_confirmation_question():
    assert av.question("Bash", {"command": "git status", "description": "x"}) == "I'd like to run a command: git status. Shall I go ahead?"
    assert av.question("Write", {"file_path": "C:\\work\\notes.txt", "content": "secret text"}) == "I'd like to write the file notes.txt. Shall I go ahead?"
    assert av.question("Edit", {"file_path": "/a/b.py", "old_string": "x", "new_string": "y"}) == "I'd like to edit the file b.py. Shall I go ahead?"
    assert av.question("mcp__calendar__create_event", {"title": "Lunch"}) == "I'd like to use the tool create event from the calendar server. Shall I go ahead?"
    long = av.question("PowerShell", {"command": "echo " + "a" * 200})
    assert len(long) < 130 and "..." in long
    assert "secret" not in av.question("Write", {"file_path": "x.txt", "content": "secret"})


def test_unknown_voices_fall_back_to_the_default(capsys):
    assert av.voice_name("george") == "george" and capsys.readouterr().out == ""
    for old in ("bm_lewis:2+am_michael:1", "Michael", "cosette", "C:\\voices\\me.wav", ""):  # an old Kokoro voice, a wrong case, a left-out voice, a file
        assert av.voice_name(old) == "michael"
        out = capsys.readouterr().out
        assert out.startswith(f"Unknown voice {old!r}, using michael.") and out.count("\n") == 1


def test_speaker_plays_chunks_back_to_back():
    speaker, out = av.Speaker(10), np.zeros((4, 1), np.float32)
    speaker.fill(out, 4, None, None)
    assert not out.any() and speaker.gaps == 0  # nothing to say: silence, and no gap
    speaker.making = True
    for chunk in ([1, 2, 3], [4, 5], [6, 7, 8, 9, 10]):
        speaker.play(np.array(chunk, np.float32))
    got = []
    for _ in range(3):
        speaker.fill(out, 4, None, None)
        got += out[:, 0].tolist()
    assert got == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 0] and speaker.gaps == 1  # ran dry while more was being made
    speaker.making = False
    speaker.fill(out, 4, None, None)
    assert speaker.gaps == 1 and not speaker.started  # the sentence is over: silence is no gap


def test_speaker_builds_a_reserve_only_after_running_dry(monkeypatch):
    monkeypatch.setattr(av, "RESUME_SECONDS", 0.5)
    speaker, out = av.Speaker(10), np.zeros((2, 1), np.float32)
    speaker.making = True
    speaker.fill(out, 2, None, None)
    assert speaker.gaps == 0  # waiting for the first chunk is no gap
    speaker.play(np.ones(3, np.float32))
    speaker.fill(out, 2, None, None)
    assert out.all()  # the first chunk plays at once
    speaker.fill(out, 2, None, None)
    assert out[:, 0].tolist() == [1, 0] and speaker.gaps == 1  # ran dry while more was being made
    speaker.play(np.ones(4, np.float32))
    speaker.fill(out, 2, None, None)
    assert not out.any() and speaker.queued() == 4 and speaker.gaps == 1  # waits for 5 samples instead of stuttering
    speaker.play(np.ones(1, np.float32))
    speaker.fill(out, 2, None, None)
    assert out.all() and speaker.queued() == 3
    speaker.fill(out, 2, None, None)
    speaker.making = False  # the sentence is all made: what is left plays, whatever the reserve
    speaker.play(np.ones(1, np.float32))
    speaker.fill(out, 2, None, None)
    assert out.all() and speaker.queued() == 0


def test_speaker_stops_at_once_and_waits_for_the_end():
    speaker, out = av.Speaker(10), np.zeros((2, 1), np.float32)
    for _ in range(3):
        speaker.play(np.ones(5, np.float32))
    speaker.fill(out, 2, None, None)
    speaker.stop()
    assert speaker.queued() == 0
    speaker.fill(out, 2, None, None)
    assert not out.any()  # queued voice is dropped, also the rest of the chunk under way
    speaker.play(np.ones(3, np.float32))
    av.threading.Timer(0.1, lambda: [speaker.fill(out, 2, None, None) for _ in range(2)]).start()
    started = av.time.monotonic()
    speaker.wait()
    assert av.time.monotonic() - started >= 0.09 and speaker.queued() == 0  # wait() returns once the device has it all


def test_speaker_resamples_for_a_device_with_another_rate():
    speaker = av.Speaker(24000, 48000)
    speaker.play(np.arange(1920, dtype=np.float32))  # one 80 ms chunk of the voice
    assert speaker.queued() == 3840
    speaker = av.Speaker(24000, 44100)
    speaker.play(np.zeros(1920, np.float32))
    assert speaker.queued() == 3528


def test_speak_says_each_sentence_and_skips_what_has_no_words():
    said = []

    class Fake:
        def say(self, text, hush):
            said.append(text)
            if "stop" in text:
                hush.set()

    av.speak(Fake(), "**Sure.** ... It is noon? [laughs]")
    assert said == ["Sure.", "It is noon?"]
    av.speak(Fake(), "--")
    av.speak(Fake(), "")
    assert said == ["Sure.", "It is noon?"]
    av.speak(Fake(), "Then stop. Not this.")
    assert said[-1] == "Then stop."  # the hush set while saying one sentence stops the rest


def played(speaker, block=480):
    """What the sound device would get: the speaker's callback output, block by block, until nothing is queued."""
    out, blocks = np.zeros((block, 1), np.float32), []
    while speaker.queued():
        speaker.fill(out, block, None, None)
        blocks.append(out[:, 0].copy())
    return np.concatenate(blocks)


def test_local_voice_round_trip():
    """Slow, downloads the models: the app's own voice, through speak() and the speaker's callback, must be understood by the local ears."""
    tts = av.Voice(av.DEFAULTS["voice"], speaker=av.Speaker)
    av.speak(tts, "Juniper, what is the weather tomorrow?")
    samples, rate = played(tts.speaker), tts.model.sample_rate
    assert tts.speaker.gaps == 0
    seconds = len(samples) / rate
    at_16k = np.interp(np.arange(0, seconds, 1 / av.RATE), np.arange(len(samples)) / rate, samples).astype(np.float32)
    heard = av.transcribe(av.load_stt("base.en"), at_16k)
    assert av.after_wake(heard, av.DEFAULTS["wake"]).lower().startswith("what is the weather tomorrow")


def room_clip(sound):
    """What main hands to transcribe: a mic-quiet sound between 0.3 s of hush, peak-normalised."""
    quiet = np.random.default_rng(0).standard_normal(int(0.3 * av.RATE)) * 1e-5
    clip = np.concatenate([quiet, sound / np.abs(sound).max() * 0.003, quiet]).astype(np.float32)
    return clip / np.abs(clip).max()


def fan(rng, n):
    spectrum = np.fft.rfft(rng.standard_normal(n))
    return np.fft.irfft(spectrum / np.sqrt(np.maximum(np.arange(len(spectrum)), 1)), n)


def test_background_noise_is_not_speech():
    """Slow: noise that is normalised to full volume must not become words."""
    rng, stt = np.random.default_rng(1), av.load_stt("base.en")
    click = np.zeros(av.RATE // 2)
    click[av.RATE // 4] = 1
    t = np.arange(int(0.8 * av.RATE)) / av.RATE
    thump = rng.standard_normal(len(t)) * np.exp(-9 * t) + np.sin(2 * np.pi * 60 * t) * np.exp(-6 * t)
    for sound in (rng.standard_normal(av.RATE // 2), click, thump, fan(rng, 3 * av.RATE)):
        assert av.transcribe(stt, room_clip(sound)) == ""


def test_wake_word_survives_the_way_from_mic_to_text():
    """Slow: the wake word said right at the start of speech, in several voices, through utterances() and transcribe();
    speech without it, including sound-alike words, never wakes."""
    (model, _), stt = av.load_tts("michael"), av.load_stt("base.en")
    rng, rate = np.random.default_rng(3), model.sample_rate
    for voice, said, wakes in (("michael", "Juniper, I think I already have my GitHub MCP in my settings.", True),
                               ("alba", "Juniper, send an email to Sarah.", True),
                               ("george", "Juniper open my calendar for Friday.", True),
                               ("mary", "Jennifer said she would call later.", False),
                               ("javert", "The jury is still out on that.", False)):
        samples = model.generate_audio(model.get_state_for_audio_prompt(voice), said).numpy()
        speech = np.interp(np.arange(0, len(samples) / rate, 1 / av.RATE), np.arange(len(samples)) / rate, samples)
        room = rng.standard_normal(av.BLOCK * 20) * 4e-5  # this laptop's quiet room
        audio = np.concatenate([room[:av.BLOCK * 10], speech / np.sqrt(np.mean(speech**2)) * 2e-3 + rng.standard_normal(len(speech)) * 4e-5, room[av.BLOCK * 10:]])
        audio = audio[:len(audio) // av.BLOCK * av.BLOCK].astype(np.float32)
        threshold = max(0.0002, 5 * float(np.sqrt(np.mean(room[:av.BLOCK * 5]**2))))
        found = next(av.utterances(list(audio.reshape(-1, av.BLOCK)), threshold))
        heard = av.transcribe(stt, found / np.abs(found).max())
        assert (av.after_wake(heard, av.DEFAULTS["wake"]) is not None) == wakes, (voice, heard)


def test_speech_in_fan_noise_is_still_understood():
    """Slow: a short and a longer request at about 10 dB signal-to-noise."""
    (model, voice), stt = av.load_tts("michael"), av.load_stt("base.en")
    rate = model.sample_rate
    for said, expected in (("Juniper, what is two plus two?", "what is"), ("Juniper.", "")):
        samples = model.generate_audio(voice, said).numpy()
        speech = np.interp(np.arange(0, len(samples) / rate, 1 / av.RATE), np.arange(len(samples)) / rate, samples)
        noise = fan(np.random.default_rng(2), len(speech))
        mixed = speech + noise / np.sqrt(np.mean(noise**2)) * np.sqrt(np.mean(speech**2)) / 10**0.5
        heard = av.after_wake(av.transcribe(stt, room_clip(mixed)), av.DEFAULTS["wake"])
        assert heard is not None and heard.lower().startswith(expected)


def test_gate_allows_an_action_only_after_a_spoken_yes(monkeypatch):
    import types
    monkeypatch.setattr(av.shutil, "which", lambda name: name)

    async def decide(said, interrupted=False, confirm=True):
        """The gate's decisions for two tool calls of one request, and how often the user was asked."""
        brain = av.Claude("sonnet", confirm=confirm)
        brain.events, brain.seen, brain.allowed = asyncio.Queue(), {"t1", "t2"}, False
        brain.passed, brain.asking, brain.interrupted = asyncio.Condition(), asyncio.Lock(), interrupted
        allowed, asked = [], 0
        for tool_id in ("t1", "t2"):
            call = asyncio.create_task(brain.gate("Write", {"file_path": "a.txt"}, types.SimpleNamespace(tool_use_id=tool_id)))
            await asyncio.sleep(0.05)
            while not brain.events.empty():
                kind, _, answer = brain.events.get_nowait()
                assert kind == "confirm"
                asked += 1
                answer.set_result(said)
            allowed.append(isinstance(await asyncio.wait_for(call, 2), av.PermissionResultAllow))
        return allowed, asked

    assert asyncio.run(decide("Yes, go ahead.")) == ([True, True], 1)  # one yes covers the rest of the request
    assert asyncio.run(decide("No.")) == ([False, False], 2)
    assert asyncio.run(decide("")) == ([False, False], 2)  # silence is a no
    assert asyncio.run(decide("Yes.", interrupted=True)) == ([False, False], 0)  # interrupted: a no without asking
    assert asyncio.run(decide("No.", confirm=False)) == ([True, True], 0)  # --no-confirm


def test_interrupting_ends_a_command_brain():
    async def interrupted():
        brain = av.Command(f'"{sys.executable}" -c "import time; time.sleep(30)"')
        asking = asyncio.create_task(brain.ask("hi").__anext__())
        await asyncio.sleep(1)
        await brain.interrupt()
        return await asyncio.wait_for(asking, 5), brain.process.wait(5)

    said, code = asyncio.run(interrupted())
    assert said == ("say", av.APOLOGY_FAILED) and code != 0


def test_only_claude_is_told_the_app_asks_for_a_yes():
    assert "spoken yes" in av.CONFIRM_STYLE and "spoken yes" not in av.STYLE


FAKE_CLI = r'''
import os, sys, time
mode, cursor = os.environ["FAKE_MODE"], "--print" in sys.argv
stdin = "" if cursor else sys.stdin.read()
if mode == "ok":
    print(f"args={sys.argv[1:-1] if cursor else sys.argv[1:]} request={(sys.argv[-1] if cursor else stdin)[-13:]!r}")
elif mode == "codex-no-login":  # measured with codex-cli 0.171.0: progress, then the 401, on stderr; nothing on stdout
    sys.stderr.write("OpenAI Codex v0.171.0\nERROR: Reconnecting... 5/5\nERROR: unexpected status 401 Unauthorized: Missing bearer or basic authentication in header\n")
    sys.exit(1)
elif mode == "gemini-no-login":  # measured with gemini-cli 0.70.0
    sys.stderr.write("Please set an Auth method in your settings.json or specify one of the following environment variables before running: GEMINI_API_KEY\n")
    sys.exit(41)
elif mode == "slow":
    time.sleep(5)
elif mode == "crash":
    raise RuntimeError("boom")
'''


@pytest.fixture
def fake_cli(tmp_path, monkeypatch):
    """Fake codex and gemini first on PATH, started as npm installs them (a .cmd shim on Windows, a script elsewhere).
    Returns a function that sets what they do."""
    (tmp_path / "fake_cli.py").write_text(FAKE_CLI)
    for name in ("codex", "gemini"):
        if av.os.name == "nt":
            (tmp_path / f"{name}.cmd").write_text(f'@"{sys.executable}" "{tmp_path / "fake_cli.py"}" %*\n')
        else:
            (tmp_path / name).write_text(f'#!/bin/sh\nexec "{sys.executable}" "{tmp_path / "fake_cli.py"}" "$@"\n')
            (tmp_path / name).chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}{av.os.pathsep}{av.os.environ['PATH']}")
    return lambda mode: monkeypatch.setenv("FAKE_MODE", mode)


def said_by(brain, text="What is two plus two?"):
    async def ask():
        return [event async for event in brain.ask(text)]
    return asyncio.run(asyncio.wait_for(ask(), 30))


def test_cli_presets_get_the_request_on_stdin_in_read_only_mode(fake_cli):
    fake_cli("ok")
    assert said_by(av.Command("codex")) == [("say", "args=['exec', '--sandbox', 'read-only', '--skip-git-repo-check', '-'] request='two plus two?'")]
    assert said_by(av.Command("gemini")) == [("say", "args=['--approval-mode', 'default'] request='two plus two?'")]
    assert not any(word in part for preset in av.BRAINS.values() for part in preset for word in ("yolo", "dangerous", "force", "full-access"))


def test_cursor_gets_the_request_as_its_last_argument(monkeypatch, tmp_path):
    (tmp_path / "fake_cli.py").write_text(FAKE_CLI)
    monkeypatch.setenv("FAKE_MODE", "ok")
    monkeypatch.setitem(av.BRAINS, "cursor", [sys.executable, str(tmp_path / "fake_cli.py"), *av.BRAINS["cursor"][1:]])
    assert said_by(av.Command("cursor")) == [("say", "args=['--print', '--mode', 'ask', '--output-format', 'text'] request='two plus two?'")]
    monkeypatch.setattr(av.shutil, "which", lambda name: r"C:\Users\me\AppData\Local\cursor-agent\agent.CMD")
    with pytest.raises(SystemExit, match="batch file"):  # cmd.exe would run an "&" of the spoken request
        av.Command("cursor")


@pytest.mark.parametrize("name, mode, printed", [
    ("codex", "codex-no-login", "codex failed (exit code 1). It may not be logged in"),
    ("gemini", "gemini-no-login", "gemini failed (exit code 41). It may not be logged in"),
    ("codex", "crash", "RuntimeError: boom"),
])
def test_a_failing_cli_says_why_and_apologises(fake_cli, capsys, name, mode, printed):
    fake_cli(mode)
    assert said_by(av.Command(name)) == [("say", av.APOLOGY_FAILED)]
    assert printed in capsys.readouterr().err


def test_a_slow_cli_is_stopped(fake_cli, capsys, monkeypatch):
    fake_cli("slow")
    monkeypatch.setattr(av, "TIMEOUT", 1)
    assert said_by(av.Command("gemini")) == [("say", av.APOLOGY_SLOW)]
    assert "gemini gave no answer within 1 s" in capsys.readouterr().err


def test_a_missing_cli_says_so(fake_cli, capsys):
    with pytest.raises(SystemExit, match="'agent' is not installed"):
        av.Command("cursor")
    fake_cli("ok")
    brain = av.Command("codex")
    brain.command[0] += ".gone"  # uninstalled while the app runs
    assert said_by(brain) == [("say", av.APOLOGY_FAILED)] and "Could not start codex" in capsys.readouterr().err


class Service:
    """A local stand-in for a chat service: each model name answers with what `replies` holds for it, a list of SSE lines
    or (status, body); `delay` holds back even the headers, like a local server loading the model."""

    def __init__(self):
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        service = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def do_POST(self):
                model = av.json.loads(self.rfile.read(int(self.headers["Content-Length"])))["model"]
                av.time.sleep(service.delay)
                reply = service.replies[model]
                if isinstance(reply, tuple):
                    body = av.json.dumps(reply[1]).encode()
                    self.send_response(reply[0])
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    return self.wfile.write(body)
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                for line in reply:
                    self.wfile.write((line + "\n\n").encode())

        self.replies, self.delay = {}, 0
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_port}/v1"
        av.threading.Thread(target=self.server.serve_forever, daemon=True).start()


@pytest.fixture
def service(keys):
    running = Service()
    yield running
    running.server.shutdown()
    running.server.server_close()


def chunk(**delta):
    return "data: " + av.json.dumps({"object": "chat.completion.chunk", "choices": [{"index": 0, "delta": delta, "finish_reason": None}]})


STREAMS = {  # the same reply in each service's documented streaming format
    "openai": [chunk(role="assistant", content="", refusal=None), chunk(content="Hi there."), chunk(content=" Bye."),
               'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}', 'data: {"choices":[],"usage":{"total_tokens":9}}', "data: [DONE]"],
    "openrouter": [": OPENROUTER PROCESSING", chunk(role="assistant", content="", reasoning="Let me think"), chunk(role="assistant", content="Hi there."),
                   chunk(role="assistant", content=" Bye."), 'data: {"choices":[{"index":0,"delta":{"content":""},"finish_reason":"stop"}],"usage":{"total_tokens":9}}', "data: [DONE]"],
    "xai": [chunk(reasoning_content="The", role="assistant"), chunk(content="Hi there.", role="assistant"), chunk(content=" Bye.", role="assistant"), "data: [DONE]"],
}


@pytest.mark.parametrize("provider", STREAMS)
def test_chat_speaks_each_services_stream(service, provider):
    service.replies["m"] = STREAMS[provider]
    brain = av.Chat(service.url, "m")
    assert said_by(brain, "hi") == [("say", "Hi there."), ("say", "Bye.")]
    assert brain.history[-1] == {"role": "assistant", "content": "Hi there. Bye."}


@pytest.mark.parametrize("status, body, printed", [  # error bodies as each service documents them
    (401, {"error": {"message": "Incorrect API key provided: fake-***.", "type": "invalid_request_error", "code": "invalid_api_key"}}, "refused the API key"),
    (400, {"code": "Client specified an invalid argument", "error": "Incorrect API key provided: fa***al."}, "refused the API key"),  # xAI
    (402, {"error": {"code": 402, "message": "Insufficient credits. Add credits and retry."}}, "no credit left"),  # OpenRouter
    (429, {"error": {"message": "You exceeded your current quota, please check your plan and billing details.", "type": "insufficient_quota"}}, "no credit left"),
    (429, {"error": {"message": "Rate limit reached for requests", "code": "rate_limit_exceeded"}}, "rate limited"),
    (404, {"error": {"message": "The model `m` does not exist or you do not have access to it.", "code": "model_not_found"}}, "does not know the model 'm'"),
    (400, {"error": {"message": "m is not a valid model ID", "code": 400}}, "does not know the model 'm'"),  # OpenRouter
    (404, {"error": "404 page not found"}, "Nothing at this address"),
])
def test_chat_names_why_the_service_refused(service, capsys, status, body, printed):
    service.replies["m"] = (status, body)
    brain = av.Chat(service.url, "m")
    assert said_by(brain, "hi") == [("say", av.APOLOGY_FAILED)] and brain.history == []
    err = capsys.readouterr().err
    assert printed in err and f"HTTP {status}" in err


def test_chat_stops_at_an_error_in_the_stream(service, capsys):
    service.replies["m"] = [chunk(content="Hi there."), 'data: {"error":{"code":502,"message":"Provider disconnected unexpectedly"},'
                            '"choices":[{"index":0,"delta":{"content":""},"finish_reason":"error"}]}', "data: [DONE]"]
    assert said_by(av.Chat(service.url, "m"), "hi") == [("say", av.APOLOGY_FAILED)]
    assert "Provider disconnected unexpectedly" in capsys.readouterr().err


def test_an_unreachable_address_fails_fast_but_a_slow_first_word_is_waited_for(service, capsys, monkeypatch):
    monkeypatch.setattr(av, "CONNECT_SECONDS", 1)
    monkeypatch.setattr(av, "TIMEOUT", 30)
    began = av.time.monotonic()
    said = said_by(av.Chat("http://10.255.255.1/v1", "m"), "hi")  # not routable: the connection is never answered
    err = capsys.readouterr().err
    if "no answer within" not in err:
        pytest.skip(f"this network refuses the address at once instead of dropping it: {err}")
    assert said == [("say", av.APOLOGY_FAILED)] and av.time.monotonic() - began < 5 and "Could not connect" in err
    service.replies["m"], service.delay = STREAMS["openai"], 2.5  # past CONNECT_SECONDS before even the headers
    assert said_by(av.Chat(service.url, "m"), "hi") == [("say", "Hi there."), ("say", "Bye.")]
    monkeypatch.setattr(av, "TIMEOUT", 1)
    assert said_by(av.Chat(service.url, "m"), "hi") == [("say", av.APOLOGY_SLOW)]
    assert "No answer within 1 s" in capsys.readouterr().err


def test_settings_folder_on_each_system(monkeypatch, tmp_path):
    monkeypatch.setattr(av.Path, "home", lambda: tmp_path)
    if sys.platform == "win32":  # elsewhere pathlib cannot pretend to be Windows
        monkeypatch.setenv("APPDATA", str(tmp_path / "Roaming"))
        assert av.config_dir() == tmp_path / "Roaming"  # unchanged on Windows
    monkeypatch.setattr(av.os, "name", "posix")
    monkeypatch.setattr(av.sys, "platform", "darwin")
    assert av.config_dir().as_posix() == (tmp_path / "Library" / "Application Support").as_posix()  # a PosixPath once os.name is posix
    monkeypatch.setattr(av.sys, "platform", "linux")
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    assert av.config_dir().as_posix() == (tmp_path / ".config").as_posix()
    monkeypatch.setenv("XDG_CONFIG_HOME", (tmp_path / "xdg").as_posix())
    assert av.config_dir().as_posix() == (tmp_path / "xdg").as_posix()


@pytest.mark.parametrize("name, platform, opener", [("posix", "darwin", "open"), ("posix", "linux", "xdg-open"), pytest.param(
    "nt", "win32", None, marks=pytest.mark.skipif(sys.platform != "win32", reason="pathlib cannot pretend to be Windows"))])
def test_open_log_uses_each_systems_opener(monkeypatch, name, platform, opener):
    started = []
    monkeypatch.setattr(av.subprocess, "Popen", lambda command: started.append(command))
    monkeypatch.setattr(av.os, "startfile", lambda path: started.append(["startfile", path]), raising=False)
    monkeypatch.setattr(av.os, "name", name)
    monkeypatch.setattr(av.sys, "platform", platform)
    av.Tray.open_log(None)
    assert started == [[opener or "startfile", av.SETTINGS.parent / "sikio.log"]]


def test_alert_shows_a_tk_message_box_off_windows(monkeypatch, capsys):
    import types
    shown = []
    fake = types.ModuleType("tkinter")
    fake.Tk = lambda: types.SimpleNamespace(withdraw=lambda: None, destroy=lambda: shown.append("closed"))
    fake.messagebox = types.SimpleNamespace(showinfo=lambda title, text, parent: shown.append(text))
    monkeypatch.setitem(sys.modules, "tkinter", fake)
    monkeypatch.setitem(sys.modules, "tkinter.messagebox", fake.messagebox)
    monkeypatch.setattr(av.os, "name", "posix")
    av.alert("Sikio is already running")
    assert shown == ["Sikio is already running", "closed"] and "already running" in capsys.readouterr().out

    def no_display():
        raise RuntimeError("no display")
    fake.Tk = no_display
    av.alert("still printed")  # the line is then only printed, which the tray app logs
    assert "still printed" in capsys.readouterr().out


def test_no_start_with_windows_item_elsewhere(monkeypatch):
    pytest.importorskip("pystray")
    monkeypatch.setattr(av, "autostart", lambda on=None: pytest.fail("the registry was read off Windows"))
    monkeypatch.setattr(av.os, "name", "posix")
    app = av.Tray()
    assert [item.text for item in app.icon.menu] == ["Pause listening", "New chat", "Settings", "Open log", "Quit"]


def test_settings_from_the_tray_open_in_their_own_process_on_macos(monkeypatch):
    ran = []
    monkeypatch.setattr(av.sys, "platform", "darwin")
    monkeypatch.setattr(av.subprocess, "run", lambda command: ran.append(command))
    monkeypatch.setattr(av, "edit_settings", lambda: pytest.fail("Tk would start outside the main thread"))
    app = av.Tray.__new__(av.Tray)  # no icon needed
    app.window = av.threading.Lock()
    app.window.acquire()
    app.edit()
    assert ran == [[sys.executable, str(av.Path(av.__file__).resolve()), "settings"]] and not app.window.locked()


def test_without_portaudio_the_app_says_what_to_install(monkeypatch):
    monkeypatch.setattr(av, "sd", None)
    monkeypatch.setattr(av, "NO_SOUND", OSError("PortAudio library not found"), raising=False)
    with pytest.raises(SystemExit, match="sudo apt install libportaudio2"):
        asyncio.run(av.run(av.argparse.Namespace(**av.DEFAULTS, text="hi")))


def test_gate_asks_anyway_when_the_tool_call_never_shows_up(monkeypatch):
    import types
    real = asyncio.wait_for
    monkeypatch.setattr(av.asyncio, "wait_for", lambda awaitable, timeout: real(awaitable, 0.1))  # the 5 s wait, shortened
    monkeypatch.setattr(av.shutil, "which", lambda name: name)

    async def gate():
        brain = av.Claude("sonnet", confirm=True)
        brain.events, brain.seen, brain.allowed = asyncio.Queue(), set(), False
        brain.passed, brain.asking, brain.interrupted = asyncio.Condition(), asyncio.Lock(), False
        call = asyncio.create_task(brain.gate("Write", {"file_path": "a.txt"}, types.SimpleNamespace(tool_use_id="t9")))
        event = await real(brain.events.get(), 2)
        event[2].set_result("No.")
        await real(call, 2)
        return event[0]

    assert asyncio.run(gate()) == "confirm"


def raw_server(handler):
    """A local server whose handler(connection) writes whatever bytes it likes; returns its port."""
    import socket
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(5)

    def serve():
        while True:
            try:
                connection, _ = server.accept()
            except OSError:
                return
            with connection:
                request = b""
                while b"\r\n\r\n" not in request:
                    request += connection.recv(65536)
                head, _, body = request.partition(b"\r\n\r\n")
                length = int(re.search(rb"(?i)content-length: *(\d+)", head).group(1))
                while len(body) < length:  # unread data at close would reset the connection
                    body += connection.recv(65536)
                handler(connection, request)
    av.threading.Thread(target=serve, daemon=True).start()
    return server


def test_chat_survives_a_server_closing_mid_stream(capsys):
    def handler(connection, request):
        data = chunk(content="Hi there. Bye").encode() + b"\n\n"
        connection.sendall(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\r\n"
                           + hex(len(data) + 50)[2:].encode() + b"\r\n" + data)  # a chunk 50 bytes short of its size: then the connection closes
    server = raw_server(handler)
    try:
        brain = av.Chat(f"http://127.0.0.1:{server.getsockname()[1]}/v1", "m")
        assert said_by(brain, "hi") == [("say", "Hi there."), ("say", av.APOLOGY_FAILED)]
        assert brain.history == []
    finally:
        server.close()
    assert "IncompleteRead" in capsys.readouterr().err


def test_a_redirect_is_not_followed_so_the_key_goes_nowhere_else(monkeypatch, capsys):
    monkeypatch.setenv("SIKIO_API_KEY", FAKE_KEY)
    got = []
    target = raw_server(lambda connection, request: got.append(request))
    port = target.getsockname()[1]
    origin = raw_server(lambda connection, request: connection.sendall(
        f"HTTP/1.1 302 Found\r\nLocation: http://127.0.0.1:{port}/x\r\nContent-Length: 0\r\n\r\n".encode()))
    try:
        brain = av.Chat(f"http://127.0.0.1:{origin.getsockname()[1]}/v1", "m")
        assert said_by(brain, "hi") == [("say", av.APOLOGY_FAILED)]
    finally:
        origin.close()
        target.close()
    assert got == [] and "HTTP 302" in capsys.readouterr().err


def test_wake_word_and_brain_are_cleaned_up_before_use():
    assert av.merged({}, {"wake": ""})["wake"] == av.DEFAULTS["wake"]
    assert av.merged({"wake": "Jarvis!"}, {})["wake"] == "Jarvis"
    assert av.merged({"brain": "  "}, {})["brain"] == av.DEFAULTS["brain"]


def test_chat_with_an_empty_stream_apologises(service, capsys):
    service.replies["m"] = ["data: [DONE]"]
    brain = av.Chat(service.url, "m")
    assert said_by(brain, "hi") == [("say", av.APOLOGY_FAILED)] and brain.history == []
    assert "sent no reply text" in capsys.readouterr().err
