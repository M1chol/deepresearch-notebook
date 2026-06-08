from textual.app import App, ComposeResult
from textual.screen import Screen
from textual.containers import Container, Horizontal
from textual.widgets import (
    Button,
    Static,
    TextArea,
    Select,
    Label,
    LoadingIndicator,
    Log,
)

import research_ui.research_backend as research_backend

effort = [("low", 0), ("medium", 1), ("high", 2)]


class Landing(Screen):
    BANNER = open("research_ui/banner.txt").read()
    CSS_PATH = "style.tcss"

    def compose(self) -> ComposeResult:
        yield Static(self.BANNER, id="title")
        yield Container(
            TextArea("", placeholder="Prompt here...", id="input"),
            Label("Select effort level:"),
            Horizontal(
                Select(effort, allow_blank=False, id="effort"),
                Button("Start", id="button"),
            ),
            id="main",
        )

    def on_button_pressed(self) -> None:
        prompt = self.query_one("#input", TextArea).text
        effort_value = self.query_one("#effort", Select).value

        self.app.switch_screen("status")
        self.app.call_after_refresh(self.start_research, prompt, effort_value)

    def start_research(self, prompt: str, effort_value: int) -> None:
        status = self.app.get_screen("status")
        research_backend.start_research(status, prompt, effort_value)


class Loading(Screen):
    CSS = "Label { content-align: center middle; }"

    def compose(self) -> ComposeResult:
        yield LoadingIndicator()


class Status(Screen):
    def compose(self) -> ComposeResult:
        yield Log(id="status-log")

    def on_mount(self) -> None:
        self.query_one("#status-log", Log).write_line("Status page active")

    def on_job_log(self, message: research_backend.JobLog) -> None:
        self.query_one("#status-log", Log).write_line(
            f"[{message.job_name}] {message.text}"
        )


class MyApp(App):
    SCREENS = {
        "landing": Landing,
        "loading": Loading,
        "status": Status,
    }

    def on_mount(self) -> None:
        self.push_screen("landing")


if __name__ == "__main__":
    MyApp().run()
