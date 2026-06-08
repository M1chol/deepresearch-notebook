from textual.message import Message
from textual import work
import asyncio


class JobLog(Message):
    def __init__(self, job_name: str, text: str) -> None:
        self.job_name = job_name
        self.text = text
        super().__init__()

@work(exclusive=True)
async def start_research(message_target, prompt: str, effort: int) -> None:
    job_name = "start_research"
    message_target.post_message(JobLog(job_name, "Research started"))
    message_target.post_message(
        JobLog(job_name, f"Prompt: {prompt!r}, effort: {effort}")
    )
    await asyncio.sleep(5)
    message_target.post_message(JobLog(job_name, "Research done"))