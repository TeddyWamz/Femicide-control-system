import smtplib
from email.message import EmailMessage
from typing import Optional


class Mailer:
    def __init__(
        self,
        server: str,
        port: int,
        username: str,
        password: str,
        sender: Optional[str] = None,
        use_ssl: bool = True,
    ):
        self.server = server
        self.port = port
        self.username = username
        self.password = password
        self.sender = sender or username
        self.use_ssl = use_ssl

    def send(self, to_email: str, subject: str, body: str):
        if not to_email:
            raise ValueError("Recipient email is required.")
        if not subject.strip():
            raise ValueError("Subject is required.")
        if not body.strip():
            raise ValueError("Message body cannot be empty.")

        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = self.sender
        msg["To"] = to_email
        msg.set_content(body)

        if self.use_ssl:
            with smtplib.SMTP_SSL(self.server, self.port) as smtp:
                smtp.login(self.username, self.password)
                smtp.send_message(msg)
        else:
            with smtplib.SMTP(self.server, self.port) as smtp:
                smtp.starttls()
                smtp.login(self.username, self.password)
                smtp.send_message(msg)
