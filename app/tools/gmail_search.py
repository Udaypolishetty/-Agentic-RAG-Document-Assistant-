import os
from pathlib import Path

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build


SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly"
]

BASE_DIR = Path(__file__).resolve().parents[2]

CREDENTIALS_FILE = BASE_DIR / "credentials.json"
TOKEN_FILE = BASE_DIR / "token.json"


def get_gmail_service():

    creds = None

    if TOKEN_FILE.exists():

        creds = Credentials.from_authorized_user_file(
            TOKEN_FILE,
            SCOPES
        )

    if not creds or not creds.valid:

        if creds and creds.expired and creds.refresh_token:

            creds.refresh(Request())

        else:

            flow = InstalledAppFlow.from_client_secrets_file(
                CREDENTIALS_FILE,
                SCOPES
            )

            creds = flow.run_local_server(
                port=0
            )

        TOKEN_FILE.write_text(
            creds.to_json(),
            encoding="utf-8"
        )

    return build(
        "gmail",
        "v1",
        credentials=creds
    )


def search_gmail(query, max_results=5):

    service = get_gmail_service()

    response = service.users().messages().list(
        userId="me",
        q=query,
        maxResults=max_results
    ).execute()

    messages = response.get(
        "messages",
        []
    )

    results = []

    for message in messages:

        data = service.users().messages().get(
            userId="me",
            id=message["id"],
            format="metadata",
            metadataHeaders=[
                "Subject",
                "From",
                "Date"
            ]
        ).execute()

        headers = data.get(
            "payload",
            {}
        ).get(
            "headers",
            []
        )

        header_map = {
            h["name"]: h["value"]
            for h in headers
        }

        results.append({

            "id": message["id"],

            "subject": header_map.get(
                "Subject",
                ""
            ),

            "from": header_map.get(
                "From",
                ""
            ),

            "date": header_map.get(
                "Date",
                ""
            )
        })

    return results

if __name__ == "__main__":

    results = search_gmail(
        "newer_than:30d"
    )

    for result in results:

        print("\nSubject:", result["subject"])
        print("From:", result["from"])
        print("Date:", result["date"])