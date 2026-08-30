import requests


def get_github_status(repo):
    """Get current information about a GitHub repository."""

    url = f"https://api.github.com/repos/{repo}"

    response = requests.get(url)

    if response.status_code != 200:
        return f"Unable to retrieve GitHub repository: {repo}"

    data = response.json()

    return {
        "name": data["full_name"],
        "description": data["description"],
        "stars": data["stargazers_count"],
        "open_issues": data["open_issues_count"],
        "updated_at": data["updated_at"]
    }
# if __name__ == "__main__":
#     print(get_github_status("openai/openai-python"))