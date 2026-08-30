from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Project Tools")


@mcp.tool()
def get_project_info(project_name: str) -> str:
    """Get information about a project."""

    if project_name.lower() == "phoenix":
        return (
            "Project Phoenix is an internal payment processing application. "
            "The backend uses Python and FastAPI. "
            "The database is PostgreSQL. "
            "The frontend uses React. "
            "The migration deadline is September 15, 2026."
        )

    return f"No information found for project: {project_name}"


if __name__ == "__main__":
    mcp.run()