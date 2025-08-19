#!/Users/raghavvishwanath/Personal/commitmsg/commitmsg_venv/bin/python3

import subprocess
import sys
import argparse


# --- Ollama ---
try:
    import ollama
except ImportError:
    ollama = None

# --- Gemini genai SDK ---
try:
    import google.genai as genai
    from google.genai import types
except ImportError:
    genai = None
    types = None


def get_git_diff() -> str:
    """Get the staged git diff."""
    return subprocess.run(
        ["git", "diff", "--cached"],
        capture_output=True,
        text=True,
    ).stdout


def get_git_branch() -> str:
    """Get the current git branch name."""
    result = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() or "unknown-branch"


def get_repo_name() -> str:
    """Get the repo's top-level directory name."""
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
    )
    path = result.stdout.strip()
    return path.split("/")[-1] if path else "unknown-repo"


def build_prompt(
    style: str, branch: str, repo: str, diff: str = None, ignorediff: bool = False
) -> str:
    """Construct a detailed prompt for commit message generation with context."""

    base = (
        "You are an expert Git commit message generator.\n"
        "Rules:\n"
        "- Output ONLY the commit message, nothing else.\n"
        "- Must be a **single concise line**\n"
        f"- Style: {style} (adapt tone accordingly).\n"
        f"- Repository: {repo} (for context, not required in output).\n"
        f"- Branch: {branch} (for context, not required in output).\n"
    )

    if not ignorediff and diff:
        base += (
            "\nHere is the staged diff:\n"
            f"{diff}\n"
            "\nFocus on summarizing WHAT and WHY, not HOW.\n"
            "- Capture the main intent (e.g., feature, bugfix, refactor).\n"
            "- Use file names, function/class names, and key changes if relevant.\n"
        )
    else:
        base += (
            "\nNo diff provided.\n"
            "Generate a commit message based only on repo, branch, and style context.\n"
        )

    return base



def generate_with_ollama(
    style: str,
    temperature: float,
    model: str,
    branch: str,
    repo: str,
    diff: str = None,
    ignorediff: bool = False,
) -> str:
    """Generate commit message using Ollama."""
    if ollama is None:
        raise RuntimeError("Ollama not installed. Run `pip install ollama`.")

    prompt = build_prompt(style, branch, repo, diff, ignorediff)
    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": temperature},
    )
    return response["message"]["content"].strip()


def generate_with_gemini(
    style: str,
    temperature: float,
    model: str,
    branch: str,
    repo: str,
    diff: str = None,
    ignorediff: bool = False,
) -> str:
    """Generate commit message using Gemini (genai SDK)."""
    if genai is None:
        raise RuntimeError(
            "Google genai SDK not installed. Run `pip install google-genai`."
        )

    client = genai.Client()
    prompt = build_prompt(style, branch, repo, diff, ignorediff)

    response = client.models.generate_content(
        model=model,
        contents=[prompt],
        config=types.GenerateContentConfig(
            temperature=temperature,
            top_p=1,
            top_k=40,
        ),
    )
    return response.text.strip()


def run_git_commit(message: str):
    """Run git commit with the generated message."""
    subprocess.run(["git", "commit", "-m", message])


def main():
    parser = argparse.ArgumentParser(
        description="AI-powered Git commit message generator"
    )
    parser.add_argument(
        "--provider",
        choices=["ollama", "gemini"],
        default="ollama",
        help="Which provider to use (default: ollama)",
    )
    parser.add_argument(
        "--style",
        default="normal",
        help="Message style (normal, humour, serious, poetic, etc.)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.5,
        help="Sampling temperature (creativity). Default = 1.5",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemma3:4b",
        help="Model to use (default for ollama: gemma3:4b, for gemini: gemini-2.0-flash-lite)",
    )
    parser.add_argument(
        "--commit",
        action="store_true",
        help="If set, ask for confirmation before committing with the generated message",
    )
    parser.add_argument(
        "--ignorediff",
        action="store_true",
        help="Ignore the git diff and generate a commit message from context only",
    )
    args = parser.parse_args()

    diff = None if args.ignorediff else get_git_diff()
    if not args.ignorediff and (not diff or not diff.strip()):
        print("No staged changes found.")
        sys.exit(0)

    branch = get_git_branch()
    repo = get_repo_name()

    # Default models per provider
    if args.provider == "gemini" and args.model == "gemma3:4b":
        args.model = "gemini-2.0-flash-lite"

    print(
        f"Using provider: {args.provider}, model: {args.model}, style: {args.style}, "
        f"temperature: {args.temperature}, branch: {branch}, repo: {repo}, ignorediff: {args.ignorediff}"
    )

    if args.provider == "ollama":
        commit_message = generate_with_ollama(
            args.style,
            args.temperature,
            args.model,
            branch,
            repo,
            diff,
            args.ignorediff,
        )
    else:
        commit_message = generate_with_gemini(
            args.style,
            args.temperature,
            args.model,
            branch,
            repo,
            diff,
            args.ignorediff,
        )

    print("\nSuggested commit message:\n")
    print(commit_message)

    if args.commit:
        confirm = (
            input("\nDo you want to commit with this message? [y/N]: ").strip().lower()
        )
        if confirm == "y":
            run_git_commit(commit_message)
            print("\n✅ Changes committed.")
        else:
            print("\n❌ Commit aborted.")


if __name__ == "__main__":
    main()
