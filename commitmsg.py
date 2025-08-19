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


def build_prompt(diff: str, style: str, branch: str, repo: str) -> str:
    """Construct a style-aware prompt for commit message generation with context."""
    return (
        "You are a Git commit message generator.\n"
        "Rules:\n"
        "- Only output the commit message (no explanations, no extra text).\n"
        "- Must be a single short line.\n"
        f"- Style: {style}.\n"
        f"- Repository: {repo}.\n"
        f"- Branch: {branch}.\n\n"
        f"Diff:\n{diff}"
    )


def generate_with_ollama(diff: str, style: str, temperature: float,
                         model: str, branch: str, repo: str) -> str:
    """Generate commit message using Ollama."""
    if ollama is None:
        raise RuntimeError("Ollama not installed. Run `pip install ollama`.")

    prompt = build_prompt(diff, style, branch, repo)
    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": temperature},
    )
    return response["message"]["content"].strip()


def generate_with_gemini(diff: str, style: str, temperature: float,
                         model: str, branch: str, repo: str) -> str:
    """Generate commit message using Gemini (genai SDK)."""
    if genai is None:
        raise RuntimeError("Google genai SDK not installed. Run `pip install google-genai`.")

    client = genai.Client()
    prompt = build_prompt(diff, style, branch, repo)

    response = client.models.generate_content(
        model=model,
        contents=[prompt],
        config=types.GenerateContentConfig(
            temperature=temperature,
            top_p=1,
            top_k=40,
        )
    )
    return response.text.strip()


def run_git_commit(message: str):
    """Run git commit with the generated message."""
    subprocess.run(["git", "commit", "-m", message])


def main():
    parser = argparse.ArgumentParser(description="AI-powered Git commit message generator")
    parser.add_argument("--provider", choices=["ollama", "gemini"], default="ollama",
                        help="Which provider to use (default: ollama)")
    parser.add_argument("--style", default="normal",
                        help="Message style (normal, humour, serious, poetic, etc.)")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature (creativity). Default = 0.7")
    parser.add_argument("--model", type=str,
                        default="gemma3:4b",
                        help="Model to use (default for ollama: gemma3:4b, for gemini: gemini-2.0-flash-lite)")
    parser.add_argument("--commit", action="store_true",
                        help="If set, ask for confirmation before committing with the generated message")
    args = parser.parse_args()

    diff = get_git_diff()
    if not diff.strip():
        print("No staged changes found.")
        sys.exit(0)

    branch = get_git_branch()
    repo = get_repo_name()

    # Default models per provider
    if args.provider == "gemini" and args.model == "gemma3:4b":
        args.model = "gemini-2.0-flash-lite"

    print(f"Using provider: {args.provider}, model: {args.model}, style: {args.style}, "
          f"temperature: {args.temperature}, branch: {branch}, repo: {repo}")

    if args.provider == "ollama":
        commit_message = generate_with_ollama(diff, args.style, args.temperature, args.model, branch, repo)
    else:
        commit_message = generate_with_gemini(diff, args.style, args.temperature, args.model, branch, repo)

    print("\nSuggested commit message:\n")
    print(commit_message)

    if args.commit:
        confirm = input("\nDo you want to commit with this message? [y/N]: ").strip().lower()
        if confirm == "y":
            run_git_commit(commit_message)
            print("\n✅ Changes committed.")
        else:
            print("\n❌ Commit aborted.")


if __name__ == "__main__":
    main()
