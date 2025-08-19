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


def build_prompt(diff: str, style: str) -> str:
    """Construct a style-aware prompt for commit message generation."""
    return (
        "You are a Git commit message generator.\n"
        "Rules:\n"
        "- Only output the commit message (no explanations, no extra text).\n"
        "- Must be a single short line.\n"
        f"- Style: {style}.\n\n"
        f"Diff:\n{diff}"
    )


def generate_with_ollama(diff: str, style: str, temperature: float, model: str) -> str:
    """Generate commit message using Ollama."""
    if ollama is None:
        raise RuntimeError("Ollama not installed. Run `pip install ollama`.")

    prompt = build_prompt(diff, style)
    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": temperature},
    )
    return response["message"]["content"].strip()


def generate_with_gemini(diff: str, style: str, temperature: float, model_name: str) -> str:
    """Generate commit message using Gemini (genai SDK)."""
    if genai is None:
        raise RuntimeError("Google genai SDK not installed. Run `pip install google-genai`.")

    client = genai.Client()
    prompt = build_prompt(diff, style)

    response = client.models.generate_content(
        model=model_name,
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
    try:
        subprocess.run(
            ["git", "commit", "-m", message],
            check=True
        )
        print("\n✅ Commit successful!")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Failed to commit: {e}")


def main():
    parser = argparse.ArgumentParser(description="AI-powered Git commit message generator")
    parser.add_argument("--provider", choices=["ollama", "gemini"], default="ollama",
                        help="Which provider to use (default: ollama)")
    parser.add_argument("--style", default="normal",
                        help="Message style (normal, humour, serious, poetic, etc.)")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature (creativity). Default = 0.7")
    parser.add_argument("--model", default=None,
                        help="Model to use (default depends on provider: Ollama=gemma3:4b, Gemini=gemini-2.0-flash-lite)")
    parser.add_argument("--commit", action="store_true",
                        help="Actually run `git commit -m <msg>` with the generated message")
    args = parser.parse_args()

    diff = get_git_diff()
    if not diff.strip():
        print("No staged changes found.")
        sys.exit(0)

    # pick default models if none specified
    if args.model is None:
        args.model = "gemma3:4b" if args.provider == "ollama" else "gemini-2.0-flash-lite"

    print(f"Using provider: {args.provider}, model: {args.model}, style: {args.style}, temperature: {args.temperature}")

    if args.provider == "ollama":
        commit_message = generate_with_ollama(diff, args.style, args.temperature, args.model)
    else:
        commit_message = generate_with_gemini(diff, args.style, args.temperature, args.model)

    print("\nSuggested commit message:\n")
    print(commit_message)

    if args.commit:
        confirm = input("\nDo you want to commit with this message? [y/N]: ").strip().lower()
        if confirm == "y":
            run_git_commit(commit_message)
        else:
            print("❌ Commit aborted.")


if __name__ == "__main__":
    main()
