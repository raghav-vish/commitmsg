# 📝 AI Commit Message Generator

Generate Git commit messages with **Ollama** or **Gemini**, using your staged changes (or just repo/branch context).

---

## ⚡ Setup

Clone and make executable:

```bash
https://github.com/raghav-vish/commitmsg.git
chmod +x <repo>/commitmsg.py
```

Add an alias (put this in `~/.zshrc` or `~/.bashrc`):

```bash
alias commitmsg="/full/path/to/<repo>/commitmsg.py"
```

Reload your shell:

```bash
source ~/.zshrc
```

---

## 🚀 Usage

From any git repo:

```bash
commitmsg [options]
```

Examples:

```bash
commitmsg                      # default (Ollama, normal style)
commitmsg --provider gemini    # use Gemini
commitmsg --style serious      # style control
commitmsg --ignorediff         # ignore diff, use branch/repo only
commitmsg --commit             # confirm + commit
```

---

## 📦 Requirements

* Python + `pip install ollama google-genai`
* Git
* Ollama (if using local models)
* Google Cloud creds (if using Gemini)
