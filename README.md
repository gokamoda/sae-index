

# Usage

- Install dependencies:
    ```bash
    make install
    ```
- Run the greet command:
    ```bash
    make greet
    ```

- Run greet with overrides:
    ```bash
    uv run greet --config configs/debug.yaml --override debug.message=hello
    ```

# SFTP Configuration

```
{
	"name": "SFTP",
	"protocol": "sftp",
	"profiles": {
		"sh100": {
			"host": "sh100",
			"remotePath": "/home/{user}/template"
		},
		"sa6000": {
			"host": "sa6000",
			"remotePath": "/home/{user}/template"
		}
	},
	"agent": "$SSH_AUTH_SOCK",
	"uploadOnSave": true,
	"ignore": [
		".venv",
		".vscode",
		".git",
		".DS_Store",
		"__pycache__",
		".python-version",
		".env",
		".ruff_cache",
		"*.png",
		"*.jpg",
		"*.pdf",
		"*.log",
		"uv.lock",
		"*.pt",
		"-info"
	]
}
```