# Local Git Hooks

This directory contains local Git hooks. They are not enabled by default.

## Activation

To enable these hooks for your local repository, run the following command once:

```bash
git config core.hooksPath .githooks
```

This will configure Git to use the hooks in this directory. The setting is local to this repository and will not affect your global Git configuration.
