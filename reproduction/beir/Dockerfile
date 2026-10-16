FROM python:3.12-slim-trixie

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64 \
    PYSERINI_CACHE=/workspace/.cache/pyserini

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        openjdk-21-jdk-headless \
        libgomp1 \
        fish \
        neovim \
        fzf \
        git \
        ripgrep \
    && rm -rf /var/lib/apt/lists/*

# Mount the host dotfiles repository at /root/dotfiles when running.
# Keep the config directories writable for fish_variables and editor state.
RUN mkdir -p /root/.config/fish /root/.config/nvim \
    && ln -s /root/dotfiles/fish/config.fish /root/.config/fish/config.fish \
    && ln -s /root/dotfiles/nvim/init.lua /root/.config/nvim/init.lua \
    && ln -s /root/dotfiles/nvim/snippets /root/.config/nvim/snippets

# Install the CPU build first so Pyserini does not pull CUDA dependencies.
RUN python -m pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu \
        torch \
    && python -m pip install --no-cache-dir pyserini==2.4.0

WORKDIR /workspace
CMD ["fish"]
