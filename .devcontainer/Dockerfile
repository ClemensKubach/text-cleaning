FROM ubuntu:latest

ARG USER=developer
ARG DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN apt update \
    && apt install -y --no-install-recommends sudo git zsh curl wget ca-certificates \
    && apt clean \
    && apt autoremove -y \
    && rm -rf /var/lib/apt/lists/*

# Create user
RUN useradd -m -s /usr/bin/zsh ${USER} \
    && echo "${USER} ALL=(ALL) NOPASSWD: ALL" >/etc/sudoers.d/${USER} \
    && chmod 0440 /etc/sudoers.d/${USER}

# Switch to new user
USER ${USER}
ENV HOME=/home/${USER}
WORKDIR ${HOME}

# Install Oh My Zsh
RUN sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"

# Setup plugins
ENV ZSH_CUSTOM=${HOME}/.oh-my-zsh/custom
RUN mkdir -p ${ZSH_CUSTOM}/plugins && \
    git clone https://github.com/zsh-users/zsh-autosuggestions ${ZSH_CUSTOM}/plugins/zsh-autosuggestions && \
    git clone https://github.com/zsh-users/zsh-syntax-highlighting.git ${ZSH_CUSTOM}/plugins/zsh-syntax-highlighting && \
    sed -i.bak 's/plugins=(\(.*\))/plugins=(\1 zsh-autosuggestions zsh-syntax-highlighting)/' ${HOME}/.zshrc

# Install spaceship prompt
ENV SPACESHIP_REMOTE=https://github.com/spaceship-prompt/spaceship-prompt.git
RUN git clone ${SPACESHIP_REMOTE} ${ZSH_CUSTOM}/themes/spaceship-prompt --depth=1 && \
    ln -s ${ZSH_CUSTOM}/themes/spaceship-prompt/spaceship.zsh-theme ${ZSH_CUSTOM}/themes/spaceship.zsh-theme && \
    sed -i 's|ZSH_THEME="robbyrussell"|ZSH_THEME="spaceship"|g' ${HOME}/.zshrc

# Install uv and add completions
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    echo 'eval "$(uv generate-shell-completion zsh)"' >> ${HOME}/.zshrc && \
    echo 'eval "$(uvx --generate-shell-completion zsh)"' >> ${HOME}/.zshrc