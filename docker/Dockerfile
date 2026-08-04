FROM lmsysorg/sglang-omni:dev AS base


# install UCX
RUN git clone https://github.com/openucx/ucx.git \
    && cd ucx \
    && git checkout v1.20.x \
    && ./autogen.sh \
    && ./contrib/configure-release-mt  \
    --enable-shared                    \
    --disable-static                   \
    --disable-doxygen-doc              \
    --enable-optimizations             \
    --enable-cma                       \
    --enable-devel-headers             \
    --with-cuda=/usr/local/cuda        \
    --with-verbs                       \
    --with-dm                          \
    --prefix=/usr/local                \
    && make -j \
    && make -j install-strip \
    && ldconfig

# Install the repository and its pinned runtime dependencies into the image so
# containers do not depend on an ad-hoc host-side pip install.
WORKDIR /opt/sglang-omni
COPY . /opt/sglang-omni
RUN uv pip install --system --break-system-packages --no-cache --index-url https://pypi.org/simple -e .

ENV SGLANG_JIT_DEEPGEMM_PRECOMPILE=0 \
    FLASHINFER_DISABLE_VERSION_CHECK=1 \
    TOKENIZERS_PARALLELISM=false \
    PYTHONUNBUFFERED=1
