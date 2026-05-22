# ---- builder ----
FROM rust:1-bookworm AS builder
WORKDIR /build
RUN apt-get update \
 && apt-get install -y --no-install-recommends pkg-config libssl-dev \
 && rm -rf /var/lib/apt/lists/*
# Cache dependency compilation before copying the real source.
COPY Cargo.toml Cargo.lock ./
RUN mkdir src && echo 'fn main() {}' > src/main.rs \
 && cargo build --release \
 && rm -rf src
COPY src ./src
RUN cargo build --release

# ---- runtime ----
FROM debian:bookworm-slim AS runtime
# ca-certificates: HTTPS for the one-time model prefetch; libgomp1: ONNX Runtime OpenMP dep.
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates libgomp1 \
 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY --from=builder /build/target/release/knowledge-mcp /app/knowledge-mcp
COPY corpus /corpus
# Bake the embedding model into the image: no runtime download, fast readiness.
# Also fails the build early if the binary can't run here (e.g. missing ONNX runtime).
RUN /app/knowledge-mcp --warm
ENV CORPUS_DIR=/corpus PORT=7099
EXPOSE 7099
ENTRYPOINT ["/app/knowledge-mcp"]
