FROM rust:1.95-bookworm AS builder

RUN USER=root cargo new --bin mcp-master
WORKDIR /mcp-master

COPY ./Cargo.toml ./Cargo.toml
COPY ./Cargo.lock ./Cargo.lock

RUN cargo build --release
RUN rm src/*.rs target/release/deps/mcp_master*

COPY ./src ./src
RUN cargo build --release

FROM debian:bookworm-slim

RUN apt-get update \
        && apt-get install -y --no-install-recommends ca-certificates \
        && rm -rf /var/lib/apt/lists/*


ARG APP=/app
ARG APP_USER=nasr

RUN groupadd $APP_USER \
 && useradd -g $APP_USER $APP_USER \
 && mkdir -p $APP

COPY --from=builder /mcp-master/target/release/mcp-master $APP/mcp-master

USER $APP_USER
WORKDIR $APP
# CMD ["./mcp-master", "--terminal-mode"]
CMD ["sleep", "infinity"]
