FROM rust:latest as builder
RUN USER=root cargo new --bin mcp-master

WORKDIR /mcp-master

COPY ./Cargo.toml ./Cargo.toml
RUN cargo build --release
RUN rm src/*.rs ./target/release/deps/mcp_master*
ADD . ./
RUN cargo build --release

ARG APP=/app
ARG APP_USER=nasr

RUN groupadd $APP_USER && useradd -g $APP_USER $APP_USER && mkdir -p $APP
RUN cp /mcp-master//target/release/mcp-master $APP/mcp-master

USER $USER
WORKDIR $APP
