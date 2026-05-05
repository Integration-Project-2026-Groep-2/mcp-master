FROM rust:1.78 as builder

WORKDIR /app
COPY . .
RUN cargo build --release


FROM alpine:3.20

RUN apk add --no-cache ca-certificates

WORKDIR /app
COPY --from=builder /app/target/release/mcp-master .

CMD ["./mcp-master"]
