use anyhow::{Context, Result};
use rmcp::{
    ServiceExt,
    model::{CallToolRequestParams, ClientCapabilities, ClientInfo, Implementation},
    transport::StreamableHttpClientTransport,
};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info,mcp_master=debug".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    let base_url = std::env::var("MCP_BASE_URL").unwrap_or_else(|_| "http://localhost".to_string());
    let port = std::env::var("MCP_PORT").unwrap_or_else(|_| "7002".to_string());
    let url = format!("{base_url}:{port}/mcp");
    tracing::info!(%url, "connecting to controlroom-mcp");

    let transport = StreamableHttpClientTransport::from_uri(url);
    let client_info = ClientInfo::new(
        ClientCapabilities::default(),
        Implementation::new("mcp-master", env!("CARGO_PKG_VERSION")),
    );

    let client = client_info.serve(transport).await?;
    tracing::info!(server = ?client.peer_info(), "MCP session initialized");

    let tools = client.list_tools(Default::default()).await?;
    for t in &tools.tools {
        tracing::info!(name = %t.name, description = ?t.description, "tool");
    }

    if tools.tools.iter().any(|t| t.name == "heartbeat_status") {
        let args = serde_json::json!({ "limit": 5 })
            .as_object()
            .cloned()
            .context("heartbeat_status arguments must serialize to a JSON object")?;
        let result = client
            .call_tool(CallToolRequestParams::new("heartbeat_status").with_arguments(args))
            .await?;
        tracing::info!(?result, "heartbeat_status result");
    }

    client.cancel().await?;
    Ok(())
}
