use rmcp::{
    ServiceExt,
    transport::StreamableHttpClientTransport,
};

use url::Url;
use std::error::Error;
use std::env;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {

    let base_url = env::var("MCP_BASE_URL").expect("MCP_BASE_URL must be set");
    let port = env::var("MCP_PORT").unwrap_or_else(|_| "5555".to_string());

    let controlroom_url = format!("{}:{}", base_url, port);

    let server_url = Url::parse(&controlroom_url)?;
    let transport = StreamableHttpClientTransport::from_uri(server_url.as_str());

    let client = ().serve(transport).await?;

    let tools = client.list_tools(None).await?;
    for t in &tools.tools {
        println!("- {}", t.name);
    }

    if tools.tools.iter().any(|t| t.name == "heartbeat_status") {
        use serde_json::json;
        let mut map = serde_json::Map::new();
        map.insert("service".to_string(), json!("elasticsearch"));

        let result = client
            .call_tool(rmcp::model::CallToolRequestParams {
                meta: None,
                task: None,
                name: "heartbeat_status".into(),
                arguments: Some(map),
            })
            .await?;

        println!("Tool result: {:?}", result);
    }

    Ok(())
}
