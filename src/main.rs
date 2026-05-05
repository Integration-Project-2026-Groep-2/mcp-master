use rmcp::{
    ServiceExt,
    transport::{TokioChildProcess, ConfigureCommandExt},
};
use tokio::process::Command;
use std::error::Error;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {

    let transport = TokioChildProcess::new(
        Command::new("npx").configure(|cmd| {
            cmd.arg("-y")
                .arg("@modelcontextprotocol/server-everything");
            }),
    )?;

    let client = ().serve(transport).await?;

    let tools = client.list_tools(None).await?;

    // note(nasr): debug, list all tools to stdio
    for t in &tools.tools {
        println!("- {}", t.name);
    }

    if tools.tools.iter().any(|t| t.name == "heartbeat_status") {
        use serde_json::json;

        let mut map = serde_json::Map::new();
        map.insert("service".to_string(), json!(""));
        map.insert("limit".to_string(), json!(5));

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

    tokio::signal::ctrl_c().await?;

    Ok(())
}
