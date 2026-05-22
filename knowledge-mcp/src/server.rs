use std::sync::Arc;

use rmcp::handler::server::wrapper::Parameters;
use rmcp::model::{CallToolResult, Content, ServerCapabilities, ServerInfo};
use rmcp::transport::streamable_http_server::{
    StreamableHttpService, session::local::LocalSessionManager, tower::StreamableHttpServerConfig,
};
use rmcp::{ErrorData, ServerHandler, tool, tool_handler, tool_router};

use crate::index::HybridIndex;

#[derive(serde::Deserialize, schemars::JsonSchema)]
struct SearchParams {
    /// Natural-language question or keywords to search the project docs.
    query: String,
    /// Max results to return (default 5).
    #[serde(default)]
    k: Option<usize>,
}

#[derive(Clone)]
pub struct DocsServer {
    index: Arc<HybridIndex>,
}

#[tool_router]
impl DocsServer {
    pub fn new(index: Arc<HybridIndex>) -> Self {
        Self { index }
    }

    #[tool(
        description = "Search the project documentation (architecture, RabbitMQ/XML contracts, tool specs) and return the most relevant excerpts with their source file. Use for conceptual 'how does X work?' questions about the system.",
        annotations(title = "Search project docs", read_only_hint = true)
    )]
    async fn search_docs(
        &self,
        Parameters(SearchParams { query, k }): Parameters<SearchParams>,
    ) -> Result<CallToolResult, ErrorData> {
        let k = k.unwrap_or(5).clamp(1, 20);
        let hits = self
            .index
            .search(&query, k)
            .map_err(|e| ErrorData::internal_error(e.to_string(), None))?;
        let body = if hits.is_empty() {
            format!("No relevant documentation found for: {query}")
        } else {
            hits.iter()
                .enumerate()
                .map(|(i, h)| format!("### {}. {}\n{}", i + 1, h.source, h.text.trim()))
                .collect::<Vec<_>>()
                .join("\n\n")
        };
        Ok(CallToolResult::success(vec![Content::text(body)]))
    }
}

#[tool_handler]
impl ServerHandler for DocsServer {
    #[allow(
        clippy::field_reassign_with_default,
        reason = "ServerInfo is #[non_exhaustive]; struct-literal construction is rejected"
    )]
    fn get_info(&self) -> ServerInfo {
        let mut info = ServerInfo::default();
        info.capabilities = ServerCapabilities::builder().enable_tools().build();
        info.instructions =
            Some("Knowledge base over the project docs. Call search_docs with a question.".into());
        info
    }
}

pub async fn serve(index: Arc<HybridIndex>, port: u16) -> anyhow::Result<()> {
    let service = StreamableHttpService::new(
        move || Ok::<_, std::io::Error>(DocsServer::new(index.clone())),
        Arc::new(LocalSessionManager::default()),
        StreamableHttpServerConfig::default(),
    );
    let app = axum::Router::new().nest_service("/mcp", service);
    let addr = format!("0.0.0.0:{port}");
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    eprintln!("knowledge-mcp: serving MCP on http://{addr}/mcp");
    axum::serve(listener, app).await?;
    Ok(())
}
