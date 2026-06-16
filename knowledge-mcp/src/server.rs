use std::sync::Arc;

use rmcp::handler::server::wrapper::Parameters;
use rmcp::model::{CallToolResult, Content, ServerCapabilities, ServerInfo};
use rmcp::transport::streamable_http_server::{
    StreamableHttpService, session::local::LocalSessionManager, tower::StreamableHttpServerConfig,
};
use rmcp::{ErrorData, ServerHandler, tool, tool_handler, tool_router};
use tower_http::limit::RequestBodyLimitLayer;

use crate::index::HybridIndex;

const MAX_BODY_BYTES: usize = 1 << 20;
const MAX_QUERY_CHARS: usize = 8192;

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
        if query.len() > MAX_QUERY_CHARS {
            return Err(ErrorData::invalid_params(
                format!("query too long (max {MAX_QUERY_CHARS} chars)"),
                None,
            ));
        }
        let k = k.unwrap_or(5).clamp(1, 20);
        let index = self.index.clone();
        let q = query.clone();
        let hits = tokio::task::spawn_blocking(move || index.search(&q, k))
            .await
            .map_err(|e| ErrorData::internal_error(format!("search task failed: {e}"), None))?
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

fn build_config() -> StreamableHttpServerConfig {
    // Pure request/response tool server — stateless avoids per-session task/alloc.
    let config = StreamableHttpServerConfig::default().with_stateful_mode(false);
    match std::env::var("ALLOWED_HOSTS") {
        Ok(v) if !v.trim().is_empty() => {
            let hosts: Vec<String> = v
                .split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect();
            eprintln!("knowledge-mcp: ALLOWED_HOSTS = {hosts:?}");
            config.with_allowed_hosts(hosts)
        }
        // rmcp's default Host allow-list is a browser DNS-rebinding guard; this is an
        // internal server-to-server service, so the network policy is the control and
        // the default localhost-only list would 403 every in-cluster request.
        _ => {
            eprintln!(
                "knowledge-mcp: ALLOWED_HOSTS unset — Host check disabled (network policy is the boundary)"
            );
            config.disable_allowed_hosts()
        }
    }
}

pub async fn serve(index: Arc<HybridIndex>, port: u16) -> anyhow::Result<()> {
    let service = StreamableHttpService::new(
        move || Ok::<_, std::io::Error>(DocsServer::new(index.clone())),
        Arc::new(LocalSessionManager::default()),
        build_config(),
    );
    let app = axum::Router::new()
        .nest_service("/mcp", service)
        .layer(RequestBodyLimitLayer::new(MAX_BODY_BYTES));
    let addr = format!("0.0.0.0:{port}");
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    eprintln!("knowledge-mcp: serving MCP on http://{addr}/mcp");
    axum::serve(listener, app).await?;
    Ok(())
}
