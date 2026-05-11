//! Node + edge types for the `/architecture` HTTP endpoint.
//!
//! Wire-format is Cytoscape.js-friendly: a flat list of nodes (each with a
//! stable `id` + a `kind` discriminator + free-form `metadata`) and a flat
//! list of edges referencing nodes by `id`. Drupal-side renders directly
//! without further transformation.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Discriminator for what a node represents. Drives Drupal-side styling.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum NodeKind {
    /// mcp-master itself — the agent at the center.
    Master,
    /// HTTP consumer of mcp-master (e.g. Drupal jarvis_chat).
    Consumer,
    /// One of the team MCP-servers (CRM-MCP, Controlroom-MCP).
    McpServer,
    /// A team backend service (kassa, mailing, ...) — heartbeat producer.
    Service,
    /// External system mcp-master or another node talks to (Anthropic, SF).
    External,
    /// The RabbitMQ broker as a logical group.
    Broker,
    /// One RabbitMQ exchange (topic / direct / fanout).
    Exchange,
    /// One RabbitMQ queue.
    Queue,
}

/// One graph node. `id` is the slug-shaped stable key used in edges.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Node {
    pub id: String,
    pub label: String,
    pub kind: NodeKind,
    /// Free-form per-kind metadata: tool inventory for `mcp_server`, URL for
    /// `external`, exchange-type for `exchange`, etc.
    pub metadata: serde_json::Value,
}

/// Discriminator for edge semantics. Drives Drupal-side line-styling.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum EdgeKind {
    /// HTTPS / HTTP (REST or JSON-RPC over HTTP).
    Http,
    /// MCP-protocol over Streamable HTTP (mcp-master → mcp-server).
    Mcp,
    /// Publisher → exchange.
    AmqpPublish,
    /// Queue → consumer.
    AmqpConsume,
    /// Exchange → queue (with optional routing-key).
    AmqpBinding,
    /// MCP-server → external API (e.g. CRM-MCP → Salesforce REST).
    External,
}

/// One directed graph edge.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Edge {
    pub source: String,
    pub target: String,
    pub kind: EdgeKind,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
}

/// Top-level response shape for `GET /architecture`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArchitectureResponse {
    pub nodes: Vec<Node>,
    pub edges: Vec<Edge>,
    pub generated_at: DateTime<Utc>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn architecture_response_round_trips_via_serde() {
        let response = ArchitectureResponse {
            nodes: vec![Node {
                id: "self.mcp-master".into(),
                label: "mcp-master".into(),
                kind: NodeKind::Master,
                metadata: json!({}),
            }],
            edges: vec![Edge {
                source: "consumer.frontend".into(),
                target: "self.mcp-master".into(),
                kind: EdgeKind::Http,
                label: Some("POST /chat".into()),
            }],
            generated_at: Utc::now(),
        };
        let json = serde_json::to_string(&response).expect("serializes");
        let parsed: ArchitectureResponse = serde_json::from_str(&json).expect("deserializes");
        assert_eq!(parsed.nodes, response.nodes);
        assert_eq!(parsed.edges, response.edges);
    }

    #[test]
    fn edge_omits_label_when_none() {
        let edge = Edge {
            source: "a".into(),
            target: "b".into(),
            kind: EdgeKind::Mcp,
            label: None,
        };
        let v = serde_json::to_value(&edge).unwrap();
        assert!(
            v.get("label").is_none(),
            "label should be skipped when None"
        );
    }

    #[test]
    fn node_kind_serializes_snake_case() {
        let json = serde_json::to_string(&NodeKind::McpServer).unwrap();
        assert_eq!(json, "\"mcp_server\"");
    }
}
