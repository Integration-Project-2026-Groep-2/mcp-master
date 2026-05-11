//! Assemble the full architecture-graph from live and static sources.
//!
//! Inputs:
//! - `McpPool::catalog()` — connected MCP-servers + their tool inventories
//! - `ManagementClient::fetch_topology()` — exchanges, queues, bindings
//! - Compile-time `statics::{SERVICES, EXTERNALS, STATIC_EDGES}` — entities
//!   that have no live discovery mechanism
//!
//! Output: `ArchitectureResponse` ready for Cytoscape.js consumption.

use anyhow::Result;
use chrono::Utc;
use serde_json::json;

use super::dto::{ArchitectureResponse, Edge, EdgeKind, Node, NodeKind};
use super::rabbitmq::{ManagementClient, RabbitMqTopology};
use super::statics::{EXTERNALS, SERVICES, STATIC_EDGES};
use crate::mcp::McpPool;

pub const SELF_NODE_ID: &str = "self.mcp-master";
pub const CONSUMER_FRONTEND_ID: &str = "consumer.frontend";
pub const BROKER_NODE_ID: &str = "broker.rabbitmq";

/// Build the full graph by combining all sources. The RabbitMQ management
/// API call is the only async leg — everything else is synchronous data
/// movement. Failure of the mgmt API call propagates as `Err`.
pub async fn build_architecture(
    pool: &McpPool,
    mgmt_client: &ManagementClient,
) -> Result<ArchitectureResponse> {
    let catalog = pool.catalog();
    let topology = mgmt_client.fetch_topology().await?;
    Ok(build_from_parts(&catalog, &topology))
}

/// Pure assembly — split out so tests can construct fixtures directly
/// without spinning up real MCP-sessions or HTTP mocks.
pub(crate) fn build_from_parts(
    catalog: &crate::mcp::CatalogResponse,
    topology: &RabbitMqTopology,
) -> ArchitectureResponse {
    let mut nodes: Vec<Node> = Vec::new();
    let mut edges: Vec<Edge> = Vec::new();

    // 1. Master + consumer (anchors at top of the graph)
    nodes.push(master_node());
    nodes.push(consumer_frontend_node());

    // 2. MCP-server nodes from runtime catalog
    for server in &catalog.servers {
        nodes.push(mcp_server_node(server));
        edges.push(Edge {
            source: SELF_NODE_ID.to_string(),
            target: mcp_server_id(&server.label),
            kind: EdgeKind::Mcp,
            label: Some("MCP".to_string()),
        });
    }

    // 3. Team service nodes (status=unknown until PR 2.5 lands)
    for svc in SERVICES {
        nodes.push(service_node(svc));
    }

    // 4. External system nodes
    for ext in EXTERNALS {
        nodes.push(Node {
            id: ext.id.to_string(),
            label: ext.label.to_string(),
            kind: NodeKind::External,
            metadata: json!({ "url": ext.url }),
        });
    }

    // 5. Broker + exchanges + queues from RabbitMQ topology
    nodes.push(broker_node());
    for ex in &topology.exchanges {
        if is_system_exchange(&ex.name) {
            continue;
        }
        nodes.push(exchange_node(&ex.name, &ex.kind));
    }
    for q in &topology.queues {
        nodes.push(queue_node(&q.name));
    }

    // 6. Edges from bindings (exchange → queue/exchange)
    for binding in &topology.bindings {
        if is_system_exchange(&binding.source) {
            continue;
        }
        let source = exchange_id(&binding.source);
        let target = match binding.destination_type.as_str() {
            "queue" => queue_id(&binding.destination),
            "exchange" => exchange_id(&binding.destination),
            _ => continue, // skip unknown destination kinds
        };
        edges.push(Edge {
            source,
            target,
            kind: EdgeKind::AmqpBinding,
            label: if binding.routing_key.is_empty() {
                None
            } else {
                Some(binding.routing_key.clone())
            },
        });
    }

    // 7. Static edges (consumer→master, master→externals, services→broker, …).
    // Filter out edges whose endpoints aren't in the assembled node-set —
    // STATIC_EDGES references both `mcp.crm` and `mcp.controlroom`, but a
    // pool with only one of them connected would otherwise emit dangling
    // edges.
    let node_ids: std::collections::HashSet<&str> = nodes.iter().map(|n| n.id.as_str()).collect();
    for se in STATIC_EDGES {
        if !node_ids.contains(se.source) || !node_ids.contains(se.target) {
            continue;
        }
        edges.push(Edge {
            source: se.source.to_string(),
            target: se.target.to_string(),
            kind: se.kind.clone(),
            label: se.label.map(|s| s.to_string()),
        });
    }

    ArchitectureResponse {
        nodes,
        edges,
        generated_at: Utc::now(),
    }
}

// --- Node builders ---

fn master_node() -> Node {
    Node {
        id: SELF_NODE_ID.to_string(),
        label: "mcp-master".to_string(),
        kind: NodeKind::Master,
        metadata: json!({ "version": env!("CARGO_PKG_VERSION") }),
    }
}

fn consumer_frontend_node() -> Node {
    Node {
        id: CONSUMER_FRONTEND_ID.to_string(),
        label: "Drupal Frontend".to_string(),
        kind: NodeKind::Consumer,
        metadata: json!({}),
    }
}

fn broker_node() -> Node {
    Node {
        id: BROKER_NODE_ID.to_string(),
        label: "RabbitMQ".to_string(),
        kind: NodeKind::Broker,
        metadata: json!({}),
    }
}

fn mcp_server_node(server: &crate::mcp::ServerCatalog) -> Node {
    Node {
        id: mcp_server_id(&server.label),
        label: server.label.clone(),
        kind: NodeKind::McpServer,
        metadata: json!({
            "url": server.url,
            "connected": server.connected,
            "tool_count": server.tool_count,
            "tools": server.tools,
        }),
    }
}

fn service_node(svc: &str) -> Node {
    Node {
        id: format!("service.{svc}"),
        label: svc.to_string(),
        kind: NodeKind::Service,
        metadata: json!({ "status": "unknown" }),
    }
}

fn exchange_node(name: &str, kind: &str) -> Node {
    Node {
        id: exchange_id(name),
        label: name.to_string(),
        kind: NodeKind::Exchange,
        metadata: json!({ "kind": kind }),
    }
}

fn queue_node(name: &str) -> Node {
    Node {
        id: queue_id(name),
        label: name.to_string(),
        kind: NodeKind::Queue,
        metadata: json!({}),
    }
}

// --- ID helpers (kept centralized so static edges + runtime edges agree) ---

fn mcp_server_id(label: &str) -> String {
    format!("mcp.{label}")
}

fn exchange_id(name: &str) -> String {
    format!("exchange.{name}")
}

fn queue_id(name: &str) -> String {
    format!("queue.{name}")
}

fn is_system_exchange(name: &str) -> bool {
    name.is_empty() || name.starts_with("amq.")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::architecture::rabbitmq::{BindingInfo, ExchangeInfo, QueueInfo};
    use crate::mcp::{CatalogResponse, ServerCatalog, ToolSummary};
    use std::collections::HashSet;

    fn empty_catalog() -> CatalogResponse {
        CatalogResponse {
            servers: Vec::new(),
            generated_at: Utc::now(),
        }
    }

    fn empty_topology() -> RabbitMqTopology {
        RabbitMqTopology {
            exchanges: Vec::new(),
            queues: Vec::new(),
            bindings: Vec::new(),
        }
    }

    fn fixture_catalog() -> CatalogResponse {
        CatalogResponse {
            servers: vec![ServerCatalog {
                label: "crm".into(),
                url: "http://crm:7001/mcp".into(),
                connected: true,
                tool_count: 1,
                tools: vec![ToolSummary {
                    name: "search_contact".into(),
                    description: "Fuzzy search".into(),
                    requires_approval: false,
                    input_schema: json!({"type": "object"}),
                }],
            }],
            generated_at: Utc::now(),
        }
    }

    fn fixture_topology() -> RabbitMqTopology {
        RabbitMqTopology {
            exchanges: vec![
                ExchangeInfo {
                    name: "".into(), // system default — should be filtered
                    kind: "direct".into(),
                },
                ExchangeInfo {
                    name: "amq.direct".into(), // system — should be filtered
                    kind: "direct".into(),
                },
                ExchangeInfo {
                    name: "ai.events".into(),
                    kind: "topic".into(),
                },
            ],
            queues: vec![QueueInfo {
                name: "frontend.ai_incidents".into(),
            }],
            bindings: vec![
                BindingInfo {
                    source: "ai.events".into(),
                    destination: "frontend.ai_incidents".into(),
                    destination_type: "queue".into(),
                    routing_key: "event.incident_diagnosed".into(),
                },
                BindingInfo {
                    source: "".into(), // system default → must be filtered
                    destination: "frontend.ai_incidents".into(),
                    destination_type: "queue".into(),
                    routing_key: "frontend.ai_incidents".into(),
                },
            ],
        }
    }

    #[test]
    fn build_includes_master_and_consumer_and_broker() {
        let r = build_from_parts(&empty_catalog(), &empty_topology());
        let ids: HashSet<&str> = r.nodes.iter().map(|n| n.id.as_str()).collect();
        assert!(ids.contains(SELF_NODE_ID));
        assert!(ids.contains(CONSUMER_FRONTEND_ID));
        assert!(ids.contains(BROKER_NODE_ID));
    }

    #[test]
    fn build_includes_all_static_services() {
        let r = build_from_parts(&empty_catalog(), &empty_topology());
        let ids: HashSet<&str> = r.nodes.iter().map(|n| n.id.as_str()).collect();
        for svc in SERVICES {
            assert!(
                ids.contains(format!("service.{svc}").as_str()),
                "missing service node for {svc}"
            );
        }
    }

    #[test]
    fn build_includes_all_externals() {
        let r = build_from_parts(&empty_catalog(), &empty_topology());
        let ids: HashSet<&str> = r.nodes.iter().map(|n| n.id.as_str()).collect();
        for ext in EXTERNALS {
            assert!(ids.contains(ext.id), "missing external node: {}", ext.id);
        }
    }

    #[test]
    fn build_adds_mcp_server_nodes_from_catalog() {
        let r = build_from_parts(&fixture_catalog(), &empty_topology());
        let crm_node = r.nodes.iter().find(|n| n.id == "mcp.crm");
        assert!(crm_node.is_some(), "missing mcp.crm node");
        let node = crm_node.unwrap();
        assert_eq!(node.kind, NodeKind::McpServer);
        assert_eq!(node.metadata.get("url").unwrap(), "http://crm:7001/mcp");
        assert_eq!(node.metadata.get("tool_count").unwrap(), 1);
    }

    #[test]
    fn build_creates_master_to_mcp_server_edge_per_catalog_entry() {
        let r = build_from_parts(&fixture_catalog(), &empty_topology());
        let mcp_edges: Vec<_> = r
            .edges
            .iter()
            .filter(|e| e.source == SELF_NODE_ID && e.target == "mcp.crm")
            .collect();
        assert_eq!(mcp_edges.len(), 1);
        assert_eq!(mcp_edges[0].kind, EdgeKind::Mcp);
    }

    #[test]
    fn build_filters_system_exchanges() {
        let r = build_from_parts(&empty_catalog(), &fixture_topology());
        let exchange_ids: HashSet<&str> = r
            .nodes
            .iter()
            .filter(|n| n.kind == NodeKind::Exchange)
            .map(|n| n.id.as_str())
            .collect();
        assert!(exchange_ids.contains("exchange.ai.events"));
        assert!(!exchange_ids.contains("exchange."));
        assert!(!exchange_ids.contains("exchange.amq.direct"));
    }

    #[test]
    fn build_creates_amqp_binding_edges_skipping_system_sources() {
        let r = build_from_parts(&empty_catalog(), &fixture_topology());
        let binding_edges: Vec<_> = r
            .edges
            .iter()
            .filter(|e| e.kind == EdgeKind::AmqpBinding)
            .collect();
        assert_eq!(binding_edges.len(), 1);
        let edge = binding_edges[0];
        assert_eq!(edge.source, "exchange.ai.events");
        assert_eq!(edge.target, "queue.frontend.ai_incidents");
        assert_eq!(edge.label.as_deref(), Some("event.incident_diagnosed"));
    }

    #[test]
    fn build_produces_no_dangling_edges() {
        let r = build_from_parts(&fixture_catalog(), &fixture_topology());
        let node_ids: HashSet<&str> = r.nodes.iter().map(|n| n.id.as_str()).collect();
        for edge in &r.edges {
            assert!(
                node_ids.contains(edge.source.as_str()),
                "dangling edge source: {}",
                edge.source
            );
            assert!(
                node_ids.contains(edge.target.as_str()),
                "dangling edge target: {}",
                edge.target
            );
        }
    }

    #[test]
    fn build_includes_static_edges_whose_endpoints_are_present() {
        // Empty catalog → mcp.* nodes absent → edges referencing them
        // are filtered out. Edges whose endpoints all exist must still
        // appear (e.g. consumer.frontend → self.mcp-master, services →
        // broker).
        let r = build_from_parts(&empty_catalog(), &empty_topology());
        let node_ids: HashSet<&str> = r.nodes.iter().map(|n| n.id.as_str()).collect();
        for se in STATIC_EDGES {
            let endpoints_present = node_ids.contains(se.source) && node_ids.contains(se.target);
            let matched = r
                .edges
                .iter()
                .any(|e| e.source == se.source && e.target == se.target);
            assert_eq!(
                matched, endpoints_present,
                "static edge {}→{} presence in output should match endpoint availability",
                se.source, se.target
            );
        }
    }

    #[test]
    fn build_skips_static_edges_referencing_disconnected_mcp_servers() {
        // mcp.controlroom is not in the fixture catalog (only crm).
        // Edges sourcing from it must be filtered.
        let r = build_from_parts(&fixture_catalog(), &empty_topology());
        for edge in &r.edges {
            assert_ne!(edge.source, "mcp.controlroom");
        }
    }
}
