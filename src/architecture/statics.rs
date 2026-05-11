//! Compile-time architecture-config: services list, external systems, and
//! the static edges between them.
//!
//! Why inline Rust const instead of a TOML/YAML file: mcp-master loads no
//! config files today — `dotenvy::dotenv()` + `std::env::var` is the only
//! config pattern. Inline consts are type-checked, refactor-safe, and
//! avoid adding a file-loading dep for data that changes only on deploy.

use super::dto::EdgeKind;

/// Team backend services that publish heartbeats to RabbitMQ. Listed here
/// so the architecture graph has service-nodes even before service-health
/// probing lands (PR 2.5). Status is reported as "unknown" until then.
pub const SERVICES: &[&str] = &[
    "kassa",
    "crm",
    "controlroom",
    "frontend",
    "mailing",
    "facturatie",
    "planning",
    "iot",
];

/// External system descriptor (Anthropic, Salesforce, etc.) — entities
/// mcp-master or an MCP-server talks to, but that have no API to discover
/// them dynamically. Updated only on deploy.
pub struct ExternalSpec {
    pub id: &'static str,
    pub label: &'static str,
    pub url: &'static str,
}

pub const EXTERNALS: &[ExternalSpec] = &[
    ExternalSpec {
        id: "external.anthropic",
        label: "Anthropic API",
        url: "https://api.anthropic.com",
    },
    ExternalSpec {
        id: "external.salesforce",
        label: "Salesforce",
        url: "https://login.salesforce.com",
    },
    ExternalSpec {
        id: "external.github",
        label: "GitHub Actions",
        url: "https://api.github.com",
    },
    ExternalSpec {
        id: "external.teams",
        label: "Microsoft Teams",
        url: "https://graph.microsoft.com",
    },
    ExternalSpec {
        id: "external.elasticsearch",
        label: "Elasticsearch",
        url: "internal",
    },
];

/// Edge between two well-known node-ids that doesn't come from runtime
/// discovery — e.g. "Drupal Frontend → mcp-master HTTP". Compiled-in.
pub struct StaticEdge {
    pub source: &'static str,
    pub target: &'static str,
    pub kind: EdgeKind,
    pub label: Option<&'static str>,
}

pub const STATIC_EDGES: &[StaticEdge] = &[
    // Consumer → master
    StaticEdge {
        source: "consumer.frontend",
        target: "self.mcp-master",
        kind: EdgeKind::Http,
        label: Some("POST /chat"),
    },
    // Master → externals
    StaticEdge {
        source: "self.mcp-master",
        target: "external.anthropic",
        kind: EdgeKind::Http,
        label: Some("Messages API"),
    },
    // MCP-servers → externals
    StaticEdge {
        source: "mcp.crm",
        target: "external.salesforce",
        kind: EdgeKind::External,
        label: Some("REST API"),
    },
    StaticEdge {
        source: "mcp.controlroom",
        target: "external.elasticsearch",
        kind: EdgeKind::External,
        label: Some("query"),
    },
    StaticEdge {
        source: "mcp.controlroom",
        target: "external.github",
        kind: EdgeKind::External,
        label: Some("Actions API"),
    },
    // Controlroom watchdog → Teams (outbound webhook, see Controlroom watchdog.go)
    StaticEdge {
        source: "service.controlroom",
        target: "external.teams",
        kind: EdgeKind::Http,
        label: Some("webhook"),
    },
    // Services → broker (heartbeat producers; one edge per service)
    StaticEdge {
        source: "service.kassa",
        target: "broker.rabbitmq",
        kind: EdgeKind::AmqpPublish,
        label: Some("heartbeat"),
    },
    StaticEdge {
        source: "service.crm",
        target: "broker.rabbitmq",
        kind: EdgeKind::AmqpPublish,
        label: Some("heartbeat"),
    },
    StaticEdge {
        source: "service.frontend",
        target: "broker.rabbitmq",
        kind: EdgeKind::AmqpPublish,
        label: Some("heartbeat"),
    },
    StaticEdge {
        source: "service.mailing",
        target: "broker.rabbitmq",
        kind: EdgeKind::AmqpPublish,
        label: Some("heartbeat"),
    },
    StaticEdge {
        source: "service.facturatie",
        target: "broker.rabbitmq",
        kind: EdgeKind::AmqpPublish,
        label: Some("heartbeat"),
    },
    StaticEdge {
        source: "service.planning",
        target: "broker.rabbitmq",
        kind: EdgeKind::AmqpPublish,
        label: Some("heartbeat"),
    },
    StaticEdge {
        source: "service.iot",
        target: "broker.rabbitmq",
        kind: EdgeKind::AmqpPublish,
        label: Some("heartbeat"),
    },
    StaticEdge {
        source: "service.controlroom",
        target: "broker.rabbitmq",
        kind: EdgeKind::AmqpPublish,
        label: Some("heartbeat-watchdog"),
    },
    // Master → broker (publish ai.events audit-feed)
    StaticEdge {
        source: "self.mcp-master",
        target: "broker.rabbitmq",
        kind: EdgeKind::AmqpPublish,
        label: Some("ai.events"),
    },
];

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn services_list_contains_expected_team_services() {
        let set: HashSet<&str> = SERVICES.iter().copied().collect();
        for expected in &[
            "kassa",
            "crm",
            "frontend",
            "mailing",
            "facturatie",
            "planning",
            "iot",
            "controlroom",
        ] {
            assert!(set.contains(expected), "missing service: {expected}");
        }
    }

    #[test]
    fn services_list_has_no_duplicates() {
        let set: HashSet<&str> = SERVICES.iter().copied().collect();
        assert_eq!(set.len(), SERVICES.len(), "duplicate service in SERVICES");
    }

    #[test]
    fn externals_have_unique_ids() {
        let set: HashSet<&str> = EXTERNALS.iter().map(|e| e.id).collect();
        assert_eq!(set.len(), EXTERNALS.len(), "duplicate id in EXTERNALS");
    }

    #[test]
    fn externals_ids_use_external_prefix() {
        for ext in EXTERNALS {
            assert!(
                ext.id.starts_with("external."),
                "external id should be prefixed: {}",
                ext.id
            );
        }
    }

    #[test]
    fn static_edges_reference_known_node_ids() {
        // Build the set of well-known node-ids the static edges may
        // reference. Anything not in this set is a typo.
        let mut known: HashSet<String> = HashSet::new();
        known.insert("self.mcp-master".to_string());
        known.insert("consumer.frontend".to_string());
        known.insert("broker.rabbitmq".to_string());
        for svc in SERVICES {
            known.insert(format!("service.{svc}"));
        }
        for ext in EXTERNALS {
            known.insert(ext.id.to_string());
        }
        // MCP-server ids come from runtime catalog. Static edges may
        // reference well-known ones (crm, controlroom) by convention.
        known.insert("mcp.crm".to_string());
        known.insert("mcp.controlroom".to_string());

        for edge in STATIC_EDGES {
            assert!(
                known.contains(edge.source),
                "static edge has unknown source: {}",
                edge.source
            );
            assert!(
                known.contains(edge.target),
                "static edge has unknown target: {}",
                edge.target
            );
        }
    }
}
