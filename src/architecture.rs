//! Live integration-architecture graph for the `/architecture` HTTP endpoint.
//!
//! Combines runtime data (MCP-pool catalog from PR 1, RabbitMQ topology
//! from the broker management API) with compile-time static config
//! (external systems + edges that have no API to discover them) into one
//! Cytoscape.js-renderable JSON payload. Drupal Frontend consumes this
//! one-shot to draw the interactive architecture page.

// Types + statics land in this commit; their consumers (builder + handler)
// arrive in follow-up commits within the same PR. Suppress until then.
#![allow(dead_code, unused_imports)]

pub mod dto;
pub mod rabbitmq;
pub mod statics;

pub use dto::{ArchitectureResponse, Edge, EdgeKind, Node, NodeKind};
