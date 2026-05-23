//! Service -> GitHub repo coordinates for the agent's write tools.
//!
//! Controlroom's `request_changes_with_files` takes raw owner/repo/base and its
//! service-name resolution is unreliable, so we hand the agent the canonical
//! mapping directly. Built-in defaults cover the known org repos; the
//! `SERVICE_REPO_MAP` env var (JSON object keyed by service) overrides/extends them.

use std::collections::BTreeMap;

use serde::Deserialize;

const DEFAULT_OWNER: &str = "Integration-Project-2026-Groep-2";

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct RepoTarget {
    pub owner: String,
    pub repo: String,
    pub default_branch: String,
}

fn builtin_map() -> BTreeMap<String, RepoTarget> {
    [
        ("crm", "CRM"),
        ("facturatie", "Facturatie"),
        ("frontend", "Frontend"),
        ("mailing", "Mailing"),
        ("planning", "Planning"),
        ("kassa", "Kassa"),
        ("controlroom", "Controlroom"),
    ]
    .into_iter()
    .map(|(service, repo)| {
        (
            service.to_string(),
            RepoTarget {
                owner: DEFAULT_OWNER.to_string(),
                repo: repo.to_string(),
                default_branch: "main".to_string(),
            },
        )
    })
    .collect()
}

/// Built-in defaults overlaid with `SERVICE_REPO_MAP` (JSON) entries, if present.
/// Unparseable env values are ignored (warn + defaults) rather than failing the
/// request — a bad override must not take the write-flow down.
pub fn service_repo_map() -> BTreeMap<String, RepoTarget> {
    let mut map = builtin_map();
    match std::env::var("SERVICE_REPO_MAP") {
        Ok(raw) if !raw.trim().is_empty() => {
            match serde_json::from_str::<BTreeMap<String, RepoTarget>>(&raw) {
                Ok(overrides) => {
                    for (service, target) in overrides {
                        map.insert(service.to_lowercase(), target);
                    }
                }
                Err(e) => {
                    tracing::warn!("SERVICE_REPO_MAP unparseable, using built-in defaults: {e}");
                }
            }
        }
        _ => {}
    }
    map
}

/// A prompt fragment listing the canonical repo coordinates, instructing the
/// agent to pass them as explicit arguments to GitHub write tools instead of
/// relying on (currently unreliable) service-name resolution.
pub fn repo_hints_prompt() -> String {
    let map = service_repo_map();
    let mut out = String::from(
        "\nKnown GitHub repositories (INTERNAL — use only to fill tool arguments):\n\
         When a GitHub write tool such as request_changes_with_files is needed, take repo \
         and base from the list below. The base branch MUST be exactly the one listed for \
         the service; a wrong base causes a 'reference not found' error. Never mention these \
         coordinates, branch names, or tool names in your reply to the user — describe the \
         proposed change in plain business terms.\n",
    );
    for (service, target) in &map {
        out.push_str(&format!(
            "- service '{}': repo={} base={} owner={}\n",
            service, target.repo, target.default_branch, target.owner
        ));
    }
    out
}

#[cfg(test)]
mod tests;
