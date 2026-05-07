use anyhow::{Context, Result};
use serde_json::json;

pub struct TeamsConfig {
    pub team_id: String,
    pub channel_id: String,
    pub access_token: String,
}

impl TeamsConfig {
    pub fn from_env() -> Result<Self> {
        Ok(Self {
            team_id: std::env::var("TEAMS_ID").context("TEAMS_ID not set")?,
            channel_id: std::env::var("CHANNEL_ID").context("CHANNEL_ID not set")?,
            access_token: std::env::var("TEAMS_TOKEN").context("TEAMS_TOKEN not set")?,
        })
    }
}

pub async fn publish_to_teams(config: &TeamsConfig, message: &str) -> Result<()> {
    let url = format!(
        "https://graph.microsoft.com/v1.0/teams/{}/channels/{}/messages",
        config.team_id, config.channel_id
    );

    let body = json!({
        "body": {
            "contentType": "text",
            "content": message
        }
    });

    let res = reqwest::Client::new()
        .post(&url)
        .bearer_auth(&config.access_token)
        .json(&body)
        .send()
        .await
        .context("sending message to Teams")?;

    let status = res.status();

    if status == reqwest::StatusCode::CREATED {
        tracing::info!("message posted to Teams");
        Ok(())
    } else {
        let error_body = res.text().await.unwrap_or_default();
        anyhow::bail!("Teams API returned {}: {}", status, error_body);
    }
}

pub async fn poll_messages(
    client: &reqwest::Client,
    teams_config: &TeamsConfig,
) -> Result<Vec<serde_json::Value>> {
    let url = format!(
        "https://graph.microsoft.com/v1.0/teams/{}/channels/{}/messages",
        teams_config.team_id, teams_config.channel_id
    );
    let res = client
        .get(&url)
        .bearer_auth(&teams_config.access_token)
        .send()
        .await?;
    let body: serde_json::Value = res.json().await?;

    let messages = body["value"].as_array().cloned().unwrap_or_default();

    Ok(messages)
}
