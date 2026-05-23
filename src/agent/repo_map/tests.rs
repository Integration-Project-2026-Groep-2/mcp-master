use super::*;

fn with_env<R>(value: Option<&str>, f: impl FnOnce() -> R) -> R {
    let prev = std::env::var("SERVICE_REPO_MAP").ok();
    unsafe {
        match value {
            Some(v) => std::env::set_var("SERVICE_REPO_MAP", v),
            None => std::env::remove_var("SERVICE_REPO_MAP"),
        }
    }
    let r = f();
    unsafe {
        match prev {
            Some(p) => std::env::set_var("SERVICE_REPO_MAP", p),
            None => std::env::remove_var("SERVICE_REPO_MAP"),
        }
    }
    r
}

#[test]
#[serial_test::serial]
fn defaults_contain_known_services_with_correct_casing() {
    with_env(None, || {
        let map = service_repo_map();
        let crm = map.get("crm").expect("crm present");
        assert_eq!(crm.owner, "Integration-Project-2026-Groep-2");
        assert_eq!(crm.repo, "CRM");
        assert_eq!(crm.default_branch, "main");
        // Title-case repos, not the all-lowercase service key.
        assert_eq!(map.get("facturatie").unwrap().repo, "Facturatie");
        assert_eq!(map.get("kassa").unwrap().repo, "Kassa");
        assert_eq!(map.len(), 7);
    });
}

#[test]
#[serial_test::serial]
fn env_override_extends_and_replaces_entries() {
    let json = r#"{"newsvc":{"owner":"o","repo":"NewRepo","default_branch":"dev"},
                   "CRM":{"owner":"o2","repo":"crm-fork","default_branch":"trunk"}}"#;
    with_env(Some(json), || {
        let map = service_repo_map();
        let new = map.get("newsvc").expect("override added newsvc");
        assert_eq!(new.repo, "NewRepo");
        assert_eq!(new.default_branch, "dev");
        // Override key is lowercased, replacing the built-in crm entry.
        let crm = map.get("crm").expect("crm still present");
        assert_eq!(crm.repo, "crm-fork");
        assert_eq!(crm.owner, "o2");
    });
}

#[test]
#[serial_test::serial]
fn garbage_env_falls_back_to_defaults() {
    with_env(Some("not json"), || {
        let map = service_repo_map();
        assert_eq!(map.get("crm").unwrap().repo, "CRM");
        assert_eq!(map.len(), 7);
    });
}

#[test]
#[serial_test::serial]
fn hints_prompt_lists_explicit_coordinates() {
    with_env(None, || {
        let p = repo_hints_prompt();
        assert!(p.contains("request_changes_with_files"));
        assert!(p.contains("owner=Integration-Project-2026-Groep-2"));
        assert!(p.contains("repo=CRM"));
        assert!(p.contains("base=main"));
    });
}
