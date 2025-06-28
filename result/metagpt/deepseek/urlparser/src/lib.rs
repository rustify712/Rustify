//! URL parsing module in Rust
//! 
//! Provides functionality to parse URLs into their components

use std::collections::HashMap;

/// Maximum length for various URL components
const URL_PROTOCOL_MAX_LENGTH: usize = 16;
const URL_HOSTNAME_MAX_LENGTH: usize = 128;
const URL_TLD_MAX_LENGTH: usize = 16;
const URL_AUTH_MAX_LENGTH: usize = 32;

/// List of known URI schemes
const URL_SCHEMES: &[&str] = &[
    // Official IANA registered schemes
    "aaa", "aaas", "about", "acap", "acct", "adiumxtra", "afs", "aim", "apt", "attachment", "aw",
    "beshare", "bitcoin", "bolo", "callto", "cap", "chrome", "chrome-extension", "cid", "coap", 
    "coaps", "content", "crid", "cvs", "data", "dav", "dict", "dns", "dtn", "dvb", "ed2k", 
    "facetime", "fax", "feed", "file", "finger", "fish", "ftp", "geo", "gg", "git", "gizmoproject",
    "go", "gopher", "gtalk", "h323", "hcp", "http", "https", "iax", "icap", "icon", "im", "imap",
    "info", "ipn", "ipp", "irc", "irc6", "ircs", "iris", "iris.beep", "iris.xpc", "iris.xpcs",
    "iris.lws", "itms", "jabber", "jar", "jms", "keyparc", "lastfm", "ldap", "ldaps", "magnet",
    "mailserver", "mailto", "maps", "market", "message", "mid", "mms", "modem", "ms-help",
    "msnim", "msrp", "msrps", "mtqp", "mumble", "mupdate", "mvn", "news", "nfs", "ni", "nih",
    "nntp", "notes", "oid", "paquelocktoken", "pack", "palm", "paparazzi", "pkcs11", "platform",
    "pop", "pres", "prospero", "proxy", "psyc", "query", "reload", "res", "resource", "rmi",
    "rsync", "rtmp", "rtsp", "secondlife", "service", "session", "sftp", "sgn", "shttp", "sieve",
    "sip", "sips", "skype", "smb", "sms", "snews", "snmp", "soap.beep", "soap.beeps", "soldat",
    "spotify", "ssh", "steam", "svn", "tag", "teamspeak", "tel", "telnet", "tftp", "things",
    "thismessage", "tn3270", "tip", "tv", "udp", "unreal", "urn", "ut2004", "vemmi", "ventrilo",
    "videotex", "view-source", "wais", "webcal", "ws", "wss", "wtai", "wyciwyg", "xcon", 
    "xcon-userid", "xfire", "xmlrpc.beep", "xmlrpc.beeps", "xmpp", "xri", "ymsgr",
    
    // Unofficial schemes
    "javascript", "jdbc", "doi"
];

/// Represents the parsed components of a URL
#[derive(Debug, Default, Clone)]
pub struct UrlData {
    pub href: String,
    pub protocol: Option<String>,
    pub host: Option<String>,
    pub auth: Option<String>,
    pub hostname: Option<String>,
    pub pathname: Option<String>,
    pub search: Option<String>,
    pub path: Option<String>,
    pub hash: Option<String>,
    pub query: Option<String>,
    pub port: Option<String>,
}

impl UrlData {
    /// Creates a new empty UrlData instance
    pub fn new() -> Self {
        Self::default()
    }

    /// Parses query string into key-value pairs
    pub fn parse_query(&self) -> HashMap<String, String> {
        let mut params = HashMap::new();
        if let Some(query) = &self.query {
            for pair in query.split('&') {
                let mut kv = pair.splitn(2, '=');
                if let Some(key) = kv.next() {
                    let value = kv.next().unwrap_or("");
                    params.insert(key.to_string(), value.to_string());
                }
            }
        }
        params
    }
}

/// Parses a URL string into its components
pub fn url_parse(url: &str) -> Option<UrlData> {
    let mut data = UrlData::new();
    data.href = url.to_string();

    // Parse protocol
    if let Some(colon) = url.find(':') {
        let protocol = &url[..colon];
        if URL_SCHEMES.contains(&protocol) {
            data.protocol = Some(protocol.to_string());
            
            // Parse host part
            let mut rest = &url[colon+1..];
            if rest.starts_with("//") {
                rest = &rest[2..];
                
                // Parse auth if exists
                if let Some(at) = rest.find('@') {
                    data.auth = Some(rest[..at].to_string());
                    rest = &rest[at+1..];
                }
                
                // Parse hostname and port
                let host_end = rest.find('/').unwrap_or(rest.len());
                let host_part = &rest[..host_end];
                
                if let Some(colon) = host_part.find(':') {
                    data.hostname = Some(host_part[..colon].to_string());
                    data.port = Some(host_part[colon+1..].to_string());
                } else {
                    data.hostname = Some(host_part.to_string());
                }
                data.host = Some(host_part.to_string());
                
                // Parse path and query
                rest = &rest[host_end..];
                if !rest.is_empty() {
                    data.path = Some(rest.to_string());
                    
                    // Split pathname and query/hash
                    let hash_pos = rest.find('#');
                    let query_pos = rest.find('?');
                    
                    match (hash_pos, query_pos) {
                        (Some(h), Some(q)) if h < q => {
                            data.pathname = Some(rest[..h].to_string());
                            data.hash = Some(rest[h+1..].to_string());
                        },
                        (Some(h), Some(q)) => {
                            data.pathname = Some(rest[..q].to_string());
                            data.query = Some(rest[q+1..h].to_string());
                            data.hash = Some(rest[h+1..].to_string());
                        },
                        (Some(h), None) => {
                            data.pathname = Some(rest[..h].to_string());
                            data.hash = Some(rest[h+1..].to_string());
                        },
                        (None, Some(q)) => {
                            data.pathname = Some(rest[..q].to_string());
                            data.query = Some(rest[q+1..].to_string());
                        },
                        (None, None) => {
                            data.pathname = Some(rest.to_string());
                        }
                    }
                }
            }
        }
    }
    
    Some(data)
}

/// Checks if a string is a known URL protocol
pub fn url_is_protocol(s: &str) -> bool {
    URL_SCHEMES.contains(&s)
}

/// Checks if a string is an SSH protocol
pub fn url_is_ssh(s: &str) -> bool {
    s == "ssh" || s == "sftp"
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_url_is_protocol() {
        assert!(url_is_protocol("http"));
        assert!(url_is_protocol("https"));
        assert!(url_is_protocol("ftp"));
        assert!(!url_is_protocol("unknown"));
    }

    #[test]
    fn test_url_is_ssh() {
        assert!(url_is_ssh("ssh"));
        assert!(url_is_ssh("sftp"));
        assert!(!url_is_ssh("http"));
    }

    #[test]
    fn test_url_parse_basic() {
        let url = "https://example.com/path?query=1#hash";
        let parsed = url_parse(url).unwrap();
        
        assert_eq!(parsed.protocol, Some("https".to_string()));
        assert_eq!(parsed.hostname, Some("example.com".to_string()));
        assert_eq!(parsed.pathname, Some("/path".to_string()));
        assert_eq!(parsed.query, Some("query=1".to_string()));
        assert_eq!(parsed.hash, Some("hash".to_string()));
    }

    #[test]
    fn test_url_parse_with_port() {
        let url = "http://localhost:8080/api";
        let parsed = url_parse(url).unwrap();
        
        assert_eq!(parsed.protocol, Some("http".to_string()));
        assert_eq!(parsed.hostname, Some("localhost".to_string()));
        assert_eq!(parsed.port, Some("8080".to_string()));
        assert_eq!(parsed.pathname, Some("/api".to_string()));
    }

    #[test]
    fn test_url_parse_with_auth() {
        let url = "https://user:pass@example.com";
        let parsed = url_parse(url).unwrap();
        
        assert_eq!(parsed.auth, Some("user:pass".to_string()));
        assert_eq!(parsed.hostname, Some("example.com".to_string()));
    }

    #[test]
    fn test_url_parse_query_params() {
        let url = "https://example.com/search?q=rust&lang=en";
        let parsed = url_parse(url).unwrap();
        let params = parsed.parse_query();
        
        assert_eq!(params.get("q"), Some(&"rust".to_string()));
        assert_eq!(params.get("lang"), Some(&"en".to_string()));
    }

    #[test]
    fn test_url_parse_minimal() {
        let url = "http://example.com";
        let parsed = url_parse(url).unwrap();
        
        assert_eq!(parsed.protocol, Some("http".to_string()));
        assert_eq!(parsed.hostname, Some("example.com".to_string()));
        assert_eq!(parsed.pathname, Some("".to_string()));
    }

    #[test]
    fn test_url_parse_no_path() {
        let url = "https://example.com";
        let parsed = url_parse(url).unwrap();
        
        assert_eq!(parsed.protocol, Some("https".to_string()));
        assert_eq!(parsed.hostname, Some("example.com".to_string()));
        assert_eq!(parsed.pathname, Some("".to_string()));
    }

    #[test]
    fn test_url_parse_only_hash() {
        let url = "https://example.com#section";
        let parsed = url_parse(url).unwrap();
        
        assert_eq!(parsed.protocol, Some("https".to_string()));
        assert_eq!(parsed.hostname, Some("example.com".to_string()));
        assert_eq!(parsed.pathname, Some("".to_string()));
        assert_eq!(parsed.hash, Some("section".to_string()));
    }

    #[test]
    fn test_url_parse_only_query() {
        let url = "https://example.com?param=value";
        let parsed = url_parse(url).unwrap();
        
        assert_eq!(parsed.protocol, Some("https".to_string()));
        assert_eq!(parsed.hostname, Some("example.com".to_string()));
        assert_eq!(parsed.pathname, Some("".to_string()));
        assert_eq!(parsed.query, Some("param=value".to_string()));
    }
}