// lib.rs

use std::collections::HashSet;
use std::ffi::CString;
use std::ptr;

const URL_VERSION: f32 = 0.0.2;
const URL_PROTOCOL_MAX_LENGTH: usize = 16;
const URL_HOSTNAME_MAX_LENGTH: usize = 128;
const URL_TLD_MAX_LENGTH: usize = 16;
const URL_AUTH_MAX_LENGTH: usize = 32;

lazy_static! {
    static ref URL_SCHEMES: HashSet<&'static str> = {
        let schemes = vec![
            "aaa", "aaas", "about", "acap", "acct", "adiumxtra", "afp", "afs", "aim", "apt", "attachment", "aw",
            "beshare", "bitcoin", "bolo", "callto", "cap", "chrome", "crome-extension", "com-evenbrite-attendee",
            "cid", "coap", "coaps", "content", "crid", "cvs", "data", "dav", "dict", "lna-playsingle", "dln-playcontainer",
            "dns", "dtn", "dvb", "ed2k", "facetime", "fax", "feed", "file", "finger", "fish", "ftp", "geo", "gg", "git",
            "gizmoproject", "go", "gopher", "gtalk", "h323", "hcp", "http", "https", "iax", "icap", "icon", "im",
            "imap", "info", "ipn", "ipp", "irc", "irc6", "ircs", "iris", "iris.beep", "iris.xpc", "iris.xpcs", "iris.lws",
            "itms", "jabber", "jar", "jms", "keyparc", "lastfm", "ldap", "ldaps", "magnet", "mailserver", "mailto",
            "maps", "market", "message", "mid", "mms", "modem", "ms-help", "mssettings-power", "msnim", "msrp",
            "msrps", "mtqp", "mumble", "mupdate", "mvn", "news", "nfs", "ni", "nih", "nntp", "notes", "oid",
            "paquelocktoken", "pack", "palm", "paparazzi", "pkcs11", "platform", "pop", "pres", "prospero", "proxy",
            "psyc", "query", "reload", "res", "resource", "rmi", "rsync", "rtmp", "rtsp", "secondlife", "service", "session",
            "sftp", "sgn", "shttp", "sieve", "sip", "sips", "skype", "smb", "sms", "snews", "snmp", "soap.beep", "soap.beeps",
            "soldat", "spotify", "ssh", "steam", "svn", "tag", "teamspeak", "tel", "telnet", "tftp", "things", "thismessage",
            "tn3270", "tip", "tv", "udp", "unreal", "urn", "ut2004", "vemmi", "ventrilo", "videotex", "view-source", "wais", "webcal",
            "ws", "wss", "wtai", "wyciwyg", "xcon", "xcon-userid", "xfire", "xmlrpc.beep", "xmlrpc.beeps", "xmpp", "xri", "ymsgr",
            "javascript", "jdbc", "doi"
        ];
        schemes.into_iter().collect()
    };
}

#[derive(Debug)]
pub struct UrlData {
    href: String,
    protocol: Option<String>,
    host: Option<String>,
    auth: Option<String>,
    hostname: Option<String>,
    pathname: Option<String>,
    search: Option<String>,
    path: Option<String>,
    hash: Option<String>,
    query: Option<String>,
    port: Option<String>,
}

impl UrlData {
    pub fn new(url: &str) -> Self {
        UrlData {
            href: url.to_string(),
            protocol: None,
            host: None,
            auth: None,
            hostname: None,
            pathname: None,
            search: None,
            path: None,
            hash: None,
            query: None,
            port: None,
        }
    }

    pub fn parse(url: &str) -> Option<Self> {
        let mut data = UrlData::new(url);
        let protocol = UrlData::get_protocol(url)?;
        data.protocol = Some(protocol.clone());

        let is_ssh = UrlData::is_ssh(&protocol);
        let protocol_len = protocol.len() + 3;

        let auth = UrlData::get_auth(url, protocol_len);
        data.auth = auth.clone();

        let auth_len = auth.map_or(0, |a| a.len() + 1);

        let hostname = if is_ssh {
            UrlData::get_part(url, "%[^:]", protocol_len + auth_len)
        } else {
            UrlData::get_part(url, "%[^/]", protocol_len + auth_len)
        };
        data.hostname = hostname.clone();

        let host = hostname.as_ref().map(|h| h.split(':').next().unwrap_or("").to_string());
        data.host = host.clone();

        let path = if is_ssh {
            UrlData::get_part(url, ":%s", protocol_len + auth_len + hostname.as_ref().map_or(0, |h| h.len()))
        } else {
            UrlData::get_part(url, "/%s", protocol_len + auth_len + hostname.as_ref().map_or(0, |h| h.len()))
        };
        data.path = path.clone();

        let pathname = path.as_ref().map(|p| p.split('?').next().unwrap_or("").to_string());
        data.pathname = pathname.clone();

        let search = path.as_ref().map(|p| p.split('#').next().unwrap_or("").to_string());
        data.search = search.clone();

        let query = search.as_ref().map(|s| s.split('?').nth(1).unwrap_or("").to_string());
        data.query = query.clone();

        let hash = path.as_ref().map(|p| p.split('#').nth(1).unwrap_or("").to_string());
        data.hash = hash.clone();

        let port = hostname.as_ref().map(|h| h.split(':').nth(1).unwrap_or("").to_string());
        data.port = port.clone();

        Some(data)
    }

    fn get_protocol(url: &str) -> Option<String> {
        let protocol: String = url.split("://").next()?.to_string();
        if UrlData::is_protocol(&protocol) {
            Some(protocol)
        } else {
            None
        }
    }

    fn get_auth(url: &str, offset: usize) -> Option<String> {
        UrlData::get_part(url, "%[^@]", offset)
    }

    fn get_part(url: &str, format: &str, offset: usize) -> Option<String> {
        let url_part = &url[offset..];
        let part: String = url_part.split(format).next()?.to_string();
        if part.is_empty() {
            None
        } else {
            Some(part)
        }
    }

    fn is_protocol(protocol: &str) -> bool {
        URL_SCHEMES.contains(protocol)
    }

    fn is_ssh(protocol: &str) -> bool {
        protocol == "ssh" || protocol == "git"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_url_parse() {
        let url = "http://user:pass@host:8080/path?query#fragment";
        let data = UrlData::parse(url).unwrap();
        assert_eq!(data.protocol, Some("http".to_string()));
        assert_eq!(data.auth, Some("user:pass".to_string()));
        assert_eq!(data.hostname, Some("host:8080".to_string()));
        assert_eq!(data.host, Some("host".to_string()));
        assert_eq!(data.port, Some("8080".to_string()));
        assert_eq!(data.path, Some("/path?query#fragment".to_string()));
        assert_eq!(data.pathname, Some("/path".to_string()));
        assert_eq!(data.search, Some("?query".to_string()));
        assert_eq!(data.query, Some("query".to_string()));
        assert_eq!(data.hash, Some("fragment".to_string()));
    }
}