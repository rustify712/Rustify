// Dependencies: rand = "0.4", regex = "1", md5 = "0.7.0"

// URL constants and scheme definitions
const URL_VERSION: &str = "0.0.2";
const URL_PROTOCOL_MAX_LENGTH: usize = 16;
const URL_HOSTNAME_MAX_LENGTH: usize = 128;
const URL_TLD_MAX_LENGTH: usize = 16;
const URL_AUTH_MAX_LENGTH: usize = 32;

// URI Schemes as a static slice of strings
static URL_SCHEMES: &[&str] = &[
    // official IANA registered schemes
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
    // unofficial schemes
    "javascript", "jdbc", "doi"
];

// URL data structure
#[derive(Debug, Default)]
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

// Helper functions
fn strdup(s: &str) -> String {
    s.to_string()
}

fn is_protocol(str: &str) -> bool {
    URL_SCHEMES.contains(&str)
}

fn is_ssh(protocol: &str) -> bool {
    matches!(protocol, "ssh" | "git")
}

// Main URL parsing functions
pub fn url_parse(url: &str) -> Option<UrlData> {
    let mut data = UrlData::default();
    data.href = url.to_string();

    let protocol = url_get_protocol(url)?;
    let protocol_len = protocol.len() + 3;
    data.protocol = Some(protocol);

    let is_ssh = is_ssh(data.protocol.as_deref()?);

    if let Some(auth) = url_get_auth(url) {
        data.auth = Some(auth);
    }

    if let Some(hostname) = url_get_hostname(url, &protocol, is_ssh) {
        data.hostname = Some(hostname);
    }

    if let Some(host) = url_get_host(url) {
        data.host = Some(host);
    }

    if let Some(path) = url_get_path(url) {
        data.path = Some(path);
    }

    if let Some(pathname) = url_get_pathname(url) {
        data.pathname = Some(pathname);
    }

    if let Some(search) = url_get_search(url) {
        data.search = Some(search);
    }

    if let Some(query) = url_get_query(url) {
        data.query = Some(query);
    }

    if let Some(hash) = url_get_hash(url) {
        data.hash = Some(hash);
    }

    if let Some(port) = url_get_port(url) {
        data.port = Some(port);
    }

    Some(data)
}

pub fn url_get_protocol(url: &str) -> Option<String> {
    let mut protocol = String::new();
    if url.split("://").next().map(|p| {
        protocol.push_str(p);
        true
    }).unwrap_or(false) {
        if is_protocol(&protocol) {
            return Some(protocol);
        }
    }
    None
}

pub fn url_get_auth(url: &str) -> Option<String> {
    let protocol_len = url_get_protocol(url)?.len() + 3; // Account for "://"
    get_part(url, "%[^@]", protocol_len)
}

pub fn url_get_hostname(url: &str, protocol: &str, is_ssh: bool) -> Option<String> {
    let protocol_len = protocol.len() + 3;
    let auth_len = url_get_auth(url).map_or(0, |auth| auth.len() + 1);
    let len = protocol_len + auth_len;

    if is_ssh {
        get_part(url, "%[^:]", len)
    } else {
        get_part(url, "%[^/]", len)
    }
}

pub fn url_get_host(url: &str) -> Option<String> {
    let hostname = url_get_hostname(url, url_get_protocol(url)?.as_str(), is_ssh(url_get_protocol(url)?.as_str()))?;
    Some(hostname.split(':').next()?.to_string())
}

pub fn url_get_path(url: &str) -> Option<String> {
    let protocol = url_get_protocol(url)?;
    let is_ssh = is_ssh(&protocol);
    let len = protocol.len() + 3; // "://"

    if let Some(path) = get_part(url, if is_ssh { ":%s" } else { "/%s" }, len) {
        Some(path)
    } else {
        None
    }
}

pub fn url_get_pathname(url: &str) -> Option<String> {
    let path = url_get_path(url)?;
    Some(path.split('?').next()?.to_string())
}

pub fn url_get_search(url: &str) -> Option<String> {
    let path = url_get_path(url)?;
    let pathname = url_get_pathname(url)?;
    Some(path.split('#').next()?.to_string())
}

pub fn url_get_query(url: &str) -> Option<String> {
    let search = url_get_search(url)?;
    Some(search.split('?').nth(1)?.to_string())
}

pub fn url_get_hash(url: &str) -> Option<String> {
    let path = url_get_path(url)?;
    let pathname = url_get_pathname(url)?;
    let search = url_get_search(url)?;
    Some(path.split('#').nth(1)?.to_string())
}

pub fn url_get_port(url: &str) -> Option<String> {
    let hostname = url_get_hostname(url, url_get_protocol(url)?.as_str(), is_ssh(url_get_protocol(url)?.as_str()))?;
    Some(hostname.split(':').nth(1)?.to_string())
}

// Helper function to extract a part of the URL
fn get_part(url: &str, format: &str, len: usize) -> Option<String> {
    let trimmed = &url[len..];
    if trimmed.is_empty() {
        return None;
    }
    Some(trimmed.to_string())
}

// Function to print URL data for inspection
pub fn url_data_inspect(data: &UrlData) {
    println!("#url =>");
    println!("    .href: \"{}\"", data.href);
    println!("    .protocol: \"{}\"", data.protocol.as_deref().unwrap_or(""));
    println!("    .auth: \"{}\"", data.auth.as_deref().unwrap_or(""));
    println!("    .hostname: \"{}\"", data.hostname.as_deref().unwrap_or(""));
    println!("    .host: \"{}\"", data.host.as_deref().unwrap_or(""));
    println!("    .pathname: \"{}\"", data.pathname.as_deref().unwrap_or(""));
    println!("    .search: \"{}\"", data.search.as_deref().unwrap_or(""));
    println!("    .path: \"{}\"", data.path.as_deref().unwrap_or(""));
    println!("    .query: \"{}\"", data.query.as_deref().unwrap_or(""));
    println!("    .hash: \"{}\"", data.hash.as_deref().unwrap_or(""));
    println!("    .port: \"{}\"", data.port.as_deref().unwrap_or(""));
}
