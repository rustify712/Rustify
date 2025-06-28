use std::collections::HashSet;
use std::ffi::CString;
use std::ptr;
use std::str;

// Constants
const URL_VERSION: &str = "0.0.2";
const URL_PROTOCOL_MAX_LENGTH: usize = 16;
const URL_HOSTNAME_MAX_LENGTH: usize = 128;
const URL_TLD_MAX_LENGTH: usize = 16;
const URL_AUTH_MAX_LENGTH: usize = 32;

// URI Schemes
static URL_SCHEMES: [&str; 175] = [
    // Official IANA registered schemes
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
    // Unofficial schemes
    "javascript", "jdbc", "doi",
];

// URL Data Struct
#[derive(Debug)]
struct UrlData {
    href: String,
    protocol: String,
    host: String,
    auth: String,
    hostname: String,
    pathname: String,
    search: String,
    path: String,
    hash: String,
    query: String,
    port: String,
}

impl UrlData {
    fn new() -> Self {
        UrlData {
            href: String::new(),
            protocol: String::new(),
            host: String::new(),
            auth: String::new(),
            hostname: String::new(),
            pathname: String::new(),
            search: String::new(),
            path: String::new(),
            hash: String::new(),
            query: String::new(),
            port: String::new(),
        }
    }
}

// URL Parsing Functions

fn url_parse(url: &str) -> Option<UrlData> {
    let mut data = UrlData::new();
    data.href = url.to_string();

    let protocol = url_get_protocol(url)?;
    data.protocol = protocol.clone();
    let protocol_len = protocol.len() + 3; // +3 for "://"

    let is_ssh = url_is_ssh(&protocol);

    let auth = url_get_auth(url).unwrap_or_default();
    data.auth = auth.clone();
    let auth_len = auth.len();

    let hostname = if is_ssh {
        url_get_hostname(url, protocol_len + auth_len, true)?
    } else {
        url_get_hostname(url, protocol_len + auth_len, false)?
    };
    data.hostname = hostname.clone();
    let hostname_len = hostname.len();

    let host = url_get_host(&hostname)?;
    data.host = host.clone();
    let host_len = host.len();

    let path = if is_ssh {
        url_get_path(url, protocol_len + auth_len + hostname_len, true)?
    } else {
        url_get_path(url, protocol_len + auth_len + hostname_len, false)?
    };
    data.path = path.clone();

    let pathname = url_get_pathname(&path)?;
    data.pathname = pathname.clone();
    let pathname_len = pathname.len();

    let search = url_get_search(&path, pathname_len)?;
    data.search = search.clone();
    let search_len = search.len();

    let query = url_get_query(&search)?;
    data.query = query;

    let hash = url_get_hash(&path, pathname_len + search_len)?;
    data.hash = hash;

    let port = url_get_port(&hostname, host_len)?;
    data.port = port;

    Some(data)
}

fn url_get_protocol(url: &str) -> Option<String> {
    let mut protocol = String::new();
    for c in url.chars() {
        if c == ':' {
            break;
        }
        protocol.push(c);
    }
    if url_is_protocol(&protocol) {
        Some(protocol)
    } else {
        None
    }
}

fn url_get_auth(url: &str) -> Option<String> {
    let protocol = url_get_protocol(url)?;
    let protocol_len = protocol.len() + 3; // +3 for "://"
    get_part(url, protocol_len, "@")
}

fn url_get_hostname(url: &str, offset: usize, is_ssh: bool) -> Option<String> {
    let url = &url[offset..];
    if is_ssh {
        url.split(':').next().map(|s| s.to_string())
    } else {
        url.split('/').next().map(|s| s.to_string())
    }
}

fn url_get_host(hostname: &str) -> Option<String> {
    hostname.split(':').next().map(|s| s.to_string())
}

fn url_get_path(url: &str, offset: usize, is_ssh: bool) -> Option<String> {
    let url = &url[offset..];
    if is_ssh {
        Some(format!(":{}", url))
    } else {
        Some(format!("/{}", url))
    }
}

fn url_get_pathname(path: &str) -> Option<String> {
    path.split('?').next().map(|s| s.to_string())
}

fn url_get_search(path: &str, pathname_len: usize) -> Option<String> {
    let path = &path[pathname_len..];
    path.split('#').next().map(|s| s.to_string())
}

fn url_get_query(search: &str) -> Option<String> {
    if search.starts_with('?') {
        Some(search[1..].to_string())
    } else {
        None
    }
}

fn url_get_hash(path: &str, offset: usize) -> Option<String> {
    let path = &path[offset..];
    Some(path.to_string())
}

fn url_get_port(hostname: &str, host_len: usize) -> Option<String> {
    let hostname = &hostname[host_len + 1..];
    Some(hostname.to_string())
}

fn get_part(url: &str, offset: usize, delimiter: &str) -> Option<String> {
    let url = &url[offset..];
    url.split(delimiter).next().map(|s| s.to_string())
}

fn url_is_protocol(protocol: &str) -> bool {
    URL_SCHEMES.contains(&protocol)
}

fn url_is_ssh(protocol: &str) -> bool {
    protocol == "ssh" || protocol == "git"
}

// URL Inspection Functions

fn url_inspect(url: &str) {
    if let Some(data) = url_parse(url) {
        url_data_inspect(&data);
    }
}

fn url_data_inspect(data: &UrlData) {
    println!("#url =>");
    println!("    .href: \"{}\"", data.href);
    println!("    .protocol: \"{}\"", data.protocol);
    println!("    .host: \"{}\"", data.host);
    println!("    .auth: \"{}\"", data.auth);
    println!("    .hostname: \"{}\"", data.hostname);
    println!("    .pathname: \"{}\"", data.pathname);
    println!("    .search: \"{}\"", data.search);
    println!("    .path: \"{}\"", data.path);
    println!("    .hash: \"{}\"", data.hash);
    println!("    .query: \"{}\"", data.query);
    println!("    .port: \"{}\"", data.port);
}