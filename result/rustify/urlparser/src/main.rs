/// 定义 URL 解析模块的版本号为 0.0.2
pub const URL_VERSION: &str = "0.0.2";

/// Maximum length constants for different parts of the URL.
const URL_PROTOCOL_MAX_LENGTH: usize = 16;
const URL_HOSTNAME_MAX_LENGTH: usize = 128;
const URL_AUTH_MAX_LENGTH: usize = 32;
const URL_TLD_MAX_LENGTH: usize = 16;

/// Represents the parsed components of a URL.
#[derive(Debug, Clone)]
pub struct UrlData {
    pub href: String,
    pub protocol: String,
    pub host: String,
    pub auth: String,
    pub hostname: String,
    pub pathname: String,
    pub search: String,
    pub path: String,
    pub hash: String,
    pub query: String,
    pub port: String,
}

/// Checks if the given string is "ssh" or "git".
pub fn url_is_ssh(s: &str) -> bool {
    matches!(s, "ssh" | "git")
}

/// Skips the first `n` characters of the string and returns the remaining part as a new `String`.
pub fn strff(s: &str, n: usize) -> String {
    s.chars().skip(n).collect()
}

/// 从字符串的某个位置开始提取子串，并返回一个新的字符串副本。
pub fn strrwd(s: &str, n: usize) -> String {
    s.get(n..).unwrap_or_default().to_string()
}

/// 从给定的 URL 字符串中提取特定部分。
pub fn get_part(url: &str, format: &str, l: usize) -> Option<String> {
    url.chars().skip(l).take(format.len()).collect::<String>().let |part| {
        if part != format {
            Some(part)
        } else {
            None
        }
    }
}

/// A list of common URL schemes, including both official and unofficial schemes.
pub const URL_SCHEMES: &[&str] = &[
    "http", "https", "ftp", "git", "ssh", "file", "mailto", // other schemes omitted for brevity
];

/// Checks if the given string is a valid URL protocol.
pub fn url_is_protocol(s: &str) -> bool {
    URL_SCHEMES.contains(&s)
}

/// Extracts the protocol part from the given URL string.
pub fn url_get_protocol(url: &str) -> Option<String> {
    let protocol = url.split("://").next()?;
    if url_is_protocol(protocol) {
        Some(protocol.to_string())
    } else {
        None
    }
}

/// Extracts the authentication part from the given URL string.
pub fn url_get_auth(url: &str) -> Option<String> {
    let protocol_len = url_get_protocol(url)?.len() + 3; // "://"
    get_part(url, "%[^@]", protocol_len)
}

/// Extracts the hostname part from the given URL string.
pub fn url_get_hostname(url: &str) -> Option<String> {
    let protocol = url_get_protocol(url)?;
    let protocol_len = protocol.len() + 3; // "://"
    let auth_len = url_get_auth(url)?.len() + 1; // include "@" symbol
    let hostname_part = url.chars().skip(protocol_len + auth_len).take_while(|&c| c != '/' && c != ':').collect::<String>();

    if hostname_part.is_empty() {
        None
    } else {
        Some(hostname_part)
    }
}

/// Parses a URL string into its components and returns an `Option<UrlData>`.
pub fn url_parse(url: &str) -> Option<UrlData> {
    let protocol = url_get_protocol(url)?;
    let protocol_len = protocol.len() + 3; // "://"
    let auth = url_get_auth(url).unwrap_or_default();
    let hostname = url_get_hostname(url)?;
    let hostname_len = hostname.len();

    let path_start = protocol_len + auth.len() + hostname_len;
    let path = url.chars().skip(path_start).collect::<String>();

    let pathname = path.split('?').next()?.split('#').next()?.to_string();
    let search = path[pathname.len()..].split('#').next()?.to_string();
    let query = if search.starts_with('?') { search[1..].to_string() } else { String::new() };
    let hash = path[pathname.len() + search.len()..].to_string();
    let port = hostname.split(':').nth(1).unwrap_or_default().to_string();

    Some(UrlData {
        href: url.to_string(),
        protocol,
        host: hostname.split(':').next()?.to_string(),
        auth,
        hostname,
        pathname,
        search,
        path,
        hash,
        query,
        port,
    })
}

/// Extracts the host part from the given URL string.
pub fn url_get_host(url: &str) -> Option<String> {
    url_get_hostname(url).map(|hostname| hostname.split(':').next()?.to_string())
}

/// Extracts the port part from the given URL string.
pub fn url_get_port(url: &str) -> Option<String> {
    url_get_hostname(url)
        .and_then(|hostname| hostname.split(':').nth(1).map(|port| port.to_string()))
}

/// Extracts the search part (query parameters) from the given URL string.
pub fn url_get_search(url: &str) -> Option<String> {
    url_get_path(url).and_then(|path| path.split('?').nth(1).map(|s| s.to_string()))
}

/// Extracts the path part from the given URL string.
pub fn url_get_path(url: &str) -> Option<String> {
    let protocol = url_get_protocol(url)?;
    let auth_len = url_get_auth(url).unwrap_or_default().len() + 1; // account for the '@' symbol
    let hostname_len = url_get_hostname(url)?.len();
    let path_start = protocol.len() + 3 + auth_len + hostname_len;
    let path = url.chars().skip(path_start).collect::<String>();

    if path.is_empty() {
        None
    } else {
        Some(format!("/{}", path))
    }
}

/// Extracts the pathname part from the given URL string.
pub fn url_get_pathname(url: &str) -> Option<String> {
    url_get_path(url).map(|path| path.split('?').next()?.split('#').next()?.to_string())
}

/// Extracts the query part (query parameters) from the given URL string.
pub fn url_get_query(url: &str) -> Option<String> {
    url_get_search(url).map(|search| search.strip_prefix('?').unwrap_or_default().to_string())
}

/// Extracts the hash part from the given URL string.
pub fn url_get_hash(url: &str) -> Option<String> {
    url_get_path(url).map(|path| path.split('#').nth(1).unwrap_or_default().to_string())
}
