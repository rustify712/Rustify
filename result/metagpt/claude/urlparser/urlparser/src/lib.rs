//! URL解析库
//! 从C语言版本转换而来

/// URL协议方案的最大长度
pub const URL_PROTOCOL_MAX_LENGTH: usize = 16;

/// URL主机名部分的最大长度
pub const URL_HOSTNAME_MAX_LENGTH: usize = 128;

/// URL顶级域名部分的最大长度
pub const URL_TLD_MAX_LENGTH: usize = 16;

/// URL认证部分的最大长度
pub const URL_AUTH_MAX_LENGTH: usize = 32;

/// URI方案列表
/// 来源: http://en.wikipedia.org/wiki/URI_scheme
static URL_SCHEMES: &[&str] = &[
    // 官方IANA注册的方案
    "aaa", "aaas", "about", "acap", "acct", "adiumxtra", "afp", "afs", "aim", "apt",
    "attachment", "aw", "beshare", "bitcoin", "bolo", "callto", "cap", "chrome",
    "crome-extension", "com-evenbrite-attendee", "cid", "coap", "coaps", "content", "crid",
    "cvs", "data", "dav", "dict", "lna-playsingle", "dln-playcontainer", "dns", "dtn",
    "dvb", "ed2k", "facetime", "fax", "feed", "file", "finger", "fish", "ftp", "geo",
    "gg", "git", "gizmoproject", "go", "gopher", "gtalk", "h323", "hcp", "http", "https",
    "iax", "icap", "icon", "im", "imap", "info", "ipn", "ipp", "irc", "irc6", "ircs",
    "iris", "iris.beep", "iris.xpc", "iris.xpcs", "iris.lws", "itms", "jabber", "jar",
    "jms", "keyparc", "lastfm", "ldap", "ldaps", "magnet", "mailserver", "mailto", "maps",
    "market", "message", "mid", "mms", "modem", "ms-help", "mssettings-power", "msnim",
    "msrp", "msrps", "mtqp", "mumble", "mupdate", "mvn", "news", "nfs", "ni", "nih",
    "nntp", "notes", "oid", "paquelocktoken", "pack", "palm", "paparazzi", "pkcs11",
    "platform", "pop", "pres", "prospero", "proxy", "psyc", "query", "reload", "res",
    "resource", "rmi", "rsync", "rtmp", "rtsp", "secondlife", "service", "session", "sftp",
    "sgn", "shttp", "sieve", "sip", "sips", "skype", "smb", "sms", "snews", "snmp",
    "soap.beep", "soap.beeps", "soldat", "spotify", "ssh", "steam", "svn", "tag",
    "teamspeak", "tel", "telnet", "tftp", "things", "thismessage", "tn3270", "tip", "tv",
    "udp", "unreal", "urn", "ut2004", "vemmi", "ventrilo", "videotex", "view-source",
    "wais", "webcal", "ws", "wss", "wtai", "wyciwyg", "xcon", "xcon-userid", "xfire",
    "xmlrpc.beep", "xmlrpc.beeps", "xmpp", "xri", "ymsgr",
    // 非官方方案
    "javascript", "jdbc", "doi"
];

/// URL数据结构，定义了解析后的URL各个部分
#[derive(Debug, Clone)]
pub struct Url {
    pub href: String,
    pub protocol: String,
    pub host: String,
    pub auth: Option<String>,
    pub hostname: String,
    pub pathname: String,
    pub search: String,
    pub path: String,
    pub hash: String,
    pub query: String,
    pub port: Option<String>,
}

impl Url {
    /// 解析URL字符串
    pub fn parse(url: &str) -> Option<Self> {
        let href = url.to_string();
        let protocol = Self::get_protocol(url)?;
        let is_ssh = Self::is_ssh(&protocol);
        
        // 获取认证信息和主机名
        let (auth, hostname) = Self::extract_auth_and_hostname(url, &protocol)?;
        
        // 获取主机（不含端口）
        let host = hostname.split(':').next()?.to_string();
        
        // 获取路径相关信息
        let path = Self::get_path(url).unwrap_or_default();
        let pathname = Self::get_pathname(&path).unwrap_or_default();
        let search = Self::get_search(&path).unwrap_or_default();
        let query = if search.starts_with('?') {
            search[1..].to_string()
        } else {
            String::new()
        };
        let hash = Self::get_hash(&path).unwrap_or_default();
        
        // 获取端口
        let port = hostname.split(':').nth(1).map(String::from);
        
        Some(Self {
            href,
            protocol,
            host,
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

    /// 检查字符串是否是有效的协议
    pub fn is_protocol(s: &str) -> bool {
        URL_SCHEMES.contains(&s)
    }

    /// 检查是否是SSH协议
    pub fn is_ssh(s: &str) -> bool {
        s == "ssh" || s == "git"
    }

    /// 获取URL的协议部分
    pub fn get_protocol(url: &str) -> Option<String> {
        let parts: Vec<&str> = url.splitn(2, "://").collect();
        if parts.len() != 2 {
            return None;
        }
        let protocol = parts[0].to_lowercase();
        if Self::is_protocol(&protocol) {
            Some(protocol)
        } else {
            None
        }
    }

    /// 提取认证信息和主机名
    fn extract_auth_and_hostname(url: &str, protocol: &str) -> Option<(Option<String>, String)> {
        let after_protocol = url.split(&format!("{}://", protocol)).nth(1)?;
        let parts: Vec<&str> = after_protocol.splitn(2, '@').collect();
        
        match parts.len() {
            2 => Some((Some(parts[0].to_string()), parts[1].split('/').next()?.to_string())),
            1 => Some((None, parts[0].split('/').next()?.to_string())),
            _ => None,
        }
    }

    /// 获取路径
    pub fn get_path(url: &str) -> Option<String> {
        let protocol = Self::get_protocol(url)?;
        let after_protocol = url.split(&format!("{}://", protocol)).nth(1)?;
        let after_auth = after_protocol.split('@').last()?;
        
        if Self::is_ssh(&protocol) {
            after_auth.find(':').map(|i| after_auth[i..].to_string())
        } else {
            match after_auth.find('/') {
                Some(i) => Some(after_auth[i..].to_string()),
                None => Some(String::from("/"))
            }
        }
    }

    /// 获取路径名
    fn get_pathname(path: &str) -> Option<String> {
        Some(path.split('?').next()?.to_string())
    }

    /// 获取搜索字符串
    fn get_search(path: &str) -> Option<String> {
        if let Some(query_start) = path.find('?') {
            if let Some(hash_start) = path.find('#') {
                Some(path[query_start..hash_start].to_string())
            } else {
                Some(path[query_start..].to_string())
            }
        } else {
            Some(String::new())
        }
    }

    /// 获取哈希部分
    fn get_hash(path: &str) -> Option<String> {
        path.split('#').nth(1).map(|h| format!("#{}", h)).or(Some(String::new()))
    }

    /// 打印URL信息
    pub fn inspect(&self) {
        println!("#url =>");
        println!("    .href: \"{}\"", self.href);
        println!("    .protocol: \"{}\"", self.protocol);
        println!("    .host: \"{}\"", self.host);
        println!("    .auth: \"{}\"", self.auth.as_deref().unwrap_or(""));
        println!("    .hostname: \"{}\"", self.hostname);
        println!("    .pathname: \"{}\"", self.pathname);
        println!("    .search: \"{}\"", self.search);
        println!("    .path: \"{}\"", self.path);
        println!("    .hash: \"{}\"", self.hash);
        println!("    .query: \"{}\"", self.query);
        println!("    .port: \"{}\"", self.port.as_deref().unwrap_or(""));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_url() {
        let url = "https://example.com/path";
        let parsed = Url::parse(url).unwrap();
        assert_eq!(parsed.protocol, "https");
        assert_eq!(parsed.host, "example.com");
        assert_eq!(parsed.pathname, "/path");
    }

    #[test]
    fn test_parse_complex_url() {
        let url = "https://user:pass@example.com:8080/path?query=value#hash";
        let parsed = Url::parse(url).unwrap();
        assert_eq!(parsed.protocol, "https");
        assert_eq!(parsed.auth.unwrap(), "user:pass");
        assert_eq!(parsed.host, "example.com");
        assert_eq!(parsed.port.unwrap(), "8080");
        assert_eq!(parsed.pathname, "/path");
        assert_eq!(parsed.query, "query=value");
        assert_eq!(parsed.hash, "#hash");
    }

    #[test]
    fn test_ssh_url() {
        let url = "git://user@github.com:organization/repo.git";
        let parsed = Url::parse(url).unwrap();
        assert_eq!(parsed.protocol, "git");
        assert_eq!(parsed.auth.unwrap(), "user");
        assert_eq!(parsed.host, "github.com");
        assert_eq!(parsed.path, ":organization/repo.git");
    }

    #[test]
    fn test_url_without_path() {
        let url = "http://example.com";
        let parsed = Url::parse(url).unwrap();
        assert_eq!(parsed.pathname, "/");
        assert_eq!(parsed.search, "");
        assert_eq!(parsed.hash, "");
    }
}